#!/usr/bin/env python3
"""
Sweep XGB multihead v2 thresholds with Optuna.

EXPLORE lane, read-only to GX1_DATA.
Writes results under: reports/xgb_sweeps/SWEEP_<UTC>/.

Modes:
- single (default): one study; Optuna chooses session unless --session is set.
- separate_sessions: sequential EU/OVERLAP/US (or chosen --sessions) studies in one run,
  writing study_{S}.db, trials_{S}.csv, best_{S}.json, README_{S}.md plus a top README.md.
  When --parallel-sessions=1, the three studies run in parallel processes (1 job each by default).
Label cache: per (year, session, horizon) NPZ stored under cache/ to reuse tape MFE/MAE windows.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd

from gx1.time.session_detector import get_session_vectorized
from gx1.xgb.multihead.xgb_multihead_model_v1 import XGBMultiheadModel
from gx1.xgb.preprocess.xgb_input_sanitizer import XGBInputSanitizer


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
GX1_DATA_ROOT = Path(os.environ.get("GX1_DATA_ROOT", "/home/andre2/GX1_DATA"))
FROZEN_FINAL_DECISION_PATH = Path("/home/andre2/src/GX1_ENGINE/reports/xgb_head_final/FINAL_20260225_115828/FINAL_HEAD_DECISION.json")
BUCKET_POLICY_V2 = {
    "note": "Applied only for H=24; allow_buckets if present takes precedence; suppress_buckets always applied.",
    "allow_vol": {
        "EU": {
            "LONG": [2, 3],  # allow vol2/vol3
            "SHORT": [1],    # allow vol1
        },
        "OVERLAP": {
            "LONG": [0, 1],  # allow vol0/vol1
            "SHORT": [0],    # allow vol0
        },
    },
    "allow_buckets": {
        "EU": {
            "LONG": ["tr1_vol0_sp1", "tr1_vol3_sp1"],
            "SHORT": ["tr1_vol0_sp1", "tr1_vol3_sp1"],
        },
        "OVERLAP": {
            "LONG": ["tr1_vol0_sp1", "tr1_vol1_sp1"],
            "SHORT": ["tr1_vol0_sp1", "tr1_vol1_sp1"],
        },
    },
    "suppress_buckets": {
        "EU": {
            "LONG": ["tr1_vol0_sp1", "tr1_vol1_sp1", "tr1_vol2_sp1", "tr1_vol3_sp1"],
            "SHORT": ["tr1_vol0_sp1", "tr1_vol1_sp1", "tr1_vol2_sp1", "tr1_vol3_sp1"],
        },
        "OVERLAP": {
            "LONG": ["tr1_vol0_sp1", "tr1_vol1_sp1", "tr1_vol2_sp1", "tr1_vol3_sp1"],
            "SHORT": ["tr1_vol0_sp1", "tr1_vol1_sp1", "tr1_vol2_sp1", "tr1_vol3_sp1"],
        },
    },
}

# Hardcoded canonical paths (SSoT-style)
BUNDLE_DIR = Path("/home/andre2/GX1_DATA/models/models/xgb_universal_multihead_v2__CANONICAL")
BASE28_MANIFEST = Path("/home/andre2/GX1_DATA/data/data/prebuilt/BASE28_CANONICAL/CURRENT_MANIFEST.json")
TAPE_ROOT = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL")

REQUIRED_TAPE_COLS = ["time", "high", "low", "close"]

SESSIONS = ("EU", "OVERLAP", "US")
HORIZONS = (6, 24)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest() -> Dict[str, Any]:
    if not BASE28_MANIFEST.exists():
        raise FileNotFoundError(f"BASE28 manifest not found: {BASE28_MANIFEST}")
    return _read_json(BASE28_MANIFEST)


def _resolve_base28_parquet() -> Path:
    m = _load_manifest()
    parquet_rel = m.get("parquet_path") or m.get("parquet") or m.get("path")
    if not parquet_rel:
        raise RuntimeError("MANIFEST missing parquet_path")
    p = Path(parquet_rel)
    if not p.is_absolute():
        p = BASE28_MANIFEST.parent / p
    if not p.exists():
        raise FileNotFoundError(f"BASE28 parquet missing: {p}")
    return p.resolve()


def _load_base28() -> pd.DataFrame:
    """
    Load BASE28 parquet and enforce a timezone-aware UTC DatetimeIndex.

    Notes:
    - BASE28_CANONICAL stores `time` as an explicit column (UTC).
    - We accept either a `time` column or an index, but normalize to index=UTC.
    """
    df = pd.read_parquet(_resolve_base28_parquet())

    if "time" in df.columns:
        ts = pd.to_datetime(df["time"], utc=True, errors="raise")
    else:
        ts = pd.to_datetime(df.index, utc=True, errors="raise")

    if not isinstance(ts, pd.DatetimeIndex):
        ts = pd.DatetimeIndex(ts)

    if ts.tz is None or str(ts.tz) != "UTC":
        raise RuntimeError("BASE28 time not UTC")

    if ts.has_duplicates:
        raise RuntimeError("BASE28 duplicate timestamps")

    # Deterministic sort if needed
    if not ts.is_monotonic_increasing:
        order = np.argsort(ts.values)
        df = df.iloc[order].copy()
        ts = pd.DatetimeIndex(ts.values[order], tz="UTC")

    df = df.copy()
    df.index = ts
    if "time" in df.columns:
        df = df.drop(columns=["time"])
    return df


def _load_tape_year(year: int) -> pd.DataFrame:
    """
    Load canonical market tape for a given year and validate.
    """
    path = TAPE_ROOT / f"year={year}" / "part-000.parquet"
    if not path.exists():
        raise FileNotFoundError(f"TAPE_NOT_FOUND: {path}")

    df = pd.read_parquet(path, columns=REQUIRED_TAPE_COLS)

    if "time" not in df.columns:
        raise RuntimeError(f"TAPE_MISSING_TIME: {path}")

    # Robust tz-aware UTC check
    ts = pd.to_datetime(df["time"], utc=True, errors="raise")
    if not isinstance(ts, pd.DatetimeIndex):
        ts = pd.DatetimeIndex(ts)
    if ts.tz is None or str(ts.tz) != "UTC":
        raise RuntimeError(f"TAPE_TIME_NOT_UTC: {path}")

    if ts.isnull().any():
        raise RuntimeError(f"TAPE_TIME_HAS_NULL: {path}")
    if ts.duplicated().any():
        raise RuntimeError(f"TAPE_TIME_HAS_DUPLICATES: {path}")
    if not ts.is_monotonic_increasing:
        raise RuntimeError(f"TAPE_TIME_NOT_SORTED: {path}")

    df = df.copy()
    df["time"] = ts

    for col in ["high", "low", "close"]:
        if col not in df.columns:
            raise RuntimeError(f"TAPE_COL_MISSING: {col} {path}")
        if df[col].isnull().any():
            raise RuntimeError(f"TAPE_COL_HAS_NULL: {col} {path}")
        if np.isinf(df[col].to_numpy(dtype=float)).any():
            raise RuntimeError(f"TAPE_COL_HAS_INF: {col} {path}")

    return df


def _load_model_and_contracts(bundle_dir: Path) -> Tuple[XGBMultiheadModel, Dict[str, Any], XGBInputSanitizer]:
    model_path = bundle_dir / "xgb_universal_multihead_v2.joblib"
    feature_contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_base28_v1.json"
    sanitizer_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_base28_v1.json"
    output_contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_multihead_output_contract_v1.json"

    for p in [model_path, feature_contract_path, sanitizer_path, output_contract_path]:
        if not p.exists():
            raise FileNotFoundError(f"REQUIRED_ARTIFACT_MISSING: {p}")

    model = XGBMultiheadModel.load(model_path)
    feature_contract = _read_json(feature_contract_path)
    sanitizer = XGBInputSanitizer.from_config(str(sanitizer_path))

    features = feature_contract.get("features") or feature_contract.get("ordered_features")
    if not features or len(features) != 28:
        raise RuntimeError("FEATURE_CONTRACT_INVALID")

    return model, {"features": features}, sanitizer


def _load_normalizer(bundle_dir: Path, ref_year: int) -> Optional[Dict[str, Any]]:
    path = bundle_dir / f"XGB_MULTIHEAD_V2_DRIFT_NORMALIZER_REF{ref_year}.json"
    if not path.exists():
        return None
    return _read_json(path)


def _apply_normalizer(
    prob_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray],
    norm_obj: Dict[str, Any],
    session: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply drift-normalizer mapping for a given session.
    """
    p_long, p_short, p_flat = prob_tuple
    heads = norm_obj.get("heads", {})
    if session not in heads:
        raise RuntimeError(f"DRIFT_NORMALIZER_MISSING_SESSION: {session}")

    ref = heads[session]
    ref_q = np.asarray(ref.get("quantiles"), dtype=float)
    ref_a = np.asarray(ref.get("a_values"), dtype=float)
    ref_b = np.asarray(ref.get("b_values"), dtype=float)

    if ref_q.size == 0 or ref_a.size != ref_q.size or ref_b.size != ref_q.size:
        raise RuntimeError(f"DRIFT_NORMALIZER_INVALID: session={session}")

    eps = 1e-12
    a_logit = np.log(np.clip(p_short, eps, 1.0) / np.clip(p_long, eps, 1.0))
    b_logit = np.log(np.clip(p_flat, eps, 1.0) / np.clip(p_long, eps, 1.0))

    # Map via empirical CDF -> ref grid
    sorted_a = np.sort(a_logit)
    sorted_b = np.sort(b_logit)
    denom_a = max(len(sorted_a) - 1, 1)
    denom_b = max(len(sorted_b) - 1, 1)

    pct_a = np.searchsorted(sorted_a, a_logit, side="left") / denom_a
    pct_b = np.searchsorted(sorted_b, b_logit, side="left") / denom_b

    a_mapped = np.interp(pct_a, ref_q, ref_a)
    b_mapped = np.interp(pct_b, ref_q, ref_b)

    exp_a = np.exp(a_mapped)
    exp_b = np.exp(b_mapped)
    denom = 1.0 + exp_a + exp_b

    # softmax reconstruct: long, short, flat
    return 1.0 / denom, exp_a / denom, exp_b / denom


def _build_label_cache(
    *,
    year: int,
    session: str,
    horizon: int,
    df_sess_index: pd.Index,
    prices: pd.DataFrame,
    cache_path: Path,
) -> Dict[str, np.ndarray]:
    closes = prices["close"].to_numpy(dtype=float)
    highs = prices["high"].to_numpy(dtype=float)
    lows = prices["low"].to_numpy(dtype=float)
    n = len(closes)
    max_future_high = np.full(n, np.nan, dtype=float)
    min_future_low = np.full(n, np.nan, dtype=float)
    valid = np.zeros(n, dtype=bool)

    h = int(horizon)
    # Simple O(n*h) since h is small (6/24)
    for i in range(n):
        j = i + h
        if j >= n:
            continue
        window_hi = highs[i + 1 : j + 1]
        window_lo = lows[i + 1 : j + 1]
        if window_hi.size == h and window_lo.size == h:
            max_future_high[i] = float(np.max(window_hi))
            min_future_low[i] = float(np.min(window_lo))
            valid[i] = True

    # Persist cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        index_time=df_sess_index.view("int64"),
        close=closes,
        max_future_high=max_future_high,
        min_future_low=min_future_low,
        valid=valid,
    )
    return {
        "close": closes,
        "max_future_high": max_future_high,
        "min_future_low": min_future_low,
        "valid": valid,
    }


def _get_label_cache_entry(
    *,
    year: int,
    session: str,
    horizon: int,
    df_sess_index: pd.Index,
    prices: pd.DataFrame,
    cache_root: Path,
    label_cache: Dict[Tuple[int, str, int], Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    key = (year, session, int(horizon))
    if key in label_cache:
        return label_cache[key]

    cache_path = cache_root / f"labels_year={year}_session={session}_h={int(horizon)}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        entry = {
            "close": data["close"],
            "max_future_high": data["max_future_high"],
            "min_future_low": data["min_future_low"],
            "valid": data["valid"],
        }
        label_cache[key] = entry
        return entry

    entry = _build_label_cache(
        year=year,
        session=session,
        horizon=int(horizon),
        df_sess_index=df_sess_index,
        prices=prices,
        cache_path=cache_path,
    )
    label_cache[key] = entry
    return entry


def _compute_year_metrics(
    *,
    year: int,
    df_year: pd.DataFrame,
    features: List[str],
    model: XGBMultiheadModel,
    sanitizer: XGBInputSanitizer,
    normalizer: Optional[Dict[str, Any]],
    apply_normalizer_flag: bool,
    session: str,
    threshold: float,
    horizon: int,
    side_mode: str,
    reference_year: int,
    label_cache: Dict[Tuple[int, str, int], Dict[str, np.ndarray]],
    cache_root: Path,
) -> Dict[str, Any]:
    """
    Compute per-year metrics for a given (session, threshold, horizon, side_mode).
    """
    if df_year.empty:
        return {
            "n_session_bars_total": 0,
            "n_signals_threshold": 0,
            "signal_rate": 0.0,
            "session_counts_all": {},
        }

    if session not in SESSIONS:
        raise RuntimeError(f"INVALID_SESSION: {session}")

    # --- Session source (SSoT): prefer session_id if present ---
    df_year = df_year.copy()

    if "session_id" in df_year.columns:
        # session_id mapping: 0=EU, 1=OVERLAP, 2=US
        sid = pd.to_numeric(df_year["session_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
        session_arr = np.full(len(sid), "UNKNOWN", dtype=object)
        session_arr[sid == 0] = "EU"
        session_arr[sid == 1] = "OVERLAP"
        session_arr[sid == 2] = "US"
        df_year["_session"] = session_arr
    else:
        # fallback: infer from timestamp
        df_year["_session"] = get_session_vectorized(df_year.index)

    session_counts_all = pd.Series(df_year["_session"]).value_counts().to_dict()

    # --- Tape join (inner on time) ---
    # Ensure we don't accidentally double-carry "time" as a data column
    if "time" in df_year.columns:
        df_year = df_year.drop(columns=["time"])

    tape_df = _load_tape_year(year)

    df_year_reset = df_year.reset_index()
    df_year_reset = df_year_reset.rename(columns={df_year_reset.columns[0]: "time"})

    joined = df_year_reset.merge(tape_df, on="time", how="inner", validate="many_to_one")
    base_hit = len(joined) / max(len(df_year_reset), 1)
    if base_hit < 0.995:
        raise RuntimeError(f"TAPE_JOIN_RATIO_TOO_LOW: base_hit={base_hit:.6f} year={year}")

    tape_prices = joined.set_index("time")[["high", "low", "close"]]

    # Sanitize features (full df_year, then select session rows)
    X, _stats = sanitizer.sanitize(df_year, features, allow_nan_fill=True, nan_fill_value=0.0)
    df_feat = pd.DataFrame(X[:, : len(features)], columns=features, index=df_year.index)
    df_feat["_session"] = df_year["_session"].values

    session_mask = df_feat["_session"] == session
    n_total = int(session_mask.sum())
    if n_total == 0:
        return {
            "n_session_bars_total": 0,
            "n_signals_threshold": 0,
            "signal_rate": 0.0,
            "session_counts_all": session_counts_all,
        }

    df_sess = df_feat.loc[session_mask, features]

    # Predict (session head)
    outputs = model.predict_proba(df_sess, session, features)
    p_long = np.asarray(outputs.p_long, dtype=float)
    p_short = np.asarray(outputs.p_short, dtype=float)
    p_flat = np.asarray(outputs.p_flat, dtype=float)

    if apply_normalizer_flag:
        if normalizer is None:
            raise RuntimeError("DRIFT_NORMALIZER_NOT_LOADED")
        p_long, p_short, p_flat = _apply_normalizer((p_long, p_short, p_flat), normalizer, session)

    # Sanity: finite + sums to ~1
    if not (np.isfinite(p_long).all() and np.isfinite(p_short).all() and np.isfinite(p_flat).all()):
        raise RuntimeError("PROBS_NON_FINITE")
    s = p_long + p_short + p_flat
    if not np.all(np.abs(s - 1.0) < 1e-3):
        raise RuntimeError(f"PROBS_NOT_SUM_TO_1: min={float(s.min()):.6f} max={float(s.max()):.6f}")

    # Align prices to session index (strict)
    prices = tape_prices.reindex(df_sess.index)
    if prices.isnull().any().any():
        raise RuntimeError("TAPE_PRICE_NAN")

    # Signal filter
    max_side = np.maximum(p_long, p_short)

    # Reference-year guard: if threshold is above p99, treat as no-signal (avoid nonsense optima)
    p99 = float(np.percentile(max_side, 99))
    if year == reference_year and threshold >= 0.60 and p99 < threshold:
        return {
            "n_session_bars_total": n_total,
            "n_signals_threshold": 0,
            "signal_rate": 0.0,
            "n_skipped_no_horizon": 0,
            "n_long": 0,
            "n_short": 0,
            "edge_year": -1e6,
            "p99_max_side": p99,
            "session_counts_all": session_counts_all,
        }

    sig_mask = max_side >= threshold
    n_sig = int(sig_mask.sum())
    signal_rate = float(n_sig / n_total)

    if n_sig == 0:
        return {
            "n_session_bars_total": n_total,
            "n_signals_threshold": 0,
            "signal_rate": signal_rate,
            "n_skipped_no_horizon": 0,
            "n_long": 0,
            "n_short": 0,
            "edge_year": -1e6,
            "p99_max_side": p99,
            "session_counts_all": session_counts_all,
        }

    horizon_int = int(horizon)
    cache_entry = _get_label_cache_entry(
        year=year,
        session=session,
        horizon=horizon_int,
        df_sess_index=df_sess.index,
        prices=prices,
        cache_root=cache_root,
        label_cache=label_cache,
    )

    valid_all = cache_entry["valid"]
    close_arr = cache_entry["close"]
    hi_arr = cache_entry["max_future_high"]
    lo_arr = cache_entry["min_future_low"]

    sig_idx = sig_mask
    valid_sig = valid_all[sig_idx]
    n_skipped_no_horizon = int((~valid_sig).sum())

    sig_valid_mask = sig_idx & valid_all
    if sig_valid_mask.sum() == 0:
        return {
            "n_session_bars_total": n_total,
            "n_signals_threshold": n_sig,
            "signal_rate": signal_rate,
            "n_skipped_no_horizon": n_skipped_no_horizon,
            "n_long": 0,
            "n_short": 0,
            "edge_year": -1e6,
            "p99_max_side": p99,
            "session_counts_all": session_counts_all,
        }

    c0 = close_arr[sig_valid_mask]
    hi = hi_arr[sig_valid_mask]
    lo = lo_arr[sig_valid_mask]
    side_long_mask = p_long[sig_valid_mask] >= p_short[sig_valid_mask]

    long_mfe: List[float] = []
    long_mae: List[float] = []
    short_mfe: List[float] = []
    short_mae: List[float] = []

    if side_mode in ("both", "long_only"):
        lmfe = ((hi[side_long_mask] - c0[side_long_mask]) / c0[side_long_mask]) * 10000.0
        lmae = ((c0[side_long_mask] - lo[side_long_mask]) / c0[side_long_mask]) * 10000.0
        long_mfe = lmfe.tolist()
        long_mae = lmae.tolist()

    if side_mode in ("both", "short_only"):
        smfe = ((c0[~side_long_mask] - lo[~side_long_mask]) / c0[~side_long_mask]) * 10000.0
        smae = ((hi[~side_long_mask] - c0[~side_long_mask]) / c0[~side_long_mask]) * 10000.0
        short_mfe = smfe.tolist()
        short_mae = smae.tolist()

    def _agg(arr: List[float]) -> Tuple[float, float]:
        if not arr:
            return 0.0, 0.0
        a = np.asarray(arr, dtype=float)
        return float(np.percentile(a, 50)), float(np.percentile(a, 90))

    lmfe50, lmfe90 = _agg(long_mfe)
    lmae50, lmae90 = _agg(long_mae)
    smfe50, smfe90 = _agg(short_mfe)
    smae50, smae90 = _agg(short_mae)

    n_long = len(long_mfe)
    n_short = len(short_mfe)

    # Edge per year: emphasize median edge, lightly tail edge; symmetric penalty weight
    edges: List[float] = []
    if n_long:
        edges.append((lmfe50 - 1.0 * lmae50) + 0.15 * (lmfe90 - lmae90))
    if n_short:
        edges.append((smfe50 - 1.0 * smae50) + 0.15 * (smfe90 - smae90))
    edge_year = float(np.mean(edges)) if edges else -1e6

    return {
        "n_session_bars_total": n_total,
        "n_signals_threshold": n_sig,
        "signal_rate": signal_rate,
        "n_skipped_no_horizon": n_skipped_no_horizon,
        "n_long": n_long,
        "n_short": n_short,
        "lmfe_p50": lmfe50,
        "lmfe_p90": lmfe90,
        "lmae_p50": lmae50,
        "lmae_p90": lmae90,
        "smfe_p50": smfe50,
        "smfe_p90": smfe90,
        "smae_p50": smae50,
        "smae_p90": smae90,
        "edge_year": edge_year,
        "p99_max_side": p99,
        "session_counts_all": session_counts_all,
    }


def _score_trial(per_year: Dict[int, Dict[str, Any]]) -> float:
    scores: List[float] = []
    insufficient_years = 0
    penalty = 0.0

    for _y, m in per_year.items():
        n_sig = int(m.get("n_signals_threshold", 0))
        rate = float(m.get("signal_rate", 0.0))

        if n_sig < 50:
            insufficient_years += 1
            penalty += 200.0
            continue

        if rate < 0.005:
            penalty += 50.0
        if rate > 0.10:
            penalty += 20.0

        skipped = int(m.get("n_skipped_no_horizon", 0))
        if skipped / max(n_sig, 1) > 0.01:
            penalty += 10.0

        # Side balance penalty (degenerate skew) when both sides present
        n_long = int(m.get("n_long", 0))
        n_short = int(m.get("n_short", 0))
        if n_long > 0 and n_short > 0 and min(n_long, n_short) < 50:
            penalty += 150.0

        # Tail-risk penalty on MAE
        mae_pen = 0.0
        for mae_p50, mae_p90 in [
            (m.get("lmae_p50"), m.get("lmae_p90")),
            (m.get("smae_p50"), m.get("smae_p90")),
        ]:
            if mae_p90 is not None:
                mae_pen += 0.05 * max(0.0, float(mae_p90) - 120.0)
            if mae_p50 is not None:
                mae_pen += 0.02 * max(0.0, float(mae_p50) - 40.0)
        penalty += mae_pen

        scores.append(float(m.get("edge_year", -1e6)))

    if insufficient_years > 2:
        penalty += 1000.0

    if not scores:
        return float(-1e6 - penalty)

    return float(np.mean(scores) - penalty)


def _make_storage(db_path: Path) -> optuna.storages.RDBStorage:
    """
    SQLite storage with a generous timeout to reduce 'database is locked' issues for n_jobs > 1.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite:///{db_path}"
    storage = optuna.storages.RDBStorage(
        url=url,
        engine_kwargs={
            "connect_args": {"timeout": 120},
            "pool_pre_ping": True,
        },
        skip_compatibility_check=True,
    )
    return storage


def _run_one_study(
    *,
    out_root: Path,
    years: List[int],
    reference_year: int,
    trials: int,
    jobs: int,
    timeout: Optional[int],
    seed: int,
    use_drift_normalizer: bool,
    base_df: Optional[pd.DataFrame],
    model: Optional[XGBMultiheadModel],
    features: Optional[List[str]],
    sanitizer: Optional[XGBInputSanitizer],
    normalizer: Optional[Dict[str, Any]],
    fixed_session: Optional[str],
    study_suffix: str,
    run_mode: str,
    side_mode_fixed: Optional[str],
) -> Dict[str, Any]:
    """
    Run a single Optuna study.

    If fixed_session is provided, session is locked; otherwise Optuna can choose session.
    """
    if fixed_session is not None and fixed_session not in SESSIONS:
        raise RuntimeError(f"INVALID_FIXED_SESSION: {fixed_session}")

    # Lazy-load heavy artifacts inside the process (helps parallel sessions)
    if base_df is None:
        base_df = _load_base28()
    if model is None or features is None or sanitizer is None:
        model, contracts, sanitizer = _load_model_and_contracts(BUNDLE_DIR)
        features = contracts["features"]
    if use_drift_normalizer and normalizer is None:
        normalizer = _load_normalizer(BUNDLE_DIR, reference_year)
        if normalizer is None:
            raise RuntimeError(f"DRIFT_NORMALIZER_REQUIRED_BUT_MISSING: ref_year={reference_year}")

    cache_root = out_root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    label_cache: Dict[Tuple[int, str, int], Dict[str, np.ndarray]] = {}

    study_db = out_root / f"study{study_suffix}.db"
    storage = _make_storage(study_db)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study_name = f"xgb_v2_sweep{study_suffix}"
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        # deterministic seed per trial
        t_seed = seed + int(trial.number)
        np.random.seed(t_seed)
        random.seed(t_seed)
        os.environ["PYTHONHASHSEED"] = str(t_seed)

        threshold = float(trial.suggest_float("threshold", 0.55, 0.75, step=0.005))
        horizon = int(trial.suggest_categorical("horizon", [6, 24]))
        if side_mode_fixed is not None:
            side_mode = side_mode_fixed
        else:
            side_mode = str(trial.suggest_categorical("side_mode", ["both", "long_only", "short_only"]))

        if fixed_session is not None:
            session = fixed_session
        else:
            session = str(trial.suggest_categorical("session", list(SESSIONS)))

        per_year: Dict[int, Dict[str, Any]] = {}
        for y in years:
            df_year = base_df.loc[base_df.index.year == y]
            if df_year.empty:
                continue

            metrics = _compute_year_metrics(
                year=int(y),
                df_year=df_year,
                features=features,
                model=model,
                sanitizer=sanitizer,
                normalizer=normalizer,
                apply_normalizer_flag=bool(use_drift_normalizer),
                session=session,
                threshold=threshold,
                horizon=horizon,
                side_mode=side_mode,
                reference_year=int(reference_year),
                label_cache=label_cache,
                cache_root=cache_root,
            )
            per_year[int(y)] = metrics

        if trial.number == 0:
            for yy, m in per_year.items():
                sc = m.get("session_counts_all")
                if sc is not None:
                    print(f"[session_counts] session={session} year={yy} counts={sc}")

        score = _score_trial(per_year)
        trial.set_user_attr("session", session)
        trial.set_user_attr("per_year", per_year)
        return float(score)

    study.optimize(objective, n_trials=int(trials), n_jobs=int(jobs), timeout=timeout)

    best = study.best_trial
    best_out = {
        "value": float(best.value) if best.value is not None else None,
        "params": best.params,
        "user_attrs": best.user_attrs,
        "mode": run_mode,
        "session": fixed_session or "OPTUNA",
        "years": [int(y) for y in years],
        "reference_year": int(reference_year),
        "use_drift_normalizer": bool(use_drift_normalizer),
        "side_mode_fixed": side_mode_fixed,
        "git_hash": os.popen("cd /home/andre2/src/GX1_ENGINE && git rev-parse HEAD").read().strip(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Output names
    if fixed_session is not None:
        best_path = out_root / f"best_{fixed_session}.json"
        trials_path = out_root / f"trials_{fixed_session}.csv"
        readme_path = out_root / f"README_{fixed_session}.md"
    else:
        best_path = out_root / "best.json"
        trials_path = out_root / "trials.csv"
        readme_path = out_root / "README.md"

    best_path.write_text(json.dumps(best_out, indent=2), encoding="utf-8")

    # Trials CSV (lightweight)
    rows: List[Dict[str, Any]] = []
    for t in study.trials:
        row: Dict[str, Any] = {
            "number": int(t.number),
            "state": str(t.state),
            "value": float(t.value) if t.value is not None else None,
        }
        row.update({f"param_{k}": v for k, v in t.params.items()})
        # include resolved session for locked-mode studies too
        sess = t.user_attrs.get("session")
        if sess is not None:
            row["session"] = str(sess)
        rows.append(row)
    pd.DataFrame(rows).to_csv(trials_path, index=False)

    # README
    git_hash = os.popen("cd /home/andre2/src/GX1_ENGINE && git rev-parse HEAD").read().strip()
    cmd_parts = [
        "/home/andre2/venvs/gx1/bin/python",
        "gx1/scripts/sweep_xgb_multihead_v2_optuna.py",
        "--years",
        *[str(y) for y in years],
        "--reference-year",
        str(reference_year),
        "--trials",
        str(trials),
        "--jobs",
        str(jobs),
        "--use-drift-normalizer",
        "1" if use_drift_normalizer else "0",
        "--mode",
        run_mode,
    ]
    if run_mode == "single" and fixed_session is not None:
        cmd_parts += ["--session", fixed_session]
    if side_mode_fixed is not None:
        cmd_parts += ["--side-mode-fixed", side_mode_fixed]
    if run_mode == "separate_sessions":
        cmd_parts += ["--sessions", fixed_session]  # fixed_session is the current session here
    readme = f"""# XGB multihead v2 sweep

Mode:
  mode = {run_mode}
  session = {fixed_session if fixed_session is not None else "optuna"}
  side_mode_fixed = {side_mode_fixed or "optuna"}

Command:
  {' '.join(cmd_parts)}

Bundle: {BUNDLE_DIR}
BASE28 manifest: {BASE28_MANIFEST}
Tape root: {TAPE_ROOT}
Study DB: {study_db}
Git commit: {git_hash}

Best:
  value = {best_out['value']}
  params = {best_out['params']}
Cache:
  label cache dir = {cache_root}
"""
    readme_path.write_text(readme, encoding="utf-8")

    return {
        "study_path": str(study_db),
        "best": best_out,
        "trials_csv": str(trials_path),
        "session": fixed_session or "OPTUNA",
    }


def run_sweep(args: argparse.Namespace) -> Dict[str, Any]:
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    years = [int(y) for y in args.years]
    reference_year = int(args.reference_year)

    use_norm = bool(int(args.use_drift_normalizer))
    normalizer = _load_normalizer(BUNDLE_DIR, reference_year) if use_norm else None
    if use_norm and normalizer is None:
        raise RuntimeError(f"DRIFT_NORMALIZER_REQUIRED_BUT_MISSING: ref_year={reference_year}")

    if args.mode == "single":
        # Single study can use intra-study parallelism
        base_df = _load_base28()
        model, contracts, sanitizer = _load_model_and_contracts(BUNDLE_DIR)
        features = contracts["features"]
        fixed_session: Optional[str] = args.session
        suffix = "" if fixed_session is None else f"_{fixed_session}"
        r = _run_one_study(
            out_root=out_root,
            years=years,
            reference_year=reference_year,
            trials=int(args.trials),
            jobs=int(args.jobs),
            timeout=args.timeout,
            seed=int(args.seed),
            use_drift_normalizer=use_norm,
            base_df=base_df,
            model=model,
            features=features,
            sanitizer=sanitizer,
            normalizer=normalizer,
            fixed_session=fixed_session,
            study_suffix=suffix,
            run_mode="single",
            side_mode_fixed=args.side_mode_fixed,
        )
        return {"mode": "single", "out_root": str(out_root), "studies": {fixed_session or "OPTUNA": r}}

    # separate_sessions
    results: Dict[str, Any] = {}
    parallel_sessions = bool(int(args.parallel_sessions))

    if parallel_sessions:
        jobs_per_study = int(args.jobs_per_study)
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(args.sessions)) as executor:
            future_map = {}
            for sess in args.sessions:
                suffix = f"_{sess}"
                fut = executor.submit(
                    _run_one_study,
                    out_root=out_root,
                    years=years,
                    reference_year=reference_year,
                    trials=int(args.trials),
                    jobs=jobs_per_study,
                    timeout=args.timeout,
                    seed=int(args.seed),
                    use_drift_normalizer=use_norm,
                    base_df=None,
                    model=None,
                    features=None,
                    sanitizer=None,
                    normalizer=None,
                    fixed_session=sess,
                    study_suffix=suffix,
                    run_mode="separate_sessions",
                    side_mode_fixed=args.side_mode_fixed,
                )
                future_map[fut] = sess
            for fut, sess in future_map.items():
                results[sess] = fut.result()
    else:
        base_df = _load_base28()
        model, contracts, sanitizer = _load_model_and_contracts(BUNDLE_DIR)
        features = contracts["features"]
        for sess in args.sessions:
            suffix = f"_{sess}"
            r = _run_one_study(
                out_root=out_root,
                years=years,
                reference_year=reference_year,
                trials=int(args.trials),
                jobs=int(args.jobs),
                timeout=args.timeout,
                seed=int(args.seed),
                use_drift_normalizer=use_norm,
                base_df=base_df,
                model=model,
                features=features,
                sanitizer=sanitizer,
                normalizer=normalizer,
                fixed_session=sess,
                study_suffix=suffix,
                run_mode="separate_sessions",
                side_mode_fixed=args.side_mode_fixed,
            )
            results[sess] = r

    # top-level README
    git_hash = os.popen("cd /home/andre2/src/GX1_ENGINE && git rev-parse HEAD").read().strip()
    cmd_parts = [
        "/home/andre2/venvs/gx1/bin/python",
        "gx1/scripts/sweep_xgb_multihead_v2_optuna.py",
        "--years",
        *[str(y) for y in years],
        "--reference-year",
        str(reference_year),
        "--trials",
        str(args.trials),
        "--jobs",
        str(args.jobs),
        "--use-drift-normalizer",
        str(args.use_drift_normalizer),
        "--mode",
        "separate_sessions",
        "--sessions",
        *args.sessions,
    ]
    if args.side_mode_fixed is not None:
        cmd_parts += ["--side-mode-fixed", args.side_mode_fixed]
    readme_lines = [
        f"# XGB multihead v2 sweep (mode=separate_sessions)",
        "",
        "Command:",
        f"  {' '.join(cmd_parts)}",
        "",
        f"Bundle: {BUNDLE_DIR}",
        f"BASE28 manifest: {BASE28_MANIFEST}",
        f"Tape root: {TAPE_ROOT}",
        f"Git commit: {git_hash}",
        f"Side-mode fixed: {args.side_mode_fixed}",
        f"Parallel sessions: {int(args.parallel_sessions)}",
        f"Jobs per study: {args.jobs_per_study if parallel_sessions else args.jobs}",
        "Label cache: cache/labels_year=*_session=*_h=*.npz",
        "",
        "Studies:",
    ]
    for sess in args.sessions:
        readme_lines.append(
            f"- {sess}: study={out_root / f'study_{sess}.db'}, trials={out_root / f'trials_{sess}.csv'}, best={out_root / f'best_{sess}.json'}, readme={out_root / f'README_{sess}.md'}"
        )
    (out_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    return {"mode": "separate_sessions", "out_root": str(out_root), "studies": results}


def _parse_fixed_config(raw: Optional[str]) -> Dict[str, Dict[str, Any]]:
    default_cfg = {
        "EU": {"threshold": 0.56, "horizon": 6, "side_mode": "both"},
        "OVERLAP": {"threshold": 0.55, "horizon": 6, "side_mode": "both"},
        "US": {"threshold": 0.60, "horizon": 6, "side_mode": "both"},
    }
    if raw is None:
        return default_cfg
    try_path = Path(raw)
    if try_path.exists():
        data = json.loads(try_path.read_text(encoding="utf-8"))
    else:
        data = json.loads(raw)
    merged = default_cfg.copy()
    merged.update({k: v for k, v in data.items() if k in SESSIONS})
    return merged


def _load_frozen_decision() -> Dict[str, Any]:
    if not FROZEN_FINAL_DECISION_PATH.exists():
        raise RuntimeError(f"FROZEN_DECISION_MISSING: {FROZEN_FINAL_DECISION_PATH}")
    return json.loads(FROZEN_FINAL_DECISION_PATH.read_text(encoding="utf-8"))


def _edge_v2(metrics: Dict[str, Any]) -> float:
    edges: List[float] = []
    if metrics.get("n_long", 0) > 0:
        edges.append(
            (metrics.get("lmfe_p50", 0.0) - 1.2 * metrics.get("lmae_p50", 0.0))
            + 0.10 * (metrics.get("lmfe_p90", 0.0) - 1.2 * metrics.get("lmae_p90", 0.0))
        )
    if metrics.get("n_short", 0) > 0:
        edges.append(
            (metrics.get("smfe_p50", 0.0) - 1.2 * metrics.get("smae_p50", 0.0))
            + 0.10 * (metrics.get("smfe_p90", 0.0) - 1.2 * metrics.get("smae_p90", 0.0))
        )
    return float(np.mean(edges)) if edges else -1e6


def run_final_validate(args: argparse.Namespace) -> Dict[str, Any]:
    out_root = WORKSPACE_ROOT / "reports" / "xgb_head_final" / f"FINAL_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_root.mkdir(parents=True, exist_ok=True)

    years = [int(y) for y in args.years]
    reference_year = int(args.reference_year)
    use_norm = bool(int(args.use_drift_normalizer))
    normalizer = _load_normalizer(BUNDLE_DIR, reference_year) if use_norm else None
    if use_norm and normalizer is None:
        raise RuntimeError(f"DRIFT_NORMALIZER_REQUIRED_BUT_MISSING: ref_year={reference_year}")

    base_df = _load_base28()
    model, contracts, sanitizer = _load_model_and_contracts(BUNDLE_DIR)
    features = contracts["features"]

    cache_root = out_root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    label_cache: Dict[Tuple[int, str, int], Dict[str, np.ndarray]] = {}

    fixed_cfg = _parse_fixed_config(args.fixed_config)
    rng_seed = int(args.seed)
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    os.environ["PYTHONHASHSEED"] = str(rng_seed)

    frozen = _load_frozen_decision()
    us_frozen = (
        frozen.get("sessions", {})
        .get("US", {})
        .get("fixed_config", {"threshold": None, "horizon": None, "side_mode": None})
    )
    if (
        float(us_frozen.get("threshold")) != float(fixed_cfg["US"]["threshold"])
        or int(us_frozen.get("horizon")) != int(fixed_cfg["US"]["horizon"])
        or str(us_frozen.get("side_mode")) != str(fixed_cfg["US"]["side_mode"])
    ):
        raise RuntimeError("US_POLICY_FROZEN_MISMATCH")
    print(
        f"[US_POLICY] US policy frozen @ {us_frozen['threshold']:.2f} / H{us_frozen['horizon']} / {us_frozen['side_mode']} (source: {FROZEN_FINAL_DECISION_PATH})"
    )

    per_session: Dict[str, Any] = {}
    for session in args.sessions:
        cfg = fixed_cfg.get(session, {})
        threshold = float(cfg.get("threshold"))
        horizon = int(cfg.get("horizon"))
        side_mode = str(cfg.get("side_mode"))

        per_year_metrics: Dict[int, Any] = {}
        for y in years:
            df_year = base_df.loc[base_df.index.year == y]
            if df_year.empty:
                continue
            m = _compute_year_metrics(
                year=int(y),
                df_year=df_year,
                features=features,
                model=model,
                sanitizer=sanitizer,
                normalizer=normalizer,
                apply_normalizer_flag=use_norm,
                session=session,
                threshold=threshold,
                horizon=horizon,
                side_mode=side_mode,
                reference_year=reference_year,
                label_cache=label_cache,
                cache_root=cache_root,
            )
            m["edge_v1"] = m.get("edge_year", -1e6)
            m["edge_v2"] = _edge_v2(m)
            per_year_metrics[int(y)] = m

        years_with_signals = [yy for yy, m in per_year_metrics.items() if m.get("n_signals_threshold", 0) >= 200]
        edge_v2_values = [per_year_metrics[yy]["edge_v2"] for yy in years_with_signals]
        gate_failures: List[str] = []
        if len(years_with_signals) < 4:
            gate_failures.append(f"INSUFFICIENT_YEARS_WITH_SIGNALS:{len(years_with_signals)}")
        if edge_v2_values:
            if float(np.median(edge_v2_values)) <= 0.0:
                gate_failures.append("EDGE_V2_MEDIAN_NONPOSITIVE")
        else:
            gate_failures.append("NO_EDGE_V2_VALUES")

        # Force EU/OVERLAP to signal-only per freeze intent
        status = "ACCEPT" if (not gate_failures and session == "US") else "DEFER_TO_TRANSFORMER"
        recommended_policy = cfg if status == "ACCEPT" else {"action": "use_as_signal_only"}
        if session in ("EU", "OVERLAP"):
            status = "DEFER_TO_TRANSFORMER"
            if "FORCED_SIGNAL_ONLY" not in gate_failures:
                gate_failures.append("FORCED_SIGNAL_ONLY")
            recommended_policy = {"action": "use_as_signal_only"}

        per_session[session] = {
            "fixed_config": cfg,
            "per_year": per_year_metrics,
            "years_with_signals_ge_200": years_with_signals,
            "edge_v2_median": float(np.median(edge_v2_values)) if edge_v2_values else None,
            "gate_failures": gate_failures,
            "status": status,
            "recommended_policy": recommended_policy,
        }

    decision = {
        "git_hash": os.popen("cd /home/andre2/src/GX1_ENGINE && git rev-parse HEAD").read().strip(),
        "bundle_dir": str(BUNDLE_DIR),
        "base28_manifest": str(BASE28_MANIFEST),
        "tape_root": str(TAPE_ROOT),
        "reference_year": reference_year,
        "use_drift_normalizer": use_norm,
        "sessions": per_session,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(
            [
                "/home/andre2/venvs/gx1/bin/python",
                "gx1/scripts/sweep_xgb_multihead_v2_optuna.py",
                "--mode",
                "final_validate",
                "--years",
                *[str(y) for y in years],
                "--reference-year",
                str(reference_year),
                "--use-drift-normalizer",
                str(int(use_norm)),
            ]
        ),
    }

    final_json = out_root / "FINAL_HEAD_DECISION.json"
    final_json.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    readme = f"""# FINAL HEAD DECISION (XGB multihead v2)

Command:
  {decision['command']}

Bundle: {BUNDLE_DIR}
BASE28 manifest: {BASE28_MANIFEST}
Tape root: {TAPE_ROOT}
Git commit: {decision['git_hash']}
Side configs: {fixed_cfg}
Label cache: {cache_root}
Output: {final_json}
"""
    (out_root / "README.md").write_text(readme, encoding="utf-8")

    return {"mode": "final_validate", "out_root": str(out_root), "final_json": str(final_json)}


def _compute_mfe_mae_vectors(
    p_long: np.ndarray,
    p_short: np.ndarray,
    cache_entry: Dict[str, np.ndarray],
    sig_mask: np.ndarray,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_all = cache_entry["valid"]
    close_arr = cache_entry["close"]
    hi_arr = cache_entry["max_future_high"]
    lo_arr = cache_entry["min_future_low"]

    sig_valid_mask = sig_mask & valid_all
    n_skipped = int((sig_mask & (~valid_all)).sum())
    if sig_valid_mask.sum() == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), n_skipped

    c0 = close_arr[sig_valid_mask]
    hi = hi_arr[sig_valid_mask]
    lo = lo_arr[sig_valid_mask]
    side_long_mask = p_long[sig_valid_mask] >= p_short[sig_valid_mask]

    # long
    lmfe = ((hi[side_long_mask] - c0[side_long_mask]) / c0[side_long_mask]) * 10000.0
    lmae = ((c0[side_long_mask] - lo[side_long_mask]) / c0[side_long_mask]) * 10000.0
    # short
    smfe = ((c0[~side_long_mask] - lo[~side_long_mask]) / c0[~side_long_mask]) * 10000.0
    smae = ((hi[~side_long_mask] - c0[~side_long_mask]) / c0[~side_long_mask]) * 10000.0
    return lmfe, lmae, smfe, smae, side_long_mask, n_skipped


def _diag_bucket_agg(values_mfe: np.ndarray, values_mae: np.ndarray) -> Tuple[float, float, float, float, float]:
    if values_mfe.size == 0:
        return 0.0, 0.0, 0.0, 0.0, -1e6
    mfe50 = float(np.percentile(values_mfe, 50))
    mfe90 = float(np.percentile(values_mfe, 90))
    mae50 = float(np.percentile(values_mae, 50))
    mae90 = float(np.percentile(values_mae, 90))
    edge = (mfe50 - 1.0 * mae50) + 0.15 * (mfe90 - mae90)
    return mfe50, mfe90, mae50, mae90, edge


def run_diag_buckets(args: argparse.Namespace) -> Dict[str, Any]:
    out_root = args.out_dir
    if out_root is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_root = WORKSPACE_ROOT / "reports" / "xgb_head_diag" / f"DIAG_{ts}"
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    years = [int(y) for y in args.years]
    reference_year = int(args.reference_year)
    use_norm = bool(int(args.use_drift_normalizer))
    normalizer = _load_normalizer(BUNDLE_DIR, reference_year) if use_norm else None
    if use_norm and normalizer is None:
        raise RuntimeError(f"DRIFT_NORMALIZER_REQUIRED_BUT_MISSING: ref_year={reference_year}")

    base_df = _load_base28()
    model, contracts, sanitizer = _load_model_and_contracts(BUNDLE_DIR)
    features = contracts["features"]

    sessions = ("EU", "OVERLAP")
    thresholds = [float(t) for t in args.thresholds]
    horizons = [int(h) for h in args.horizons]

    cache_root = out_root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    label_cache: Dict[Tuple[int, str, int], Dict[str, np.ndarray]] = {}

    results_raw: Dict[str, Any] = {s: {} for s in sessions}
    results_policy: Dict[str, Any] = {s: {} for s in sessions}
    eu_h24_diag: Dict[str, Dict[str, float]] = {}
    eu_h24_pos_raw = 0
    eu_h24_pos_policy = 0

    for session in sessions:
        for y in years:
            df_year = base_df.loc[base_df.index.year == y]
            if df_year.empty:
                continue

            # Session tagging
            if "session_id" in df_year.columns:
                sid = pd.to_numeric(df_year["session_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
                session_arr = np.full(len(sid), "UNKNOWN", dtype=object)
                session_arr[sid == 0] = "EU"
                session_arr[sid == 1] = "OVERLAP"
                session_arr[sid == 2] = "US"
                df_year["_session"] = session_arr
            else:
                df_year["_session"] = get_session_vectorized(df_year.index)

            if not {"trend_regime_id", "vol_regime_id", "spread_bucket"}.issubset(set(df_year.columns)):
                raise RuntimeError("BUCKET_COLUMNS_MISSING: require trend_regime_id, vol_regime_id, spread_bucket")

            if "time" in df_year.columns:
                df_year = df_year.drop(columns=["time"])

            tape_df = _load_tape_year(y)
            df_year_reset = df_year.reset_index().rename(columns={df_year.reset_index().columns[0]: "time"})
            joined = df_year_reset.merge(tape_df, on="time", how="inner", validate="many_to_one")
            base_hit = len(joined) / max(len(df_year_reset), 1)
            if base_hit < 0.995:
                raise RuntimeError(f"TAPE_JOIN_RATIO_TOO_LOW: base_hit={base_hit:.6f} year={y}")
            tape_prices = joined.set_index("time")[["high", "low", "close"]]

            X, _stats = sanitizer.sanitize(df_year, features, allow_nan_fill=True, nan_fill_value=0.0)
            df_feat = pd.DataFrame(X[:, : len(features)], columns=features, index=df_year.index)
            df_feat["_session"] = df_year["_session"].values

            session_mask = df_feat["_session"] == session
            if not session_mask.any():
                continue

            df_sess = df_feat.loc[session_mask, features]
            buckets_df = df_year.loc[session_mask, ["trend_regime_id", "vol_regime_id", "spread_bucket"]]

            outputs = model.predict_proba(df_sess, session, features)
            p_long = np.asarray(outputs.p_long, dtype=float)
            p_short = np.asarray(outputs.p_short, dtype=float)
            p_flat = np.asarray(outputs.p_flat, dtype=float)

            if use_norm:
                p_long, p_short, p_flat = _apply_normalizer((p_long, p_short, p_flat), normalizer, session)

            if not (np.isfinite(p_long).all() and np.isfinite(p_short).all() and np.isfinite(p_flat).all()):
                raise RuntimeError("PROBS_NON_FINITE")
            ssum = p_long + p_short + p_flat
            if not np.all(np.abs(ssum - 1.0) < 1e-3):
                raise RuntimeError("PROBS_NOT_SUM_TO_1")

            prices = tape_prices.reindex(df_sess.index)
            if prices.isnull().any().any():
                raise RuntimeError("TAPE_PRICE_NAN")

            p99_max_side_all = float(np.percentile(np.maximum(p_long, p_short), 99))

            year_out_raw = results_raw[session].setdefault(str(y), [])
            year_out_pol = results_policy[session].setdefault(str(y), [])
            for H in horizons:
                cache_entry = _get_label_cache_entry(
                    year=int(y),
                    session=session,
                    horizon=int(H),
                    df_sess_index=df_sess.index,
                    prices=prices,
                    cache_root=cache_root,
                    label_cache=label_cache,
                )
                for T in thresholds:
                    max_side = np.maximum(p_long, p_short)
                    sig_mask = max_side >= float(T)
                    n_sig = int(sig_mask.sum())
                    signal_rate = float(n_sig / max(len(df_sess), 1))

                    lmfe, lmae, smfe, smae, side_long_mask, n_skipped = _compute_mfe_mae_vectors(
                        p_long, p_short, cache_entry, sig_mask, H
                    )

                    summary = {
                        "threshold": float(T),
                        "horizon": int(H),
                        "n_session_bars_total": int(len(df_sess)),
                        "n_signals_threshold": n_sig,
                        "signal_rate": signal_rate,
                        "n_skipped_no_horizon": n_skipped,
                        "p99_max_side": p99_max_side_all,
                    }

                    buckets_long: Dict[str, List[float]] = {}
                    buckets_short: Dict[str, List[float]] = {}

                    if n_sig > 0 and (lmfe.size + smfe.size) > 0:
                        # bucket keys aligned to sig_valid_mask via sig_mask & valid_all inside compute
                        sig_valid_mask = sig_mask & cache_entry["valid"]
                        bucket_slice = buckets_df.loc[sig_valid_mask]
                        keys = [
                            f"tr{int(tr)}_vol{int(vol)}_sp{int(sp)}"
                            for tr, vol, sp in zip(
                                bucket_slice["trend_regime_id"].to_numpy(),
                                bucket_slice["vol_regime_id"].to_numpy(),
                                bucket_slice["spread_bucket"].to_numpy(),
                            )
                        ]
                        # split by side_long_mask
                        long_keys = np.array(keys)[side_long_mask] if lmfe.size else np.array([])
                        short_keys = np.array(keys)[~side_long_mask] if smfe.size else np.array([])

                        for k, mfe_v, mae_v in zip(long_keys, lmfe, lmae):
                            buckets_long.setdefault(k, [[], []])
                            buckets_long[k][0].append(mfe_v)
                            buckets_long[k][1].append(mae_v)
                        for k, mfe_v, mae_v in zip(short_keys, smfe, smae):
                            buckets_short.setdefault(k, [[], []])
                            buckets_short[k][0].append(mfe_v)
                            buckets_short[k][1].append(mae_v)

                    def _build_bucket_list(bmap: Dict[str, List[List[float]]]) -> Tuple[List[Dict[str, Any]], int]:
                        items = []
                        for key, (mfe_list, mae_list) in bmap.items():
                            mfe_arr = np.asarray(mfe_list, dtype=float)
                            mae_arr = np.asarray(mae_list, dtype=float)
                            mfe50, mfe90, mae50, mae90, edge = _diag_bucket_agg(mfe_arr, mae_arr)
                            items.append(
                                {
                                    "bucket_key": key,
                                    "n": int(len(mfe_arr)),
                                    "mfe_p50": mfe50,
                                    "mfe_p90": mfe90,
                                    "mae_p50": mae50,
                                    "mae_p90": mae90,
                                    "edge_v2": edge,
                                }
                            )
                        items_sorted = sorted(items, key=lambda x: x["edge_v2"])
                        if len(items_sorted) <= 60:
                            return items_sorted, 0
                        worst = items_sorted[:30]
                        best = items_sorted[-30:]
                        other_count = len(items_sorted) - 60
                        return worst + best, other_count

                    long_list, long_other = _build_bucket_list(buckets_long)
                    short_list, short_other = _build_bucket_list(buckets_short)

                    year_out_raw.append(
                        {
                            "threshold": float(T),
                            "horizon": int(H),
                            "summary": summary,
                            "buckets": {
                                "LONG": long_list,
                                "SHORT": short_list,
                                "other_count_long": long_other,
                                "other_count_short": short_other,
                            },
                        }
                    )

                    # EU-specific year diagnostics at fixed H=24, T=0.56 (raw, no policy effect)
                    if session == "EU" and H == 24 and abs(float(T) - 0.56) < 1e-9:
                        raw_mfe_all = np.concatenate([lmfe, smfe]) if (lmfe.size + smfe.size) else np.array([], dtype=float)
                        raw_mae_all = np.concatenate([lmae, smae]) if (lmae.size + smae.size) else np.array([], dtype=float)
                        if raw_mfe_all.size > 0:
                            mfe50 = float(np.percentile(raw_mfe_all, 50))
                            mfe90 = float(np.percentile(raw_mfe_all, 90))
                            mae50 = float(np.percentile(raw_mae_all, 50))
                            mae90 = float(np.percentile(raw_mae_all, 90))
                            edge_overall = (mfe50 - mae50) + 0.15 * (mfe90 - mae90)
                            mae_tail_ratio = mae90 / max(mae50, 1e-6)
                            eu_h24_diag[str(y)] = {
                                "mfe_p50": mfe50,
                                "mfe_p90": mfe90,
                                "mae_p50": mae50,
                                "mae_p90": mae90,
                                "mae_tail_ratio": mae_tail_ratio,
                                "edge_v2_overall": edge_overall,
                            }
                        else:
                            eu_h24_diag[str(y)] = {
                                "mfe_p50": 0.0,
                                "mfe_p90": 0.0,
                                "mae_p50": 0.0,
                                "mae_p90": 0.0,
                                "mae_tail_ratio": 0.0,
                                "edge_v2_overall": -1e6,
                            }

                    # Policy simulation (allow_buckets-only; suppress ignored)
                    allow_buckets_long = BUCKET_POLICY_V2["allow_buckets"].get(session, {}).get("LONG", []) if H == 24 else []
                    allow_buckets_short = BUCKET_POLICY_V2["allow_buckets"].get(session, {}).get("SHORT", []) if H == 24 else []

                    def _apply_policy(keys_arr: np.ndarray, allow_buckets: List[str]) -> Tuple[np.ndarray, np.ndarray]:
                        if keys_arr.size == 0:
                            return np.array([], dtype=bool), np.array([], dtype=bool)
                        if allow_buckets:
                            keep = np.isin(keys_arr, allow_buckets)
                        else:
                            keep = np.ones(len(keys_arr), dtype=bool)
                        return keep, ~keep

                    keep_long, suppressed_long_mask = _apply_policy(long_keys, allow_buckets_long)
                    keep_short, suppressed_short_mask = _apply_policy(short_keys, allow_buckets_short)

                    suppressed_buckets_counts_long: Dict[str, int] = {}
                    suppressed_buckets_counts_short: Dict[str, int] = {}
                    suppressed_buckets_counts: Dict[str, int] = {}
                    for k in long_keys[suppressed_long_mask]:
                        suppressed_buckets_counts_long[k] = suppressed_buckets_counts_long.get(k, 0) + 1
                        suppressed_buckets_counts[k] = suppressed_buckets_counts.get(k, 0) + 1
                    for k in short_keys[suppressed_short_mask]:
                        suppressed_buckets_counts_short[k] = suppressed_buckets_counts_short.get(k, 0) + 1
                        suppressed_buckets_counts[k] = suppressed_buckets_counts.get(k, 0) + 1

                    suppressed_count = int(suppressed_long_mask.sum() + suppressed_short_mask.sum())
                    n_signals_after_policy = int(n_sig - suppressed_count)
                    suppressed_rate = float(suppressed_count / max(n_sig, 1)) if n_sig else 0.0

                    def _top_suppressed(counts: Dict[str, int]) -> List[Dict[str, Any]]:
                        if not counts:
                            return []
                        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
                        return [{"bucket_key": k, "count": v} for k, v in items[:5]]

                    buckets_long_pol: Dict[str, List[float]] = {}
                    buckets_short_pol: Dict[str, List[float]] = {}

                    if n_sig > 0 and (lmfe.size + smfe.size) > 0:
                        if lmfe.size and keep_long.size:
                            for k, ok, mfe_v, mae_v in zip(long_keys, keep_long, lmfe, lmae):
                                if not ok:
                                    continue
                                buckets_long_pol.setdefault(k, [[], []])
                                buckets_long_pol[k][0].append(mfe_v)
                                buckets_long_pol[k][1].append(mae_v)
                        if smfe.size and keep_short.size:
                            for k, ok, mfe_v, mae_v in zip(short_keys, keep_short, smfe, smae):
                                if not ok:
                                    continue
                                buckets_short_pol.setdefault(k, [[], []])
                                buckets_short_pol[k][0].append(mfe_v)
                                buckets_short_pol[k][1].append(mae_v)

                    long_list_pol, long_other_pol = _build_bucket_list(buckets_long_pol)
                    short_list_pol, short_other_pol = _build_bucket_list(buckets_short_pol)

                    def _edge_from_arrays(mfe_arr: np.ndarray, mae_arr: np.ndarray) -> Dict[str, Any]:
                        if mfe_arr.size == 0:
                            return {
                                "mfe_p50": None,
                                "mfe_p90": None,
                                "mae_p50": None,
                                "mae_p90": None,
                                "edge_v2": None,
                            }
                        mfe50 = float(np.percentile(mfe_arr, 50))
                        mfe90 = float(np.percentile(mfe_arr, 90))
                        mae50 = float(np.percentile(mae_arr, 50))
                        mae90 = float(np.percentile(mae_arr, 90))
                        edge = (mfe50 - mae50) + 0.15 * (mfe90 - mae90)
                        return {
                            "mfe_p50": mfe50,
                            "mfe_p90": mfe90,
                            "mae_p50": mae50,
                            "mae_p90": mae90,
                            "edge_v2": edge,
                        }

                    raw_mfe = np.concatenate([lmfe, smfe]) if (lmfe.size + smfe.size) else np.array([], dtype=float)
                    raw_mae = np.concatenate([lmae, smae]) if (lmae.size + smae.size) else np.array([], dtype=float)
                    pol_mfe = np.concatenate(
                        [
                            lmfe[keep_long] if lmfe.size and keep_long.size else np.array([], dtype=float),
                            smfe[keep_short] if smfe.size and keep_short.size else np.array([], dtype=float),
                        ]
                    )
                    pol_mae = np.concatenate(
                        [
                            lmae[keep_long] if lmae.size and keep_long.size else np.array([], dtype=float),
                            smae[keep_short] if smae.size and keep_short.size else np.array([], dtype=float),
                        ]
                    )

                    overall_raw_metrics = _edge_from_arrays(raw_mfe, raw_mae)
                    overall_pol_metrics = _edge_from_arrays(pol_mfe, pol_mae)
                    delta_edge = (
                        None
                        if (overall_raw_metrics["edge_v2"] is None or overall_pol_metrics["edge_v2"] is None)
                        else overall_pol_metrics["edge_v2"] - overall_raw_metrics["edge_v2"]
                    )

                    unique_raw_keys = sorted(set(list(long_keys) + list(short_keys)))
                    unique_after_keys = sorted(set(list(long_keys[keep_long]) + list(short_keys[keep_short])))
                    allow_count = len(allow_buckets_long) + len(allow_buckets_short)
                    n_matched_allow = len(set(allow_buckets_long + allow_buckets_short) & set(unique_raw_keys))
                    example_raw = unique_raw_keys[:5]
                    example_allow = (allow_buckets_long + allow_buckets_short)[:5]
                    policy_warning = None
                    if allow_count > 0 and n_signals_after_policy == n_sig and allow_count < len(unique_raw_keys):
                        policy_warning = "NO_FILTERING_UNEXPECTED"

                    policy_effect = {
                        "overall": {
                            "n_signals_raw": n_sig,
                            "n_signals_after_policy": n_signals_after_policy,
                            "suppressed_rate": suppressed_rate,
                            "top_suppressed_buckets": _top_suppressed(suppressed_buckets_counts),
                            "allow_buckets_count": allow_count,
                            "unique_bucket_keys_raw_count": len(unique_raw_keys),
                            "unique_bucket_keys_after_policy_count": len(unique_after_keys),
                            "n_matched_allow_buckets": n_matched_allow,
                            "example_bucket_keys_raw": example_raw,
                            "example_allow_buckets": example_allow,
                            "policy_warning": policy_warning,
                            "mfe_p50_raw": overall_raw_metrics["mfe_p50"],
                            "mfe_p90_raw": overall_raw_metrics["mfe_p90"],
                            "mae_p50_raw": overall_raw_metrics["mae_p50"],
                            "mae_p90_raw": overall_raw_metrics["mae_p90"],
                            "edge_v2_overall_raw": overall_raw_metrics["edge_v2"],
                            "mfe_p50_policy": overall_pol_metrics["mfe_p50"],
                            "mfe_p90_policy": overall_pol_metrics["mfe_p90"],
                            "mae_p50_policy": overall_pol_metrics["mae_p50"],
                            "mae_p90_policy": overall_pol_metrics["mae_p90"],
                            "edge_v2_overall_policy": overall_pol_metrics["edge_v2"],
                            "delta_edge_v2": delta_edge,
                        },
                        "per_side": {
                            "LONG": {
                                "n_signals_raw": int(len(long_keys)),
                                "n_signals_after_policy": int(keep_long.sum()) if keep_long.size else 0,
                                "suppressed_rate": float(suppressed_long_mask.sum() / max(len(long_keys), 1)) if len(long_keys) else 0.0,
                                "top_suppressed_buckets": _top_suppressed(suppressed_buckets_counts_long),
                                "allow_buckets_count": len(allow_buckets_long),
                                "unique_bucket_keys_raw_count": len(set(long_keys)),
                                "unique_bucket_keys_after_policy_count": len(set(long_keys[keep_long])) if keep_long.size else 0,
                                "n_matched_allow_buckets": len(set(allow_buckets_long) & set(long_keys)),
                                "example_bucket_keys_raw": list(dict.fromkeys(long_keys))[:5],
                                "example_allow_buckets": allow_buckets_long[:5],
                            },
                            "SHORT": {
                                "n_signals_raw": int(len(short_keys)),
                                "n_signals_after_policy": int(keep_short.sum()) if keep_short.size else 0,
                                "suppressed_rate": float(suppressed_short_mask.sum() / max(len(short_keys), 1)) if len(short_keys) else 0.0,
                                "top_suppressed_buckets": _top_suppressed(suppressed_buckets_counts_short),
                                "allow_buckets_count": len(allow_buckets_short),
                                "unique_bucket_keys_raw_count": len(set(short_keys)),
                                "unique_bucket_keys_after_policy_count": len(set(short_keys[keep_short])) if keep_short.size else 0,
                                "n_matched_allow_buckets": len(set(allow_buckets_short) & set(short_keys)),
                                "example_bucket_keys_raw": list(dict.fromkeys(short_keys))[:5],
                                "example_allow_buckets": allow_buckets_short[:5],
                            },
                        },
                    }

                    if session == "EU" and H == 24:
                        flag_raw = overall_raw_metrics["edge_v2"] is not None and overall_raw_metrics["edge_v2"] > 0
                        flag_pol = overall_pol_metrics["edge_v2"] is not None and overall_pol_metrics["edge_v2"] > 0
                        eu_h24_pos_raw += int(flag_raw)
                        eu_h24_pos_policy += int(flag_pol)
                        eu_h24_diag[str(y)] = eu_h24_diag.get(str(y), {})
                        eu_h24_diag[str(y)].update(
                            {
                                "mfe_p50": overall_raw_metrics["mfe_p50"] or 0.0,
                                "mfe_p90": overall_raw_metrics["mfe_p90"] or 0.0,
                                "mae_p50": overall_raw_metrics["mae_p50"] or 0.0,
                                "mae_p90": overall_raw_metrics["mae_p90"] or 0.0,
                                "mae_tail_ratio": (
                                    (overall_raw_metrics["mae_p90"] / max(overall_raw_metrics["mae_p50"], 1e-6))
                                    if overall_raw_metrics["mae_p50"] not in (None, 0)
                                    else 0.0
                                ),
                                "edge_v2_overall": overall_raw_metrics["edge_v2"] or 0.0,
                                "year_quality_flag_raw": bool(flag_raw),
                                "year_quality_flag_policy": bool(flag_pol),
                            }
                        )

                    year_out_pol.append(
                        {
                            "threshold": float(T),
                            "horizon": int(H),
                            "summary": {
                                **summary,
                                "n_signals_after_policy": n_signals_after_policy,
                                "suppressed_count": suppressed_count,
                                "suppressed_rate": suppressed_rate,
                            },
                            "buckets": {
                                "LONG": long_list_pol,
                                "SHORT": short_list_pol,
                                "other_count_long": long_other_pol,
                                "other_count_short": short_other_pol,
                            },
                            "policy_effect_summary": policy_effect,
                        }
                    )

    meta = {
        "years": years,
        "thresholds": thresholds,
        "horizons": horizons,
        "reference_year": reference_year,
        "use_drift_normalizer": use_norm,
        "paths": {
            "bundle_dir": str(BUNDLE_DIR),
            "base28_manifest": str(BASE28_MANIFEST),
            "tape_root": str(TAPE_ROOT),
        },
        "git_hash": os.popen("cd /home/andre2/src/GX1_ENGINE && git rev-parse HEAD").read().strip(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    # Acceptance summary (EU/OVERLAP) using H=24, T=0.56 over quality window
    quality_window_years = [2023, 2024, 2025]
    def _compute_acceptance(session: str) -> Dict[str, Any]:
        edges = []
        total_sig = 0
        for y in quality_window_years:
            entries = results_policy.get(session, {}).get(str(y), [])
            found = None
            for e in entries:
                if abs(e.get("threshold", 0.0) - 0.56) < 1e-9 and e.get("horizon") == 24:
                    found = e
                    break
            if not found:
                continue
            o = found.get("policy_effect_summary", {}).get("overall", {})
            edge = o.get("edge_v2_overall_raw")
            sig = o.get("n_signals_raw")
            if edge is not None:
                edges.append(edge)
            if sig:
                total_sig += sig
        quality_edge = float(np.median(edges)) if edges else None
        status = "DEFER_TO_TRANSFORMER"
        if quality_edge is not None and quality_edge > 0 and total_sig >= 600:
            status = "ACCEPT_CONDITIONAL"
        return {
            "quality_window_years": quality_window_years,
            "quality_edge": quality_edge,
            "quality_signals": total_sig,
            "status": status,
        }

    acceptance_summary = {
        "schema_version": "acceptance_summary_v1",
        "quality_window_years": quality_window_years,
        "quality_edge_rule": "median(edge_v2_overall_raw) over years",
        "quality_signals_rule": "sum(n_signals_raw) over years",
        "accept_rule": "quality_edge > 0 AND quality_signals >= 600",
        "EU": _compute_acceptance("EU"),
        "OVERLAP": _compute_acceptance("OVERLAP"),
    }

    out_json = out_root / "DIAG_EU_OVERLAP.json"
    out_json.write_text(
        json.dumps(
            {
                "meta": meta,
                "policy_meta": BUCKET_POLICY_V2,
                "results_raw": results_raw,
                "results_policy_applied": results_policy,
                "eu_h24_year_diagnostics": eu_h24_diag,
                "eu_h24_summary": {
                    "n_years_positive_raw": eu_h24_pos_raw,
                    "n_years_positive_policy": eu_h24_pos_policy,
                },
                "acceptance_summary": acceptance_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cmd_parts = [
        "/home/andre2/venvs/gx1/bin/python",
        "gx1/scripts/sweep_xgb_multihead_v2_optuna.py",
        "--mode",
        "diag_buckets",
        "--years",
        *[str(y) for y in years],
        "--reference-year",
        str(reference_year),
        "--use-drift-normalizer",
        str(int(use_norm)),
    ]
    readme = f"""# DIAG BUCKETS (EU/OVERLAP)

Command:
  {' '.join(cmd_parts)}

Bundle: {BUNDLE_DIR}
BASE28 manifest: {BASE28_MANIFEST}
Tape root: {TAPE_ROOT}
Git commit: {meta['git_hash']}
Thresholds: {thresholds}
Horizons: {horizons}
Output: {out_json}
Policy sim: results_policy_applied (H=24 only), policy_meta: {BUCKET_POLICY_V2['note']}
"""
    (out_root / "README.md").write_text(readme, encoding="utf-8")

    return {"mode": "diag_buckets", "out_root": str(out_root), "diag_json": str(out_json)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep XGB multihead v2 with Optuna (single or separate sessions).")
    parser.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--reference-year", type=int, default=2025)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--use-drift-normalizer", type=int, default=1)

    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "separate_sessions", "final_validate", "diag_buckets"],
        help="single: one study; separate_sessions: per-session studies; final_validate: deterministic head validation; diag_buckets: deterministic bucket diagnostics.",
    )
    parser.add_argument(
        "--parallel-sessions",
        type=int,
        default=1,
        choices=[0, 1],
        help="When mode=separate_sessions: run session studies in parallel processes (1) or sequentially (0).",
    )
    parser.add_argument(
        "--jobs-per-study",
        type=int,
        default=1,
        help="When parallel-sessions=1, each session study runs with this n_jobs (default 1).",
    )
    parser.add_argument(
        "--sessions",
        type=str,
        nargs="+",
        choices=list(SESSIONS),
        default=list(SESSIONS),
        help="Sessions to run when mode=separate_sessions.",
    )
    parser.add_argument(
        "--session",
        type=str,
        choices=list(SESSIONS),
        default=None,
        help="Single mode: lock to session; if omitted, Optuna chooses.",
    )
    parser.add_argument(
        "--side-mode-fixed",
        type=str,
        choices=["both", "long_only", "short_only"],
        default=None,
        help="If set, fixes side_mode for all trials (no Optuna choice).",
    )
    parser.add_argument(
        "--fixed-config",
        type=str,
        default=None,
        help="JSON string or path with per-session fixed configs for final_validate mode.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.55, 0.56, 0.58],
        help="Thresholds for diag_buckets (ignored otherwise).",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[6, 24],
        help="Horizons for diag_buckets (ignored otherwise).",
    )

    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    if args.out_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.out_dir = WORKSPACE_ROOT / "reports" / "xgb_sweeps" / f"SWEEP_{ts}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not GX1_DATA_ROOT.exists():
        raise RuntimeError("GX1_DATA_ROOT missing")

    if args.mode == "final_validate":
        res = run_final_validate(args)
    elif args.mode == "diag_buckets":
        res = run_diag_buckets(args)
    else:
        res = run_sweep(args)

    # Pretty print summary
    mode = res.get("mode")
    print(f"Mode: {mode}")
    print(f"Output root: {res.get('out_root')}")
    if mode == "final_validate":
        print(f"Final decision: {res.get('final_json')}")
    elif mode == "diag_buckets":
        print(f"Diag JSON: {res.get('diag_json')}")
    else:
        studies = res.get("studies") or {}
        for sess_key, r in studies.items():
            print(f"[{sess_key}] Study: {r.get('study_path')}")
            print(f"[{sess_key}] Trials: {r.get('trials_csv')}")
            print(f"[{sess_key}] Best: {r.get('best')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())