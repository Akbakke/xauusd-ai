#!/usr/bin/env python3
"""
TRUTH XGB LANE — DO NOT DEVIATE.

This script is allowed only under the canonical BASE28_CANONICAL + multihead v2 lane.
If you are touching TRIAL160 / entry_v10_ctx / FULLYEAR_2025_GATED_FUSION or any legacy prebuilt, you are in the wrong universe.

Read the runbook:
  gx1/scripts/README_TRUTH_XGB.md

Evaluate Universal Multi-head XGB v2 on multiyear data (BASE28_CANONICAL).

SSoT / TRUTH rules enforced:
- Manifest-only prebuilt resolution via BASE28_CANONICAL/CURRENT_MANIFEST.json
- No TRIAL* / per-year ctx prebuilt paths (legacy forbidden)
- Forbid columns: __index_level_0__, index_level_0, timestamp, datetime
- Time source must be UTC tz-aware (time column preferred; else tz-aware index)
- Evaluate per head (EU/OVERLAP/US) and per output (p_long/p_short/p_flat/uncertainty)
- Writes GO/NO-GO marker based on drift metrics and clip-rate constraints

Usage:
    python3 gx1/scripts/eval_xgb_multihead_v2_multiyear.py --years 2020 2021 2022 2023 2024 2025 --reference-year 2025
"""
# --- TRUTH XGB LANE GUARD (SSoT) ---------------------------------------------
TRUTH_README = "gx1/scripts/README_TRUTH_XGB.md"

def _truth_xgb_guard() -> None:
    """
    Hard-link to canonical lane docs. If you're not following it, you're in trouble.
    """
    import os
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    readme = repo_root / TRUTH_README

    gx1_data = os.environ.get("GX1_DATA_ROOT", "")
    if not gx1_data:
        raise RuntimeError(
            "TRUTH_XGB_GUARD: GX1_DATA_ROOT is not set.\n"
            f"Read: {readme}"
        )

    # Fast sanity: discourage running from wrong lane paths (typical archaeology)
    argv = " ".join(sys.argv)
    banned = ["TRIAL160", "features_v10_ctx", "FULLYEAR_2025_GATED_FUSION", "entry_v10_ctx", "v13_refined3", "PRUNE14", "PRUNE20"]
    hit = [t for t in banned if t in argv]
    if hit:
        raise RuntimeError(
            "TRUTH_XGB_GUARD: Forbidden legacy lane detected in argv: "
            + ",".join(hit)
            + "\n"
            + "This XGB lane is BASE28_CANONICAL + xgb_universal_multihead_v2__CANONICAL only.\n"
            + f"Read: {readme}"
        )

    # Optional: ensure README actually exists (prevents silent drift)
    if not readme.exists():
        raise RuntimeError(f"TRUTH_XGB_GUARD: Missing README: {readme}")

_truth_xgb_guard()
# -----------------------------------------------------------------------------

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

from gx1.scripts._truth_lane import require_truth_xgb_lane, truth_readme_path

from gx1.xgb.preprocess.xgb_input_sanitizer import XGBInputSanitizer
from gx1.xgb.multihead.xgb_multihead_model_v1 import XGBMultiheadModel
from gx1.time.session_detector import get_session_vectorized, get_session_stats

# Optional: canonical resolver (preferred if present)
try:
    from gx1.utils.canonical_prebuilt_resolver import resolve_base28_canonical_from_current_manifest  # type: ignore
except Exception:
    resolve_base28_canonical_from_current_manifest = None  # type: ignore


# GO/NO-GO thresholds
KS_THRESHOLD = 0.15
PSI_THRESHOLD = 0.10
CLIP_RATE_THRESHOLD = 6.0
MIN_CLASS_RATE = 0.02  # Heuristic: no mean prob below 2%

QUANTILE_GRID = np.linspace(0.0, 1.0, 1025)

GX1_DATA_REQUIRED = require_truth_xgb_lane(__file__)


FORBIDDEN_COLUMNS = {"__index_level_0__", "index_level_0", "timestamp", "datetime"}


def resolve_gx1_data_dir() -> Path:
    """Resolve GX1_DATA directory."""
    if "GX1_DATA_ROOT" in os.environ:
        path = Path(os.environ["GX1_DATA_ROOT"])
        if path.exists():
            return path
    # Correct default: repo root is .../src/GX1_ENGINE → GX1_DATA is one level higher (/home/andre2/GX1_DATA)
    default = WORKSPACE_ROOT.parent.parent / "GX1_DATA"
    return default


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_drift_normalizer(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"DRIFT_NORMALIZER_NOT_FOUND: {path}")
    obj = _read_json(path)
    if "quantiles" not in obj or "heads" not in obj:
        raise RuntimeError(f"DRIFT_NORMALIZER_INVALID: missing quantiles/heads in {path}")
    return obj


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _require_no_forbidden_columns(df: pd.DataFrame) -> None:
    bad = sorted(FORBIDDEN_COLUMNS.intersection(set(df.columns)))
    if bad:
        raise RuntimeError(f"FORBIDDEN_COLUMNS_PRESENT: {bad}")


def _get_time_series_utc(df: pd.DataFrame) -> pd.Series:
    """
    Preferred time column: 'time' (UTC tz-aware).
    Fallback: df.index must be tz-aware UTC DatetimeIndex.
    Hard-fail otherwise.
    """
    if "time" in df.columns:
        ts = df["time"]
        if not pd.api.types.is_datetime64tz_dtype(ts.dtype):
            raise RuntimeError("TIME_COLUMN_NOT_TZ_AWARE: expected df['time'] tz-aware UTC")
        # Normalize timezone name check
        if str(ts.dt.tz) != "UTC":
            raise RuntimeError(f"TIME_COLUMN_NOT_UTC: got tz={ts.dt.tz}")
        return ts

    # Fallback to index
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise RuntimeError("NO_TIME_COLUMN_AND_INDEX_NOT_DATETIME: expected 'time' col or DatetimeIndex")
    if idx.tz is None:
        raise RuntimeError("INDEX_NOT_TZ_AWARE: expected tz-aware UTC index")
    if str(idx.tz) != "UTC":
        raise RuntimeError(f"INDEX_NOT_UTC: got tz={idx.tz}")
    return pd.Series(idx, index=df.index, name="time")


def resolve_base28_canonical_parquet(gx1_data: Path) -> Tuple[Path, Dict[str, Any]]:
    """
    Manifest-only resolver:
      /.../BASE28_CANONICAL/CURRENT_MANIFEST.json

    Returns (parquet_path, manifest_dict).
    """
    manifest_path = (
        gx1_data
        / "data"
        / "data"
        / "prebuilt"
        / "BASE28_CANONICAL"
        / "CURRENT_MANIFEST.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"BASE28_CURRENT_MANIFEST_NOT_FOUND: {manifest_path}")

    # Prefer canonical resolver if available (it may enforce sha/rows/cols invariants)
    if resolve_base28_canonical_from_current_manifest is not None:
        # Expected to return a dict-like payload; handle common shapes.
        resolved = resolve_base28_canonical_from_current_manifest(str(manifest_path))
        if isinstance(resolved, dict):
            parquet_path = Path(resolved.get("parquet_path") or resolved.get("parquet") or "")
            if not parquet_path.is_absolute():
                parquet_path = (manifest_path.parent / parquet_path).resolve()
            return parquet_path, resolved
        # If resolver returns just the path
        if isinstance(resolved, (str, Path)):
            parquet_path = Path(resolved).resolve()
            return parquet_path, _read_json(manifest_path)

    # Fallback: read manifest directly (best-effort; still enforces sha if present)
    manifest = _read_json(manifest_path)

    # Common keys
    parquet_rel = (
        manifest.get("parquet_path")
        or manifest.get("parquet")
        or manifest.get("path")
        or manifest.get("canonical_prebuilt_parquet")
    )
    if not parquet_rel:
        raise RuntimeError(f"BASE28_MANIFEST_MISSING_PARQUET_PATH: keys={sorted(list(manifest.keys()))}")

    parquet_path = Path(parquet_rel)
    if not parquet_path.is_absolute():
        parquet_path = (manifest_path.parent / parquet_path).resolve()

    if not parquet_path.exists():
        raise FileNotFoundError(f"BASE28_CANONICAL_PARQUET_NOT_FOUND: {parquet_path}")

    # Optional sha verification if present in manifest
    want_sha = manifest.get("parquet_sha256") or manifest.get("sha256") or manifest.get("sha256_parquet")
    if want_sha:
        got_sha = compute_file_sha256(parquet_path)
        if str(want_sha).lower() != str(got_sha).lower():
            raise RuntimeError(f"BASE28_PARQUET_SHA_MISMATCH: want={want_sha} got={got_sha}")

    # TRUTH preflight sidecars required (enforce presence)
    sidecar_manifest = Path(str(parquet_path) + ".manifest.json")
    sidecar_schema = Path(str(parquet_path) + ".schema_manifest.json")
    if not sidecar_manifest.exists():
        raise FileNotFoundError(f"PREBUILT_SIDECAR_MISSING: {sidecar_manifest}")
    if not sidecar_schema.exists():
        raise FileNotFoundError(f"PREBUILT_SIDECAR_MISSING: {sidecar_schema}")

    return parquet_path, manifest


def load_sanitizer_and_features() -> Tuple[XGBInputSanitizer, List[str], str, Path, Path]:
    """
    Load sanitizer and BASE28 feature contract (strict, no fallback).
    """
    feature_contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_base28_v1.json"
    if not feature_contract_path.exists():
        raise FileNotFoundError(f"BASE28_FEATURE_CONTRACT_NOT_FOUND: {feature_contract_path}")

    with open(feature_contract_path, "r") as f:
        contract = json.load(f)
    features = contract.get("features", [])
    schema_hash = contract.get("schema_hash", "unknown")

    sanitizer_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_base28_v1.json"
    if not sanitizer_path.exists():
        raise FileNotFoundError(f"BASE28_SANITIZER_NOT_FOUND: {sanitizer_path}")
    sanitizer = XGBInputSanitizer.from_config(str(sanitizer_path))
    return sanitizer, features, schema_hash, feature_contract_path, sanitizer_path


def load_multihead_model(gx1_data: Path, bundle_dir: Optional[Path] = None) -> Tuple[XGBMultiheadModel, Path, str]:
    """
    Load canonical multihead model (SSoT).

    Default bundle dir:
      /home/andre2/GX1_DATA/models/models/xgb_universal_multihead_v2__CANONICAL
    """
    if bundle_dir is None:
        bundle_dir = gx1_data / "models" / "models" / "xgb_universal_multihead_v2__CANONICAL"

    model_path = bundle_dir / "xgb_universal_multihead_v2.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"MULTIHEAD_MODEL_NOT_FOUND: {model_path}")

    model = XGBMultiheadModel.load(str(model_path))
    model_sha = compute_file_sha256(model_path)
    return model, model_path, model_sha


def compute_ks_statistic(dist_a: np.ndarray, dist_b: np.ndarray) -> float:
    """Compute KS statistic."""
    stat, _ = scipy_stats.ks_2samp(dist_a, dist_b)
    return float(stat)


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Compute Population Stability Index."""
    min_val = min(float(np.min(expected)), float(np.min(actual)))
    max_val = max(float(np.max(expected)), float(np.max(actual)))
    if not np.isfinite(min_val) or not np.isfinite(max_val) or min_val == max_val:
        return 0.0

    bins = np.linspace(min_val, max_val, n_bins + 1)
    expected_hist, _ = np.histogram(expected, bins=bins)
    actual_hist, _ = np.histogram(actual, bins=bins)

    eps = 1e-8
    expected_pct = expected_hist / (len(expected) + eps)
    actual_pct = actual_hist / (len(actual) + eps)

    expected_pct = np.clip(expected_pct, eps, 1)
    actual_pct = np.clip(actual_pct, eps, 1)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def _downsample_df(df: pd.DataFrame, n_bars: Optional[int]) -> pd.DataFrame:
    if not n_bars or len(df) <= n_bars:
        return df
    step = max(1, len(df) // n_bars)
    return df.iloc[::step].head(n_bars)


def _filter_year(df: pd.DataFrame, ts: pd.Series, year: int) -> pd.DataFrame:
    mask = ts.dt.year == year
    return df.loc[mask]


def _compute_margin(probs: np.ndarray) -> np.ndarray:
    """Margin = top1 - top2 for 3-class probs per row."""
    # probs shape (n, 3)
    top2 = np.sort(probs, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def _margin_stats(margins: np.ndarray) -> Dict[str, float]:
    if margins.size == 0:
        return {
            "margin_mean": 0.0,
            "margin_std": 0.0,
            "margin_p10": 0.0,
            "margin_p50": 0.0,
            "margin_p90": 0.0,
            "margin_frac_lt_0p05": 0.0,
        }
    return {
        "margin_mean": float(np.mean(margins)),
        "margin_std": float(np.std(margins)),
        "margin_p10": float(np.percentile(margins, 10)),
        "margin_p50": float(np.percentile(margins, 50)),
        "margin_p90": float(np.percentile(margins, 90)),
        "margin_frac_lt_0p05": float(np.mean(margins < 0.05)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Universal Multi-head XGB v2 (BASE28_CANONICAL)")
    parser.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022, 2023, 2024, 2025], help="Years to evaluate")
    parser.add_argument("--reference-year", type=int, default=2025, help="Reference year for drift")
    parser.add_argument("--n-bars-per-year", type=int, default=None, help="Limit bars per year (downsample)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--allow-high-clip", action="store_true", help="Allow high clip rate without failing")
    parser.add_argument("--bundle-dir", type=Path, default=None, help="Override canonical XGB bundle dir (rare)")
    parser.add_argument(
        "--apply-drift-normalizer",
        action="store_true",
        help="Apply logit-ratio CDF normalizer built from reference year (per head).",
    )
    parser.add_argument(
        "--build-drift-normalizer-only",
        action="store_true",
        help="Build drift normalizer from reference year only, write to bundle dir, then exit (no GO/NO-GO).",
    )
    args = parser.parse_args()

    if args.apply_drift_normalizer and args.build_drift_normalizer_only:
        raise RuntimeError("--apply-drift-normalizer and --build-drift-normalizer-only are mutually exclusive")

    promotion_lane = "NORM" if args.apply_drift_normalizer else "RAW"

    print("=" * 60)
    print("EVAL UNIVERSAL MULTI-HEAD XGB V2 (BASE28_CANONICAL)")
    print("=" * 60)

    gx1_data = resolve_gx1_data_dir()
    print(f"GX1_DATA: {gx1_data}")

    # Resolve BASE28 canonical parquet (manifest-only)
    print("\nResolving BASE28_CANONICAL via CURRENT_MANIFEST.json...")
    parquet_path, manifest = resolve_base28_canonical_parquet(gx1_data)
    print(f"  Parquet: {parquet_path}")
    if isinstance(manifest, dict):
        m_sha = manifest.get("parquet_sha256") or manifest.get("sha256") or manifest.get("sha256_parquet")
        if m_sha:
            print(f"  Manifest sha256: {str(m_sha)[:16]}...")

    # Load model
    print("\nLoading multihead model...")
    try:
        model, model_path, model_sha = load_multihead_model(gx1_data, bundle_dir=args.bundle_dir)
        print(f"  Model: {model_path}")
        print(f"  SHA256: {model_sha[:16]}...")
        print(f"  Sessions: {model.sessions}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return 1

    normalizer_path = model_path.parent / f"XGB_MULTIHEAD_V2_DRIFT_NORMALIZER_REF{args.reference_year}.json"
    normalizer_obj: Optional[Dict[str, Any]] = None
    normalizer_sha: Optional[str] = None
    if args.apply_drift_normalizer:
        normalizer_obj = load_drift_normalizer(normalizer_path)
        normalizer_sha = compute_file_sha256(normalizer_path)

    # Enforce sessions doctrine (best-effort: fail if unexpected heads)
    expected_sessions = {"EU", "OVERLAP", "US"}
    if set(model.sessions) != expected_sessions:
        raise RuntimeError(f"SESSIONS_MISMATCH: model.sessions={model.sessions} expected={sorted(list(expected_sessions))}")

    # Load sanitizer / features
    print("\nLoading sanitizer + feature contract...")
    sanitizer, features, schema_hash, feature_contract_path, sanitizer_path = load_sanitizer_and_features()
    print(f"  Features: {len(features)}")
    print(f"  Schema hash: {schema_hash}")
    print(f"  Feature contract: {feature_contract_path}")
    print(f"  Sanitizer config: {sanitizer_path}")

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = WORKSPACE_ROOT / "reports" / "xgb_eval" / f"MULTIHEAD_V2_EVAL_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {output_dir}")

    # Load full canonical parquet once
    print("\nLoading BASE28_CANONICAL parquet (single source)...")
    df_all = pd.read_parquet(parquet_path)
    _require_no_forbidden_columns(df_all)

    ts_all = _get_time_series_utc(df_all)

    # Evaluate each year: per head/session
    print("\nEvaluating years (per head/session)...")
    year_results: Dict[int, Dict[str, Any]] = {}

    drift_normalizer: Dict[str, Dict[str, Any]] = {}
    years_to_eval = args.years
    if args.build_drift_normalizer_only:
        years_to_eval = [args.reference_year]

    for year in years_to_eval:
        df_year = _filter_year(df_all, ts_all, year)
        if df_year.empty:
            print(f"  {year}: No rows")
            continue

        # Downsample deterministically
        df_year = _downsample_df(df_year, args.n_bars_per_year)
        ts_year = _get_time_series_utc(df_year)

        print(f"\n  {year}:")
        print(f"    Rows: {len(df_year)}")

        # Session detection (use canonical time, not forbidden cols)
        df_year = df_year.copy()
        df_year["_session"] = get_session_vectorized(ts_year)
        session_stats = get_session_stats(df_year["_session"])
        print(f"    Session distribution: {session_stats.get('percentages', {})}")

        # Sanitize (clip metrics computed here)
        try:
            X, stats = sanitizer.sanitize(df_year, features, allow_nan_fill=True, nan_fill_value=0.0)
            clip_rate = float(stats.clip_rate_pct)
            print(f"    Clip rate: {clip_rate:.2f}%")
        except Exception as e:
            print(f"    ERROR: Sanitization failed: {e}")
            continue

        # Build sanitized feature frame (avoid overwriting forbidden columns)
        df_feat = pd.DataFrame(X[:, : len(features)], columns=features, index=df_year.index)
        df_feat["_session"] = df_year["_session"]

        year_results[year] = {
            "clip_rate_pct": clip_rate,
            "n_samples": int(len(df_year)),
            "session_distribution": session_stats.get("percentages", {}),
            "heads": {},
        }

        margins_all: List[float] = []

        # Evaluate each head on its session-filtered data
        for session in model.sessions:
            session_mask = df_feat["_session"] == session
            n_session_rows = int(session_mask.sum())

            if n_session_rows == 0:
                print(f"    {session}: No rows in this session")
                continue

            df_session = df_feat.loc[session_mask, features]

            try:
                outputs = model.predict_proba(df_session, session, features)

                # Compute logit ratios (strict, avoid zero)
                eps = 1e-12
                p_long = np.clip(outputs.p_long, eps, 1.0)
                p_short = np.clip(outputs.p_short, eps, 1.0)
                p_flat = np.clip(outputs.p_flat, eps, 1.0)
                a_logit = np.log(p_short / p_long)
                b_logit = np.log(p_flat / p_long)

                # Build normalizer on reference year (build-only mode)
                if args.build_drift_normalizer_only and year == args.reference_year:
                    drift_normalizer[session] = {
                        "quantiles": QUANTILE_GRID.tolist(),
                        "a_values": np.quantile(a_logit, QUANTILE_GRID).tolist(),
                        "b_values": np.quantile(b_logit, QUANTILE_GRID).tolist(),
                        "n_samples": int(len(a_logit)),
                        "reference_year": int(args.reference_year),
                    }

                # Apply normalizer (strict: must exist already)
                if args.apply_drift_normalizer:
                    if normalizer_obj is None:
                        raise RuntimeError("DRIFT_NORMALIZER_NOT_LOADED")
                    heads_norm = normalizer_obj.get("heads", {})
                    if session not in heads_norm:
                        raise RuntimeError(
                            f"DRIFT_NORMALIZER_MISSING: session={session} ref_year={args.reference_year}"
                        )
                    ref_norm = heads_norm[session]
                    ref_q = np.asarray(ref_norm.get("quantiles", QUANTILE_GRID), dtype=float)
                    ref_a_vals = np.asarray(ref_norm.get("a_values", []), dtype=float)
                    ref_b_vals = np.asarray(ref_norm.get("b_values", []), dtype=float)
                    if len(ref_a_vals) != len(ref_q) or len(ref_b_vals) != len(ref_q):
                        raise RuntimeError(
                            f"DRIFT_NORMALIZER_INVALID_SHAPE: session={session} len_q={len(ref_q)} "
                            f"len_a={len(ref_a_vals)} len_b={len(ref_b_vals)}"
                        )

                    # Percentiles of current distribution
                    sorted_a = np.sort(a_logit)
                    sorted_b = np.sort(b_logit)
                    denom_a = max(len(sorted_a) - 1, 1)
                    denom_b = max(len(sorted_b) - 1, 1)
                    pct_a = np.searchsorted(sorted_a, a_logit, side="left") / denom_a
                    pct_b = np.searchsorted(sorted_b, b_logit, side="left") / denom_b

                    a_mapped = np.interp(pct_a, ref_q, ref_a_vals)
                    b_mapped = np.interp(pct_b, ref_q, ref_b_vals)

                    exp_a = np.exp(a_mapped)
                    exp_b = np.exp(b_mapped)
                    denom = 1.0 + exp_a + exp_b
                    p_long = 1.0 / denom
                    p_short = exp_a / denom
                    p_flat = exp_b / denom

                # Basic NaN/Inf guards on (possibly normalized) probs
                for name, arr in [
                    ("p_long", p_long),
                    ("p_short", p_short),
                    ("p_flat", p_flat),
                    ("uncertainty", outputs.uncertainty),
                ]:
                    if np.isnan(arr).any():
                        raise RuntimeError(f"NAN_IN_OUTPUT: {year}/{session}/{name}")
                    if np.isinf(arr).any():
                        raise RuntimeError(f"INF_IN_OUTPUT: {year}/{session}/{name}")

                # Store results (use normalized probs if applied)
                year_results[year]["heads"][session] = {
                    "n_rows": n_session_rows,
                    "p_long_mean": float(np.mean(p_long)),
                    "p_short_mean": float(np.mean(p_short)),
                    "p_flat_mean": float(np.mean(p_flat)),
                    "uncertainty_mean": float(np.mean(outputs.uncertainty)),
                    # keep values temporarily for drift
                    "p_long_values": p_long,
                    "p_short_values": p_short,
                    "p_flat_values": p_flat,
                    "uncertainty_values": outputs.uncertainty,
                }

                probs_stack = np.stack([p_long, p_short, p_flat], axis=1)
                margins_all.extend(_compute_margin(probs_stack).tolist())

                print(
                    f"    {session} ({n_session_rows} bars): "
                    f"p_long={np.mean(p_long):.3f}, "
                    f"p_short={np.mean(p_short):.3f}, "
                    f"p_flat={np.mean(p_flat):.3f}, "
                    f"unc={np.mean(outputs.uncertainty):.3f}"
                )

            except Exception as e:
                print(f"    {session}: ERROR - {e}")

    if not year_results:
        print("\nERROR: No years evaluated successfully")
        return 1

    # Build-only path: write normalizer and exit (no drift/markers)
    if args.build_drift_normalizer_only:
        if not drift_normalizer:
            raise RuntimeError("DRIFT_NORMALIZER_EMPTY: no reference head data collected")
        normalizer_obj_out = {
            "reference_year": int(args.reference_year),
            "quantiles": QUANTILE_GRID.tolist(),
            "heads": drift_normalizer,
        }
        normalizer_path = model_path.parent / f"XGB_MULTIHEAD_V2_DRIFT_NORMALIZER_REF{args.reference_year}.json"
        normalizer_path.write_text(json.dumps(normalizer_obj_out, indent=2), encoding="utf-8")
        normalizer_sha_out = compute_file_sha256(normalizer_path)
        print(f"Wrote drift normalizer: {normalizer_path} (sha256={normalizer_sha_out})")
        # Write minimal summary
        summary = {
            "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            "reference_year": int(args.reference_year),
            "promotion_lane": "NORM",
            "apply_drift_normalizer": False,
            "drift_normalizer_path": str(normalizer_path),
            "drift_normalizer_sha256": normalizer_sha_out,
            "status": "NORMALIZER_ONLY",
        }
        summary_path = output_dir / "EVAL_SUMMARY.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote: {summary_path}")
        return 0

        # Margin stats per year (post-normalizer if applied)
        margins_np = np.asarray(margins_all, dtype=float)
        year_results[year]["margin_stats"] = _margin_stats(margins_np)
        print(
            f"    margin_mean={year_results[year]['margin_stats']['margin_mean']:.3f} "
            f"margin_p50={year_results[year]['margin_stats']['margin_p50']:.3f} "
            f"frac_lt_0.05={year_results[year]['margin_stats']['margin_frac_lt_0p05']:.3f}"
        )

    # Drift analysis
    print("\n" + "=" * 60)
    print("DRIFT ANALYSIS (per head, per output)")
    print("=" * 60)

    ref_year = args.reference_year
    if ref_year not in year_results:
        ref_year = max(year_results.keys())
        print(f"Reference year {args.reference_year} not available, using {ref_year}")

    drift_metrics: List[Dict[str, Any]] = []
    max_ks = 0.0
    max_psi = 0.0
    max_ks_by_head: Dict[str, float] = {}
    max_psi_by_head: Dict[str, float] = {}
    worst_offenders: List[Dict[str, Any]] = []

    for session in model.sessions:
        if session not in year_results[ref_year]["heads"]:
            continue

        ref_outputs = year_results[ref_year]["heads"][session]
        max_ks_by_head[session] = 0.0
        max_psi_by_head[session] = 0.0

        for year in sorted(year_results.keys()):
            if year == ref_year:
                continue
            if session not in year_results[year]["heads"]:
                continue

            year_outputs = year_results[year]["heads"][session]

            for output_name in ["p_long", "p_short", "p_flat"]:
                ref_values = np.asarray(ref_outputs[f"{output_name}_values"])
                year_values = np.asarray(year_outputs[f"{output_name}_values"])

                ks = compute_ks_statistic(ref_values, year_values)
                psi = compute_psi(ref_values, year_values)

                max_ks = max(max_ks, ks)
                max_psi = max(max_psi, psi)
                max_ks_by_head[session] = max(max_ks_by_head[session], ks)
                max_psi_by_head[session] = max(max_psi_by_head[session], psi)

                drift_metrics.append(
                    {
                        "session": session,
                        "output": output_name,
                        "year": int(year),
                        "reference_year": int(ref_year),
                        "ks_vs_ref": float(ks),
                        "psi_vs_ref": float(psi),
                    }
                )

                if ks > KS_THRESHOLD or psi > PSI_THRESHOLD:
                    worst_offenders.append(
                        {
                            "year": int(year),
                            "session": session,
                            "output": output_name,
                            "ks": float(ks),
                            "psi": float(psi),
                        }
                    )

    worst_offenders = sorted(worst_offenders, key=lambda x: (-x["ks"], -x["psi"]))[:10]

    print(f"\nMax KS overall: {max_ks:.4f} (threshold: {KS_THRESHOLD})")
    print(f"Max PSI overall: {max_psi:.4f} (threshold: {PSI_THRESHOLD})")
    print("\nMax KS by head:")
    for session, ks_val in max_ks_by_head.items():
        status = "✅" if ks_val < KS_THRESHOLD else "❌"
        print(f"  {session}: {ks_val:.4f} {status}")
    print("\nMax PSI by head:")
    for session, psi_val in max_psi_by_head.items():
        status = "✅" if psi_val < PSI_THRESHOLD else "❌"
        print(f"  {session}: {psi_val:.4f} {status}")

    if worst_offenders:
        print("\nWorst offenders:")
        for wo in worst_offenders[:5]:
            print(f"  {wo['year']}/{wo['session']}/{wo['output']}: KS={wo['ks']:.4f}, PSI={wo['psi']:.4f}")

    # Write drift metrics
    metrics_df = pd.DataFrame(drift_metrics)
    metrics_path = output_dir / "DRIFT_METRICS.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nWrote: {metrics_path}")

    # Persist drift normalizer (if built)
    if args.apply_drift_normalizer and drift_normalizer:
        normalizer_path = model_path.parent / f"XGB_MULTIHEAD_V2_DRIFT_NORMALIZER_REF{ref_year}.json"
        normalizer_obj = {
            "reference_year": int(ref_year),
            "quantiles": QUANTILE_GRID.tolist(),
            "heads": drift_normalizer,
        }
        normalizer_path.write_text(json.dumps(normalizer_obj, indent=2), encoding="utf-8")
        print(f"Wrote drift normalizer: {normalizer_path}")

    # Remove raw arrays before JSON
    for year in year_results:
        for session in list(year_results[year].get("heads", {}).keys()):
            for key in list(year_results[year]["heads"][session].keys()):
                if key.endswith("_values"):
                    del year_results[year]["heads"][session][key]

    # Class distribution heuristic (mean probs)
    class_issues: List[str] = []
    for year in year_results:
        for session in year_results[year].get("heads", {}):
            head = year_results[year]["heads"][session]
            for class_name in ["p_long", "p_short", "p_flat"]:
                rate = float(head.get(f"{class_name}_mean", 0.0))
                if rate < MIN_CLASS_RATE:
                    class_issues.append(f"{year}/{session}: {class_name}_mean = {rate:.2%} < {MIN_CLASS_RATE:.0%}")

    # GO/NO-GO analysis
    print("\n" + "=" * 60)
    print("GO/NO-GO ANALYSIS")
    print("=" * 60)

    issues: List[str] = []

    if max_ks >= KS_THRESHOLD:
        issues.append(f"KS drift too high: {max_ks:.4f} >= {KS_THRESHOLD}")
    else:
        print(f"  ✅ KS drift OK: max={max_ks:.4f} < {KS_THRESHOLD}")

    if max_psi >= PSI_THRESHOLD:
        issues.append(f"PSI drift too high: {max_psi:.4f} >= {PSI_THRESHOLD}")
    else:
        print(f"  ✅ PSI drift OK: max={max_psi:.4f} < {PSI_THRESHOLD}")

    max_clip = max(float(r["clip_rate_pct"]) for r in year_results.values())
    if max_clip > CLIP_RATE_THRESHOLD and not args.allow_high_clip:
        issues.append(f"Clip rate too high: {max_clip:.2f}% > {CLIP_RATE_THRESHOLD}%")
    else:
        print(f"  ✅ Clip rate OK: max={max_clip:.2f}%")

    if class_issues:
        issues.extend(class_issues[:3])  # limit spam
        print(f"  ⚠️  Class distribution issues: {len(class_issues)}")
    else:
        print("  ✅ Class distribution OK")

    print("\n" + "-" * 40)
    if issues:
        print("VERDICT: ❌ NO-GO")
        for issue in issues[:5]:
            print(f"  - {issue}")
    else:
        print("VERDICT: ✅ GO")
    print("-" * 40)

    # Contract SHAs
    output_contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_multihead_output_contract_v1.json"
    meta_path = model_path.parent / "xgb_universal_multihead_v2_meta.json"

    feature_contract_sha = compute_file_sha256(feature_contract_path) if feature_contract_path.exists() else None
    sanitizer_sha = compute_file_sha256(sanitizer_path) if sanitizer_path.exists() else None
    output_contract_sha = compute_file_sha256(output_contract_path) if output_contract_path.exists() else None
    meta_sha = compute_file_sha256(meta_path) if meta_path.exists() else None

    # Write marker
    bundle_dir = model_path.parent
    marker_content: Dict[str, Any] = {
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "xgb_mode": "universal_multihead_v2",
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "meta_path": str(meta_path) if meta_path.exists() else None,
        "meta_sha256": meta_sha,
        "feature_contract_path": str(feature_contract_path),
        "feature_contract_sha256": feature_contract_sha,
        "sanitizer_config_path": str(sanitizer_path),
        "sanitizer_sha256": sanitizer_sha,
        "output_contract_path": str(output_contract_path) if output_contract_path.exists() else None,
        "output_contract_sha256": output_contract_sha,
        "schema_hash": schema_hash,
        "promotion_lane": promotion_lane,
        "apply_drift_normalizer": bool(args.apply_drift_normalizer),
        "drift_normalizer_path": str(normalizer_path) if args.apply_drift_normalizer else None,
        "drift_normalizer_sha256": normalizer_sha if args.apply_drift_normalizer else None,
        "sessions": model.sessions,
        "prebuilt": {
            "current_manifest": str(
                gx1_data / "data" / "data" / "prebuilt" / "BASE28_CANONICAL" / "CURRENT_MANIFEST.json"
            ),
            "parquet_path": str(parquet_path),
            "parquet_sha256": (manifest.get("parquet_sha256") or manifest.get("sha256") or manifest.get("sha256_parquet"))
            if isinstance(manifest, dict)
            else None,
        },
        "go_criteria": {
            "max_ks": KS_THRESHOLD,
            "max_psi": PSI_THRESHOLD,
            "max_clip_rate_pct": CLIP_RATE_THRESHOLD,
            "min_class_rate_mean_prob": MIN_CLASS_RATE,
        },
        "eval_results": {
            "reference_year": int(ref_year),
            "max_ks": float(max_ks),
            "max_psi": float(max_psi),
            "max_ks_by_head": max_ks_by_head,
            "max_psi_by_head": max_psi_by_head,
            "max_clip_rate_pct": float(max_clip),
        },
        "worst_offenders": worst_offenders[:5],
        "eval_run_dir": str(output_dir),
    }

    marker_prefix = "NORM" if args.apply_drift_normalizer else "RAW"
    if not issues:
        marker_path = bundle_dir / f"XGB_MULTIHEAD_V2_{marker_prefix}_GO_MARKER.json"
        tmp_path = bundle_dir / f".XGB_MULTIHEAD_V2_{marker_prefix}_GO_MARKER.json.tmp"
        with open(tmp_path, "w") as f:
            json.dump(marker_content, f, indent=2)
        tmp_path.rename(marker_path)
        print(f"\n✅ GO MARKER written: {marker_path}")

        no_go_path = bundle_dir / f"XGB_MULTIHEAD_V2_{marker_prefix}_NO_GO_MARKER.json"
        if no_go_path.exists():
            no_go_path.unlink()
    else:
        marker_content["issues"] = issues
        marker_path = bundle_dir / f"XGB_MULTIHEAD_V2_{marker_prefix}_NO_GO_MARKER.json"
        tmp_path = bundle_dir / f".XGB_MULTIHEAD_V2_{marker_prefix}_NO_GO_MARKER.json.tmp"
        with open(tmp_path, "w") as f:
            json.dump(marker_content, f, indent=2)
        tmp_path.rename(marker_path)
        print(f"\n❌ NO-GO MARKER written: {marker_path}")

        go_path = bundle_dir / f"XGB_MULTIHEAD_V2_{marker_prefix}_GO_MARKER.json"
        if go_path.exists():
            go_path.unlink()

    # Convert numpy scalars for JSON safety
    def convert_to_serializable(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    summary = {
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "model_sha256": model_sha,
        "sessions": model.sessions,
        "reference_year": int(ref_year),
        "promotion_lane": promotion_lane,
        "apply_drift_normalizer": bool(args.apply_drift_normalizer),
        "drift_normalizer_path": str(normalizer_path) if args.apply_drift_normalizer else None,
        "drift_normalizer_sha256": normalizer_sha if args.apply_drift_normalizer else None,
        "max_ks": float(max_ks),
        "max_psi": float(max_psi),
        "max_ks_by_head": max_ks_by_head,
        "max_psi_by_head": max_psi_by_head,
        "max_clip_rate_pct": float(max_clip),
        "verdict": "GO" if not issues else "NO-GO",
        "issues": issues,
        "year_summary": year_results,
    }

    summary_path = output_dir / "EVAL_SUMMARY.json"
    with open(summary_path, "w") as f:
        json.dump(convert_to_serializable(summary), f, indent=2)
    print(f"\nWrote: {summary_path}")

    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())