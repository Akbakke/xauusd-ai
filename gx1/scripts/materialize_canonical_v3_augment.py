#!/usr/bin/env python3
"""canonical_v3 augmentation — produces canonical_v3 from canonical_v2 by:

  1. Pruning 11 redundant features (5 exact duplicates + 6 near-duplicates @ |corr|>0.95)
  2. Adding 4 cyclic time features (hour_sin, hour_cos, dow_sin, dow_cos)
  3. Adding 1 SMC × swing-state interaction (smc_premium_state)
  4. Adding 1 cross-TF momentum feature (m5h1_momentum)
  5. (Future) V10 outputs as cross-bridge link — requires V10 inference pass; deferred to a
     follow-up step that joins this augmented parquet with V10 v2 inference.

Output: `canonical_v3.parquet` in same dir as canonical_v2.

Per audit findings (project_gx1_audit_findings_2026q2.md):
    - 5 exact duplicates: _v1_r5↔_v1_int_r5_atr, _v1h4_slope5↔_v1_int_slope_h4_atr,
      _v1_clv↔_v1_int_clv_atr, ret_20↔roc20, _v1_body_tr↔_v1_body_share_1
    - 7 near-duplicates (|corr|>0.95): atr↔_v1_atr14, std50↔rvol_60, etc.

Net feature change: -5 columns = 11 removed + 6 added. The additions are
cyclic time (4), smc_premium_state (1), and m5h1_momentum (1).

Notes:
  - This is NOT lookahead-unsafe — all derivations come from existing canonical_v2 features
    or from the timestamp itself.
  - The pruned features remain in canonical_v2; this script does not modify v2.
  - V10 v3 / V3 v6 contracts that target canonical_v3 must be updated separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.time.session_detector import m5_decision_availability

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Pairs to prune: keep the FIRST element, drop the SECOND. Choice is principled —
# keep the more general / canonical-naming variant, drop the alias.
PAIRS_TO_PRUNE = [
    # Exact duplicates (corr=1.000)
    ("_v1_r5", "_v1_int_r5_atr"),
    ("_v1h4_slope5", "_v1_int_slope_h4_atr"),
    ("_v1_clv", "_v1_int_clv_atr"),
    ("ret_20", "roc20"),
    ("_v1_body_share_1", "_v1_body_tr"),
    # Near-duplicates (|corr|>0.95)
    ("_v1_atr14", "atr"),                              # _v1 family wins (used in V3 io)
    ("rvol_60", "std50"),                              # rvol_60 is more interpretable
    ("_v1h1_ema_diff", "_v1h1_vwap_drift"),            # ema_diff is the "canonical" variant
    ("ema20_slope", "m15_ema_slope_5_canon_v2"),       # both useful but corr 0.962 → keep M5
    ("_v1_ema_diff", "_v1_vwap_drift48"),              # _v1_ema_diff is the canonical
    ("atr50", "m15_atr14_canon_v2"),                   # atr50 is more "current TF"
]
DROP_COLUMNS = [b for _, b in PAIRS_TO_PRUNE]


def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic features for the M5 row's observable decision time.

    Canonical rows are labelled by M5 bar start.  Their contents become known
    five minutes later, so hour/day boundaries must use ``label + 5min``.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        ts = df.index
    elif "time" in df.columns:
        # 2026-06-11 fix: pd.to_datetime(Series) returns a Series (no .hour) — wrap in
        # DatetimeIndex so the column-branch behaves like the index-branch (latent bug;
        # the daemon path always used the index branch).
        ts = pd.DatetimeIndex(pd.to_datetime(df["time"], utc=True))
    else:
        raise RuntimeError("[canonical_v3] no DatetimeIndex or 'time' column found")
    decision_ts = m5_decision_availability(ts)
    hour = decision_ts.hour.to_numpy(dtype=np.float32)
    dow = decision_ts.dayofweek.to_numpy(dtype=np.float32)
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)
    return df


def add_smc_premium_state_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """smc_premium_state = smc_premium_discount × indicator(smc_swing_state == 0)

    This is a conditional coordinate for the learned model: it exposes premium
    position while structure is clean HH+HL. It does not prescribe direction.
    """
    if "smc_premium_discount" not in df.columns or "smc_swing_state" not in df.columns:
        raise RuntimeError(
            "[canonical_v3] smc_premium_discount and smc_swing_state are required"
        )
    df = df.copy()
    pd_score = pd.to_numeric(
        df["smc_premium_discount"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    state_raw = pd.to_numeric(
        df["smc_swing_state"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if (
        not np.isfinite(pd_score).all()
        or np.any(pd_score < 0.0)
        or np.any(pd_score > 1.0)
    ):
        raise RuntimeError(
            "[canonical_v3] smc_premium_discount must be finite and within [0,1]"
        )
    if (
        not np.isfinite(state_raw).all()
        or np.any(state_raw != np.floor(state_raw))
        or np.any(state_raw < 0.0)
        or np.any(state_raw > 4.0)
    ):
        raise RuntimeError(
            "[canonical_v3] smc_swing_state must use the exact finite enum 0..4"
        )
    state = state_raw.astype(np.int8)
    df["smc_premium_state"] = (pd_score * (state == 0).astype(np.float32)).astype(np.float32)
    return df


def _atr_normalized_12bar_momentum(
    close: np.ndarray,
    h1_atr: np.ndarray,
) -> np.ndarray:
    """Return past-12-row price change scaled by aligned completed-H1 ATR.

    Causal HTF construction keeps the historical warmup prefix as NaN (it is
    not a neutral zero), so exactly one leading non-finite H1-ATR prefix is
    carried through as NaN for the downstream causal trim owner. Any
    non-finite value after the first finite observation is a data gap and
    fails closed. A finite zero H1 ATR is an availability state, not tiny
    volatility: such rows are exactly neutral rather than divided by an
    epsilon that would fabricate million-scale momentum.
    """

    if close.ndim != 1 or h1_atr.ndim != 1 or close.shape != h1_atr.shape:
        raise RuntimeError("[canonical_v3] close/H1 ATR shape mismatch")
    if not np.isfinite(close).all():
        raise RuntimeError("[canonical_v3] close must be finite")
    finite = np.isfinite(h1_atr)
    if not finite.any():
        raise RuntimeError("[canonical_v3] H1 ATR has no finite observations")
    first_finite = int(np.flatnonzero(finite)[0])
    if not finite[first_finite:].all():
        raise RuntimeError(
            "[canonical_v3] H1 ATR has non-finite values after causal warmup"
        )
    if np.any(h1_atr[first_finite:] < 0.0):
        raise RuntimeError("[canonical_v3] H1 ATR must be non-negative")
    delta_12 = close - np.roll(close, 12)
    delta_12[:12] = 0.0
    result = np.zeros_like(delta_12, dtype=np.float64)
    np.divide(delta_12, h1_atr, out=result, where=finite & (h1_atr > 1e-6))
    result[:first_finite] = np.nan
    return result


def add_cross_tf_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Add 12 completed-M5-bar momentum normalized by completed-H1 ATR."""
    if "close" in df.columns and "_v1h1_atr" in df.columns:
        df = df.copy()
        close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64)
        h1_atr = pd.to_numeric(
            df["_v1h1_atr"], errors="coerce"
        ).to_numpy(np.float64)
        m5h1_momentum = _atr_normalized_12bar_momentum(close, h1_atr)
        df["m5h1_momentum"] = m5h1_momentum.astype(np.float32)
    elif {"bid_close", "ask_close", "_v1h1_atr"}.issubset(df.columns):
        df = df.copy()
        close = ((pd.to_numeric(df["bid_close"], errors="coerce") +
                  pd.to_numeric(df["ask_close"], errors="coerce")) / 2.0).to_numpy(np.float64)
        h1_atr = pd.to_numeric(
            df["_v1h1_atr"], errors="coerce"
        ).to_numpy(np.float64)
        df["m5h1_momentum"] = _atr_normalized_12bar_momentum(
            close, h1_atr
        ).astype(np.float32)
    else:
        raise RuntimeError(
            "[canonical_v3] close (or bid/ask close) and _v1h1_atr are required"
        )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="canonical_v2 → canonical_v3 augmentation")
    parser.add_argument("--input", type=str,
                        default="/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V2_PREBUILT/xauusd_m5_CANONICAL_V2_2020_2026.parquet")
    parser.add_argument("--output-dir", type=str,
                        default="/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary; don't write output")
    args = parser.parse_args()

    print(f"[canonical_v3] loading: {args.input}", flush=True)
    input_path = Path(args.input).expanduser().resolve()
    df = pd.read_parquet(input_path)
    if "time" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
    df = df.sort_index()
    n_in = len(df.columns)
    print(f"[canonical_v3] input: {df.shape[0]:,} rows × {n_in} columns", flush=True)

    # Drop redundant columns
    to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    skipped = [c for c in DROP_COLUMNS if c not in df.columns]
    print(f"[canonical_v3] dropping {len(to_drop)} redundant features:", flush=True)
    for col in to_drop:
        keep = next((a for a, b in PAIRS_TO_PRUNE if b == col), "?")
        print(f"   - drop {col}  (kept {keep})", flush=True)
    if skipped:
        print(f"[canonical_v3] skipped (not present): {skipped}", flush=True)
    df = df.drop(columns=to_drop)

    # Add new features
    df = add_cyclic_time_features(df)
    print("[canonical_v3] +4 cyclic time features (hour_sin/cos, dow_sin/cos)", flush=True)
    df = add_smc_premium_state_interaction(df)
    print("[canonical_v3] +1 smc_premium_state interaction", flush=True)
    df = add_cross_tf_momentum(df)
    print("[canonical_v3] +1 m5h1_momentum cross-TF feature", flush=True)

    n_out = len(df.columns)
    print(f"[canonical_v3] output: {df.shape[0]:,} rows × {n_out} columns "
          f"(net change: {n_out - n_in:+d})", flush=True)

    # Sanity checks on new features
    new_features = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "smc_premium_state", "m5h1_momentum"]
    for f in new_features:
        if f in df.columns:
            v = df[f]
            print(f"   {f}: mean={v.mean():.4f} std={v.std():.4f} min={v.min():.4f} max={v.max():.4f} "
                  f"n_nan={int(v.isna().sum())}", flush=True)

    if args.dry_run:
        print("[canonical_v3] DRY RUN — not writing output", flush=True)
        return

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "xauusd_m5_CANONICAL_V3_2020_2026.parquet"
    print(f"[canonical_v3] writing → {out_parquet}", flush=True)
    # Reset DatetimeIndex to a `time` column so downstream readers can join on it
    # without losing the timestamp (parquet index round-trip is lossy via pd.read_parquet).
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if df.columns[0] != "time":
            df = df.rename(columns={df.columns[0]: "time"})
    df.to_parquet(out_parquet, index=False)

    # Manifest
    input_h = hashlib.sha256()
    with open(input_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            input_h.update(chunk)
    input_sha = input_h.hexdigest()
    source_summary_path = input_path.parent / "canonical_features_v2_summary.json"
    source_summary: dict[str, object] = {}
    if source_summary_path.exists():
        source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    h = hashlib.sha256()
    with open(out_parquet, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    sha = h.hexdigest()
    manifest = {
        "kind": "BASE28_CANONICAL_MANIFEST",  # compat with existing resolver
        "kind_actual_v3": "CANONICAL_V3_PREBUILT_AUGMENTED_MANIFEST",
        "parquet_path": str(out_parquet),
        "parquet_sha256": sha,
        "rows": len(df),
        "cols_total": n_out,
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "source_v2_parquet": str(input_path),
        "source_v2_parquet_sha256": input_sha,
        "source_v2_summary_path": str(source_summary_path),
        "source_v2_no_lookahead": bool(
            ((source_summary.get("htf_alignment_contract_v1") or {}) if isinstance(source_summary, dict) else {}).get(
                "no_lookahead"
            )
        ),
        "source_v2_builder_version": (
            source_summary.get("canonical_v2_builder_version") if isinstance(source_summary, dict) else None
        ),
        "source_v2_htf_alignment_contract": (
            source_summary.get("htf_alignment_contract_v1") if isinstance(source_summary, dict) else None
        ),
        "diff_from_v2": {
            "dropped": to_drop,
            "added": [c for c in new_features if c in df.columns],
            "net_columns": n_out - n_in,
        },
        "note": "canonical_v3 = canonical_v2 - 11 redundant + 6 new (cyclic time + SMC interaction + cross-TF momentum)",
    }
    manifest_path = out_dir / "CURRENT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[canonical_v3] manifest → {manifest_path}", flush=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
