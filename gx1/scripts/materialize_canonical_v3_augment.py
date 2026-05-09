#!/usr/bin/env python3
"""canonical_v3 augmentation — produces canonical_v3 from canonical_v2 by:

  1. Pruning 12 redundant features (5 exact duplicates + 7 near-duplicates @ |corr|>0.95)
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

Net feature change: 104 → 96 = -12 + 6 = (drop 12, keep+6 = 98). With cyclic (4) + smc_premium_state (1)
+ m5h1_momentum (1) = 6 added.

Notes:
  - This is NOT lookahead-unsafe — all derivations come from existing canonical_v2 features
    or from the timestamp itself.
  - The pruned features remain in canonical_v2; this script does not modify v2.
  - V10 v3 / V3 v6 contracts that target canonical_v3 must be updated separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

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
    """Add 4 cyclic time features derived from the DatetimeIndex (or 'time' column)."""
    if isinstance(df.index, pd.DatetimeIndex):
        ts = df.index
    elif "time" in df.columns:
        ts = pd.to_datetime(df["time"], utc=True)
    else:
        raise RuntimeError("[canonical_v3] no DatetimeIndex or 'time' column found")
    hour = ts.hour.to_numpy(dtype=np.float32)
    dow = ts.dayofweek.to_numpy(dtype=np.float32)
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)
    return df


def add_smc_premium_state_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """smc_premium_state = smc_premium_discount × indicator(smc_swing_state == 0)

    Interpretation: only premium pricing matters when market structure is HH+HL up.
    HH+HL up + premium near swing high = strong long bias signal.
    """
    if "smc_premium_discount" not in df.columns or "smc_swing_state" not in df.columns:
        print("[canonical_v3] WARN: smc_premium_discount or smc_swing_state missing; skipping interaction")
        return df
    df = df.copy()
    pd_score = pd.to_numeric(df["smc_premium_discount"], errors="coerce").fillna(0.5).to_numpy(np.float32)
    state = pd.to_numeric(df["smc_swing_state"], errors="coerce").fillna(0).astype(int).to_numpy()
    df["smc_premium_state"] = (pd_score * (state == 0).astype(np.float32)).astype(np.float32)
    return df


def add_cross_tf_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """m5h1_momentum = (M5_close - H1_close) / H1_atr_proxy.

    Captures intra-H1 mean-reversion potential. Uses _v1h1 features as H1 proxy.
    Source: canonical_v2 has _v1h1_atr; we synthesize H1_close via close + _v1h1_ema_diff inverse.
    Simpler proxy: use sign(_v1_close_ema_slope_3) × |_v1h1_atr| as a directional momentum scalar.

    Cleanest: compute m5_close - h1_close directly if both are present. canonical_v2 has
    `close` (M5) but not h1_close as a column. Fall back to _v1h1_ema_diff + close: not
    quite right. For now, define momentum as (close - close.shift(12)) / _v1h1_atr (12 M5 = 1h).
    """
    if "close" in df.columns and "_v1h1_atr" in df.columns:
        df = df.copy()
        close = pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64)
        h1_atr = pd.to_numeric(df["_v1h1_atr"], errors="coerce").fillna(1e-6).to_numpy(np.float64)
        delta_1h = close - np.roll(close, 12)
        delta_1h[:12] = 0.0
        m5h1_momentum = delta_1h / np.maximum(np.abs(h1_atr), 1e-6)
        df["m5h1_momentum"] = m5h1_momentum.astype(np.float32)
    elif "bid_close" in df.columns and "_v1h1_atr" in df.columns:
        df = df.copy()
        close = ((pd.to_numeric(df["bid_close"], errors="coerce") +
                  pd.to_numeric(df["ask_close"], errors="coerce")) / 2.0).to_numpy(np.float64)
        h1_atr = pd.to_numeric(df["_v1h1_atr"], errors="coerce").fillna(1e-6).to_numpy(np.float64)
        delta_1h = close - np.roll(close, 12)
        delta_1h[:12] = 0.0
        df["m5h1_momentum"] = (delta_1h / np.maximum(np.abs(h1_atr), 1e-6)).astype(np.float32)
    else:
        print("[canonical_v3] WARN: close or _v1h1_atr missing; skipping m5h1_momentum")
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
    df = pd.read_parquet(args.input)
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
    print(f"[canonical_v3] +4 cyclic time features (hour_sin/cos, dow_sin/cos)", flush=True)
    df = add_smc_premium_state_interaction(df)
    print(f"[canonical_v3] +1 smc_premium_state interaction", flush=True)
    df = add_cross_tf_momentum(df)
    print(f"[canonical_v3] +1 m5h1_momentum cross-TF feature", flush=True)

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
    import hashlib
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
        "source_v2_parquet": str(args.input),
        "diff_from_v2": {
            "dropped": to_drop,
            "added": [c for c in new_features if c in df.columns],
            "net_columns": n_out - n_in,
        },
        "note": "canonical_v3 = canonical_v2 - 12 redundant + 6 new (cyclic time + SMC interaction + cross-TF momentum)",
    }
    manifest_path = out_dir / "CURRENT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[canonical_v3] manifest → {manifest_path}", flush=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
