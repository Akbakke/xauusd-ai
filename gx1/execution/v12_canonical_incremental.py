#!/usr/bin/env python3
"""V12 incremental canonical/BASE34 updater — keeps prebuilts within ~5 min of real-time.

Replaces the 25-min full-rebuild cycle with a 5-15 sec incremental
update. Cutoff drift drops from ~50 min to ~5 min (one M5 bar — must
wait for the bar to close before features can be finalized).

Strategy
--------
For each new M5 bar that has closed since the last update:
  1. Slice canonical M5 tape to a 30-day warmup window ending at the
     new bar's close.
  2. Run build_canonical_v2 on the slice — produces features for all
     warmup bars, but we only KEEP the new ones.
  3. Apply canonical_v3 augment to the new rows (drops 12 + adds 6).
  4. Apply add_ctx_cont logic to compute the 32 BASE34-style features
     for the new rows, using the existing BASE34 prebuilt's distribution
     for percentile-based features (vol_regime_id, atr_bucket, etc.).
  5. Atomic append:
        canonical_v3 prebuilt: read full + concat new rows + write atomic
        BASE34 prebuilt: same (M1 cadence — expand each M5 bar to 5 M1 rows)
  6. Update CURRENT_MANIFEST.json to reflect new cutoff.

Per-cycle cost
--------------
~3-10 sec on a 30-day warmup slice (~8000 M5 bars). Per-bar incremental
cost ~1ms. Atomic disk write is the dominant time (~1-2 sec for 200 MB).

To run continuously:
    nohup python3 -u gx1/execution/v12_canonical_incremental.py --loop > log 2>&1 &
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.execution.v12_m1_to_m5_downsample import m1_to_m5  # noqa: E402
from gx1.features.htf_features import (  # noqa: E402
    REGIME_V4_V2_MTF_PER_TF,
    REGIME_V4_V2_MTF_SKIP,
    REGIME_V4_V2_MTF_TFS,
)
from gx1.features.basic_v1 import (  # noqa: E402
    PLUS5_FEATURES,
    compute_plus5_features,
)
from gx1.features.micro_structure_v1 import MICRO_FEATURE_NAMES_V1  # noqa: E402
from gx1.features.regime_v4_features import REGIME_V4_DERIVED_COLS  # noqa: E402
from gx1.features.swing_structure_v1 import SWING_FEATURE_NAMES_V1  # noqa: E402
from gx1.features.volume_features import VOLUME_FEATURE_NAMES  # noqa: E402
from gx1.scripts.materialize_build_canonical_features_v2 import (  # noqa: E402
    build_canonical_v2,
)
from gx1.scripts.materialize_canonical_v3_augment import (  # noqa: E402
    DROP_COLUMNS,
    add_cyclic_time_features,
    add_smc_premium_state_interaction,
    add_cross_tf_momentum,
)

LOG = logging.getLogger("v12_incr")

CANONICAL_M1_DIR = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL")
CANONICAL_M5_DIR = Path("/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL")
COLLECTOR_DIR = Path("/home/andre2/GX1_DATA/reports/v12_live_data")
CANONICAL_V3_PREBUILT = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/"
    "xauusd_m5_CANONICAL_V3_2020_2026.parquet"
)
BASE34_MANIFEST = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/BASE28_CANONICAL/CURRENT_MANIFEST.json"
)
WARMUP_DAYS = 30   # enough for ATR14, EMA200, RSI, etc. to stabilize

# PLUS5: 5 features re-added on 2026-05-21 because the PLUS5 Entry-IQL ensemble
# was trained on real values.  This function is the retained computation source.
BASE34_RAW_M1_OWNED_COLUMNS = ("open", "high", "low", "close", "volume")
_BASE34_V2_MTF_OWNED_COLUMNS = tuple(
    f"{timeframe}_{live_fragment}_v2"
    for timeframe in REGIME_V4_V2_MTF_TFS
    for live_fragment, _source_column in REGIME_V4_V2_MTF_PER_TF
    if (timeframe, live_fragment) not in REGIME_V4_V2_MTF_SKIP
)
BASE34_AUGMENT_OWNED_COLUMNS = frozenset(
    (
        "atr_bps",
        "spread_bps",
        "session_id",
        "is_ASIA",
        "_v1_is_EU",
        "_v1_is_US",
        "minutes_since_session_open",
        "minutes_to_next_session_boundary",
        "session_change_flag",
        "session_tradable",
        "_v1_int_ema_us",
        "_v1_int_range_us",
        "_v1_int_slope_h1_us",
        "trend_regime_id",
        "vol_regime_id",
        "atr_bucket",
        "spread_bucket",
        "D1_dist_from_ema200_atr",
        "D1_atr_percentile_252",
        "H1_range_compression_ratio",
        "M15_range_compression_ratio",
        "H4_trend_sign_cat",
        *MICRO_FEATURE_NAMES_V1,
        *SWING_FEATURE_NAMES_V1,
        *VOLUME_FEATURE_NAMES,
        *REGIME_V4_DERIVED_COLS,
        *_BASE34_V2_MTF_OWNED_COLUMNS,
    )
)
M1_MARKET_IDENTITY_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
)


def _exact_finite_number(value, *, context: str) -> float:
    """Return one exact numeric feature value or fail the append cycle."""

    if isinstance(value, (bool, np.bool_)):
        raise RuntimeError(f"{context}: boolean is not numeric feature evidence")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context}: feature value is not numeric") from exc
    if not np.isfinite(parsed):
        raise RuntimeError(f"{context}: feature value is non-finite")
    return parsed


def _align_exact_canonical_schema(
    existing: pd.DataFrame,
    incremental: pd.DataFrame,
) -> pd.DataFrame:
    """Return incremental columns in canonical order or reject any schema drift."""

    existing_columns = list(existing.columns)
    incremental_columns = list(incremental.columns)
    if len(existing_columns) != len(set(existing_columns)):
        raise RuntimeError("canonical_v3 existing schema contains duplicate columns")
    if len(incremental_columns) != len(set(incremental_columns)):
        raise RuntimeError("canonical_v3 incremental schema contains duplicate columns")
    existing_set = set(existing_columns)
    incremental_set = set(incremental_columns)
    missing_columns = sorted(existing_set - incremental_set)
    extra_columns = sorted(incremental_set - existing_set)
    if missing_columns or extra_columns:
        raise RuntimeError(
            "canonical_v3 incremental schema mismatch: "
            f"missing={missing_columns} extra={extra_columns}"
        )
    return incremental.loc[:, existing_columns]


def _build_base34_owned_row(
    *,
    timestamp: pd.Timestamp,
    output_columns: list[str],
    cv3_row: pd.Series,
    cv3_aug: pd.DataFrame,
    m1_row: pd.Series,
    cv3_index: pd.DatetimeIndex,
) -> dict[str, float]:
    """Build one BASE34 row from one unambiguous producer per column."""

    if len(output_columns) != len(set(output_columns)):
        raise RuntimeError("BASE34 output schema contains duplicate columns")
    augmented_timestamp = cv3_row.name
    row_data: dict[str, float] = {}
    for column in output_columns:
        if column in BASE34_RAW_M1_OWNED_COLUMNS:
            if column not in m1_row.index:
                raise RuntimeError(f"BASE34 exact M1 source lacks {column}")
            value = m1_row[column]
            context = f"BASE34 {timestamp} M1.{column}"
        elif column == "is_model_bar":
            row_data[column] = float(timestamp in cv3_index)
            continue
        elif column in BASE34_AUGMENT_OWNED_COLUMNS:
            if column not in cv3_aug.columns:
                raise RuntimeError(
                    f"BASE34 augment-owned column {column!r} lacks its exact producer"
                )
            if augmented_timestamp not in cv3_aug.index:
                raise RuntimeError(
                    "BASE34 augmented source lacks exact closed M5 state at "
                    f"{augmented_timestamp}"
                )
            value = cv3_aug.at[augmented_timestamp, column]
            context = f"BASE34 {timestamp} augmented.{column}"
        elif column in cv3_row.index:
            value = cv3_row[column]
            context = f"BASE34 {timestamp} canonical_v3.{column}"
        else:
            raise RuntimeError(
                f"BASE34 column {column!r} has no exact current-bar producer"
            )
        row_data[column] = _exact_finite_number(value, context=context)
    return row_data


def _compute_plus5_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility call-site delegating to the basic_v1 PLUS5 owner."""
    return compute_plus5_features(df)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write parquet atomically via .tmp + os.replace (no torn writes)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=("time" not in df.columns))
    os.replace(tmp, path)


def _coerce_time_col(df: pd.DataFrame) -> pd.DataFrame | None:
    """Return df with a usable 'time' COLUMN, or None for a torn/malformed read.
    Guards the 2026-06-17 KeyError:'time' race: a collector/canonical M1 parquet read
    mid-write (or with time stored as the index) lacks a 'time' column → returning None
    makes the caller skip that file THIS cycle; the next 15s cycle re-reads the completed
    file. Non-blocking, fail-safe (never fabricates data)."""
    if "time" in df.columns:
        return df
    if df.index.name == "time" or isinstance(df.index, pd.DatetimeIndex):
        return df.reset_index().rename(columns={df.index.name or "index": "time"})
    return None


def _load_m1_collector_for_window(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """Union of canonical M1 tape + live collector parquets covering [start, end]."""
    parts: list[pd.DataFrame] = []
    for fp in sorted(COLLECTOR_DIR.glob("xauusd_m1_*.parquet")):
        try:
            df = pd.read_parquet(fp)
        except Exception as exc:
            raise RuntimeError(f"live M1 source is unreadable: {fp}") from exc
        df = _coerce_time_col(df)
        if df is None:
            raise RuntimeError(f"live M1 source lacks exact time: {fp}")
        df["time"] = pd.to_datetime(df["time"], utc=True)
        sub = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)]
        if len(sub) > 0:
            parts.append(sub)
    for yr in range(start_ts.year, end_ts.year + 1):
        fp = CANONICAL_M1_DIR / f"year={yr}" / "part-000.parquet"
        if not fp.exists():
            continue
        try:
            df = pd.read_parquet(fp)
        except Exception as exc:
            raise RuntimeError(f"canonical M1 source is unreadable: {fp}") from exc
        df = _coerce_time_col(df)
        if df is None:
            raise RuntimeError(f"canonical M1 source lacks exact time: {fp}")
        df["time"] = pd.to_datetime(df["time"], utc=True)
        sub = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)]
        if len(sub) > 0:
            parts.append(sub)
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    duplicate_rows = combined[combined.duplicated(subset=["time"], keep=False)]
    identity_columns = [
        column
        for column in M1_MARKET_IDENTITY_COLUMNS
        if column in combined.columns
    ]
    if len(duplicate_rows) and not identity_columns:
        raise RuntimeError("overlapping M1 sources lack market identity columns")
    if len(duplicate_rows):
        numeric_identity = duplicate_rows[identity_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )
        if not np.isfinite(
            numeric_identity.to_numpy(dtype=np.float64)
        ).all():
            raise RuntimeError("overlapping M1 source has non-finite market values")
        distinct = (
            pd.concat(
                [duplicate_rows[["time"]].reset_index(drop=True), numeric_identity.reset_index(drop=True)],
                axis=1,
            )
            .groupby("time", sort=False)[identity_columns]
            .nunique(dropna=False)
        )
        conflicts = distinct.columns[(distinct > 1).any(axis=0)].tolist()
        if conflicts:
            first_conflict = distinct.index[(distinct[conflicts] > 1).any(axis=1)][0]
            raise RuntimeError(
                "canonical/live M1 source conflict at "
                f"{first_conflict}: columns={conflicts}"
            )
    return (
        combined.drop_duplicates(subset=["time"], keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )


def _apply_canonical_v3_augment(v2: pd.DataFrame) -> pd.DataFrame:
    v3 = v2.copy()
    if "time" in v3.columns and not isinstance(v3.index, pd.DatetimeIndex):
        v3["time"] = pd.to_datetime(v3["time"], utc=True)
        v3 = v3.set_index("time")
    to_drop = [c for c in DROP_COLUMNS if c in v3.columns]
    v3 = v3.drop(columns=to_drop)
    v3 = add_cyclic_time_features(v3)
    v3 = add_smc_premium_state_interaction(v3)
    v3 = add_cross_tf_momentum(v3)
    return v3


def update_canonical_v3_incremental() -> tuple[int, pd.Timestamp | None]:
    """Extend canonical_v3 prebuilt with any new M5 bars that have closed.

    Returns (n_appended, new_cutoff_ts). n_appended=0 means nothing new.
    """
    # Load existing prebuilt (full file — ~200 MB, ~1 sec)
    if not CANONICAL_V3_PREBUILT.exists():
        raise RuntimeError(
            f"canonical_v3 prebuilt missing: {CANONICAL_V3_PREBUILT}"
        )
    t0 = _time.perf_counter()
    cv3 = pd.read_parquet(CANONICAL_V3_PREBUILT)
    if "time" in cv3.columns:
        cv3["time"] = pd.to_datetime(cv3["time"], utc=True)
        cv3 = cv3.set_index("time")
    cv3 = cv3.sort_index()
    last_in_prebuilt = cv3.index[-1]

    # Load M1 data covering [last_in_prebuilt - WARMUP_DAYS, now]
    now_ts = pd.Timestamp.now(tz="UTC").floor("min")
    warmup_start = last_in_prebuilt - pd.Timedelta(days=WARMUP_DAYS)
    m1 = _load_m1_collector_for_window(warmup_start, now_ts)
    if m1.empty:
        raise RuntimeError("canonical/live M1 union is empty for the append window")

    # Aggregate to M5
    m5 = m1_to_m5(m1)
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    m5 = m5.set_index("time").sort_index()

    # Identify NEW M5 bars (post-prebuilt-cutoff)
    new_m5 = m5[m5.index > last_in_prebuilt]
    if new_m5.empty:
        return 0, last_in_prebuilt

    LOG.info(f"new M5 bars to append: {len(new_m5)}  (range {new_m5.index[0]} → {new_m5.index[-1]})")

    # Run canonical_v2 on warmup + new bars together
    # We need OHLC + 'time' column for build_canonical_v2
    warmup_m5 = m5[(m5.index <= last_in_prebuilt) & (m5.index >= warmup_start)]
    full_slice = pd.concat([warmup_m5, new_m5]).reset_index()
    if "time" not in full_slice.columns:
        full_slice = full_slice.rename_axis("time").reset_index()

    v2 = build_canonical_v2(full_slice)
    # Apply v3 augment
    v3_new = _apply_canonical_v3_augment(v2)
    # PLUS5: compute the 5 features on the full warmup slice (needs OHLCV history)
    # and merge into v3_new by index. Uses m5 which has OHLCV pre-augment.
    plus5_df = _compute_plus5_features(m5[["open", "high", "low", "close", "volume"]])
    for c in PLUS5_FEATURES:
        v3_new[c] = plus5_df[c].reindex(v3_new.index).astype(np.float32)
    # Phase 0a/C3 (2026-06-04): recompute the 5 HTF cols FRESH via the ONE-TRUTH
    # build_htf_tape (same math as the offline ctx builder) instead of letting the
    # BASE34 append forward-fill a FROZEN value (the H4-sign freeze: stale '2 bull' vs
    # true '0 bear', stuck since the last full build). Compute over the FULL tape
    # (existing cv3 OHLC + new bars) so the D1 270-bar percentile warmup is satisfied,
    # then assign to v3_new by index. These 5 cols PERSIST only once cv3's schema carries
    # them (one-shot backfill); pre-backfill the column-alignment below silently drops
    # them (safe no-op). A compute hiccup must NOT break the live append (that would
    # stale the whole pipeline) is forbidden: stale HTF state must stop the
    # append so runtime freshness fails closed.
    try:
        from gx1.features.htf_features import build_htf_tape, HTF_TAPE_COLUMNS
        _ohlc = ["open", "high", "low", "close"]
        if all(c in cv3.columns for c in _ohlc):
            _full_ohlc = pd.concat([cv3[_ohlc], new_m5[_ohlc]]).sort_index()
            _full_ohlc = _full_ohlc[~_full_ohlc.index.duplicated(keep="last")]
            _htf = build_htf_tape(_full_ohlc)
            for c in HTF_TAPE_COLUMNS:
                v3_new[c] = _htf[c].reindex(v3_new.index)
        else:
            raise RuntimeError("[C3_HTF] cv3 lacks exact OHLC")
    except Exception as _htf_err:
        raise RuntimeError("[C3_HTF] exact HTF recompute failed") from _htf_err
    # D1-EWM CONVERGENCE (2026-06-13, LANE B): build_canonical_v2 above ran on a WARMUP_DAYS-day
    # slice, so its D1 EWM features (d1_rsi14/d1_ema_slope_20 — both in the live XGB v3 contract,
    # consumed every M5 bar) seed UN-converged (up to ~15 RSI pts off at the tail; train uses
    # full-history). Recompute the D1 features over the FULL cv3 OHLC + new bars (cheap — ~1500 D1
    # bars) via the ONE-TRUTH compute_d1_features + merge_asof_features (the SAME functions
    # build_canonical_v2 uses) and OVERWRITE, so the live cv3 tail == training. Mirrors the HTF
    # full-tape recompute above. Fail-loud-but-non-fatal (a hiccup must never stale the live append).
    try:
        from gx1.scripts.materialize_build_canonical_features_v2 import (
            compute_d1_features, merge_asof_features)
        _ohlc = ["open", "high", "low", "close"]
        if all(c in cv3.columns for c in _ohlc):
            _full = pd.concat([cv3[_ohlc], new_m5[_ohlc]]).sort_index()
            _full = _full[~_full.index.duplicated(keep="last")].rename_axis("time").reset_index()
            _d1_full = compute_d1_features(_full)
            _d1_cols = [c for c in _d1_full.columns if c not in ("time", "_time_ns")]
            _base = pd.DataFrame({"time": v3_new.index})
            _merged = merge_asof_features(_base, _d1_full, base_time_col="time")
            for _c in _d1_cols:
                if _c in v3_new.columns:
                    v3_new[_c] = _merged[_c].to_numpy()
        else:
            raise RuntimeError("[D1_CONVERGE] cv3 lacks exact OHLC")
    except Exception as _d1err:
        raise RuntimeError(
            "[D1_CONVERGE] full-history D1 recompute failed"
        ) from _d1err
    # Take only the new bars
    v3_new = v3_new[v3_new.index > last_in_prebuilt]
    if v3_new.empty:
        LOG.warning("v3 augment produced no new rows")
        return 0, last_in_prebuilt

    # Align columns with the existing immutable schema. Missing values cannot
    # be manufactured as zeros because every field can influence the models.
    # PLUS5 cols are added to v3_new above; they will survive the alignment if
    # cv3 has them (after one-shot backfill).
    v3_new = _align_exact_canonical_schema(cv3, v3_new)
    numeric = v3_new.apply(pd.to_numeric, errors="coerce")
    invalid_columns = [
        column
        for column in numeric.columns
        if not np.isfinite(numeric[column].to_numpy(dtype=np.float64)).all()
    ]
    if invalid_columns:
        raise RuntimeError(
            "canonical_v3 incremental output contains non-finite features: "
            f"{invalid_columns}"
        )

    # Concat + atomic write
    cv3_extended = pd.concat([cv3, v3_new])
    cv3_extended.reset_index().to_parquet(
        CANONICAL_V3_PREBUILT.with_suffix(".parquet.tmp"),
        index=False,
    )
    os.replace(CANONICAL_V3_PREBUILT.with_suffix(".parquet.tmp"), CANONICAL_V3_PREBUILT)

    new_cutoff = v3_new.index[-1]
    elapsed = _time.perf_counter() - t0
    LOG.info(f"canonical_v3 extended +{len(v3_new)} bars in {elapsed*1000:.0f} ms  "
              f"new cutoff: {new_cutoff}")
    return len(v3_new), new_cutoff


def update_base34_incremental(new_cutoff: pd.Timestamp) -> int:
    """Extend BASE34 prebuilt (M1 cadence) with new bars up to new_cutoff."""
    if not BASE34_MANIFEST.exists():
        raise RuntimeError(f"BASE34 manifest missing: {BASE34_MANIFEST}")
    manifest = json.loads(BASE34_MANIFEST.read_text())
    base34_path = Path(manifest["parquet_path"])
    if not base34_path.exists():
        raise RuntimeError(f"BASE34 file missing: {base34_path}")

    t0 = _time.perf_counter()
    base34 = pd.read_parquet(base34_path)
    if not isinstance(base34.index, pd.DatetimeIndex):
        if "time" in base34.columns:
            base34["time"] = pd.to_datetime(base34["time"], utc=True)
            base34 = base34.set_index("time")
    base34 = base34.sort_index()
    last_in_base34 = base34.index[-1]

    if new_cutoff <= last_in_base34:
        return 0

    # Load M1 bars from [last_in_base34 + 1min, new_cutoff + 5min]
    # The +5min on new_cutoff is to include the M1 bars within the closing M5 bucket
    start_ts = last_in_base34 + pd.Timedelta(minutes=1)
    end_ts = new_cutoff + pd.Timedelta(minutes=5)
    m1 = _load_m1_collector_for_window(start_ts, end_ts)
    if m1.empty:
        raise RuntimeError("BASE34 append lacks exact M1 source rows")
    m1["time"] = pd.to_datetime(m1["time"], utc=True)
    m1 = m1.set_index("time").sort_index()
    new_m1 = m1[(m1.index > last_in_base34) & (m1.index <= end_ts)]
    if new_m1.empty:
        raise RuntimeError("BASE34 append has no new M1 rows for advanced cv3 cutoff")

    # Use the LATEST M5-aligned feature values from canonical_v3 as the
    # ffill-source for each new M1 bar (no lookahead — uses just-closed
    # M5 bar's features for the next 5 M1 bars).
    cv3 = pd.read_parquet(CANONICAL_V3_PREBUILT)
    if "time" in cv3.columns:
        cv3["time"] = pd.to_datetime(cv3["time"], utc=True)
        cv3 = cv3.set_index("time")
    cv3 = cv3.sort_index()

    # 2026-06-11 FREEZE FIX: the 37 BASE34-only columns (32 ctx-augment features + session/regime/
    # swing/micro flags + is_model_bar) used to be COPY-FORWARDED from the last base34 row on every
    # append → ALL of them froze from 2026-05-25 18:25 (session pinned US, atr_bps 5.348, vol MEDIUM,
    # trend NEUTRAL — journal-confirmed all the way into the live entry/exit state vectors). They are
    # now RECOMPUTED per cycle via the ONE-TRUTH live augmenter
    # (v12_ctx_augment_live.augment_canonical_v3). Full canonical history is
    # mandatory: bounded windows reset EWM, swing/BOS/CHOCH, regime-age and D1
    # trend-age state and are not state-equivalent to training.
    from gx1.execution.v12_ctx_augment_live import augment_canonical_v3
    cv3_win = cv3.copy()
    # REGIME_V4 (2026-06-13 cutover): attach the per-TF V2 multi-TF scalars REGIME_V4 needs as
    # inputs (R1/R2/R3) BEFORE augment, via the ONE-TRUTH helper shared with serve + build. Without
    # this, augment_canonical_v3's REGIME_V4 block (GX1_REGIME_V4=1) is fail-closed-missing and the
    # 52 regime cols carry-forward FROZEN on append (the 2026-05-25 freeze class). This recompute is
    # the per-append cost driver (~17s @ 420d) but runs only when a new M5 closes (run_one_cycle gates
    # update_base34_incremental on n_cv3>0 — every 5 min), and the write is atomic.
    from gx1.features.htf_features import attach_default_regime_v4_v2_scalars
    attach_default_regime_v4_v2_scalars(cv3_win)
    _m5_cols = [c for c in ("open", "high", "low", "close", "volume") if c in cv3_win.columns]
    cv3_aug = augment_canonical_v3(cv3_win, cv3_win[_m5_cols].copy())
    # For each new M1 bar, find the most-recent CLOSED M5 bucket
    new_m1_rows = []
    base34_cols = list(base34.columns)
    output_columns = list(dict.fromkeys([*base34_cols, *BASE34_RAW_M1_OWNED_COLUMNS]))
    for ts in new_m1.index:
        m5_floor = ts.floor("5min")
        # The "last closed M5 bar" = the one strictly BEFORE this M1's bucket
        # (because the bucket containing this M1 hasn't closed until the 5min mark)
        closed_m5 = m5_floor - pd.Timedelta(minutes=5)
        if closed_m5 in cv3.index:
            cv3_row = cv3.loc[closed_m5]
        else:
            raise RuntimeError(
                f"BASE34 append lacks exact closed M5 state at {closed_m5}"
            )
        # Every field has exactly one owner. Raw OHLCV comes from M1, recomputed
        # context/HTF/regime fields come from the augmenter even when a stale
        # column with the same name exists in cv3, and all remaining fields come
        # from canonical_v3.
        row_data = _build_base34_owned_row(
            timestamp=ts,
            output_columns=output_columns,
            cv3_row=cv3_row,
            cv3_aug=cv3_aug,
            m1_row=new_m1.loc[ts],
            cv3_index=cv3.index,
        )
        new_m1_rows.append((ts, row_data))

    if not new_m1_rows:
        return 0

    new_df = pd.DataFrame([d for _, d in new_m1_rows],
                            index=pd.DatetimeIndex([t for t, _ in new_m1_rows], name="time"))
    extended = pd.concat([base34, new_df])

    # Atomic write
    tmp = base34_path.with_suffix(".parquet.tmp")
    extended.to_parquet(tmp, index=True)
    os.replace(tmp, base34_path)

    # Update manifest with new SHA
    sha = hashlib.sha256(base34_path.read_bytes()).hexdigest()
    manifest["parquet_sha256"] = sha
    manifest["rows"] = len(extended)
    manifest["created_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest["note"] = f"incremental update: +{len(new_df)} M1 bars"
    manifest_tmp = BASE34_MANIFEST.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(manifest_tmp, BASE34_MANIFEST)

    elapsed = _time.perf_counter() - t0
    LOG.info(f"BASE34 extended +{len(new_df)} M1 rows in {elapsed*1000:.0f} ms  "
              f"new cutoff: {extended.index[-1]}")
    return len(new_df)


def run_one_cycle() -> dict:
    """Run one incremental update cycle. Returns stats dict."""
    t0 = _time.perf_counter()
    n_cv3, new_cv3_cutoff = update_canonical_v3_incremental()
    n_base34 = 0
    if new_cv3_cutoff is not None:
        n_base34 = update_base34_incremental(new_cv3_cutoff)
    elapsed = _time.perf_counter() - t0
    return {
        "cv3_appended": n_cv3,
        "base34_appended": n_base34,
        "new_cutoff": str(new_cv3_cutoff) if new_cv3_cutoff is not None else None,
        "elapsed_sec": round(elapsed, 2),
    }


def backfill_base34_ctx(since_ts: pd.Timestamp) -> dict:
    """One-shot repair of the 2026-05-25 FREEZE: recompute the 37 BASE34-only ctx columns for all
    rows after `since_ts` via the same ONE-TRUTH augmenter the (fixed) incremental path uses.
    Run with the gx1-canonical-incremental daemon STOPPED (this rewrites the same parquet)."""
    from gx1.execution.v12_ctx_augment_live import augment_canonical_v3
    manifest = json.loads(BASE34_MANIFEST.read_text())
    base34_path = Path(manifest["parquet_path"])
    base34 = pd.read_parquet(base34_path)
    base34.index = pd.to_datetime(base34.index, utc=True)
    cv3 = pd.read_parquet(CANONICAL_V3_PREBUILT)
    if "time" in cv3.columns:
        cv3["time"] = pd.to_datetime(cv3["time"], utc=True)
        cv3 = cv3.set_index("time")
    cv3 = cv3.sort_index()
    # Backfill must use the same full-history state as normal append.
    win = cv3.copy()
    # REGIME_V4: same ONE-TRUTH per-TF V2 scalar attach as the live append path, so the backfill
    # recomputes the 52 regime cols (not just the legacy ctx) when GX1_REGIME_V4=1.
    from gx1.features.htf_features import attach_default_regime_v4_v2_scalars
    attach_default_regime_v4_v2_scalars(win)
    _m5_cols = [c for c in ("open", "high", "low", "close", "volume") if c in win.columns]
    cv3_aug = augment_canonical_v3(win, win[_m5_cols].copy())
    target = [c for c in base34.columns if c not in cv3.columns and c != "is_model_bar"
              and c in cv3_aug.columns]
    mask = base34.index > since_ts
    n_rows = int(mask.sum())
    # map each M1 row to its last CLOSED M5 bar (same semantics as the append path);
    # int64-ns on both sides (tz-aware vs naive .values would raise in searchsorted)
    closed_ns = (base34.index[mask].floor("5min") - pd.Timedelta(minutes=5)).asi8
    aug_idx = np.searchsorted(cv3_aug.index.asi8, closed_ns, side="right") - 1
    valid = aug_idx >= 0
    if not bool(valid.all()):
        raise RuntimeError(
            "BASE34 backfill lacks exact prior augmented M5 state for target rows"
        )
    before = {c: int(base34.loc[mask, c].nunique()) for c in target[:6]}
    for c in target:
        vals = cv3_aug[c].to_numpy()[aug_idx]
        try:
            numeric = np.asarray(vals, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"BASE34 backfill column {c!r} is not numeric"
            ) from exc
        if not np.isfinite(numeric).all():
            raise RuntimeError(
                f"BASE34 backfill column {c!r} contains non-finite evidence"
            )
        base34.loc[mask, c] = pd.Series(
            numeric,
            index=base34.index[mask],
        ).astype(np.float32)
    if "is_model_bar" in base34.columns:
        base34.loc[mask, "is_model_bar"] = base34.index[mask].isin(cv3.index)
    after = {c: int(base34.loc[mask, c].nunique()) for c in target[:6]}
    tmp = base34_path.with_suffix(".parquet.tmp")
    base34.to_parquet(tmp, index=True)
    os.replace(tmp, base34_path)
    manifest["parquet_sha256"] = hashlib.sha256(base34_path.read_bytes()).hexdigest()
    manifest["rows"] = len(base34)
    manifest["created_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest["note"] = (manifest.get("note", "") +
                        f" | BACKFILL {datetime.now(timezone.utc):%Y-%m-%dT%H:%MZ}: recomputed "
                        f"{len(target)} frozen ctx cols for {n_rows} rows since {since_ts} (freeze fix).")
    manifest_tmp = BASE34_MANIFEST.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(manifest_tmp, BASE34_MANIFEST)
    return {"rows_backfilled": n_rows, "cols": len(target),
            "nunique_before_sample": before, "nunique_after_sample": after}


def main() -> int:
    p = argparse.ArgumentParser(description="V12 incremental canonical/BASE34 updater")
    p.add_argument("--loop", action="store_true", help="Loop continuously (default: one-shot)")
    p.add_argument("--interval", type=int, default=60, help="Loop interval in seconds (default 60)")
    p.add_argument("--backfill-base34-since", type=str, default=None,
                   help="One-shot: recompute the frozen BASE34 ctx cols for rows after this UTC ts "
                        "(freeze fix repair). Stop the daemon first.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ")

    if args.backfill_base34_since:
        stats = backfill_base34_ctx(pd.Timestamp(args.backfill_base34_since, tz="UTC"))
        print(json.dumps(stats, indent=2))
        return 0

    if not args.loop:
        stats = run_one_cycle()
        print(json.dumps(stats, indent=2))
        return 0

    # Hang forensics (2026-06-12, hang #3 — NOT the self-check; 60 threads in
    # futex_wait + main in do_select): SIGUSR1 dumps ALL thread stacks to stderr
    # (lands in the daemon log). The watchdog sends it before restarting, so the
    # next hang self-documents its root cause. py-spy needs ptrace/sudo — this doesn't.
    import faulthandler
    import signal as _signal
    faulthandler.register(_signal.SIGUSR1, all_threads=True)
    LOG.info(f"starting incremental updater loop (interval={args.interval}s)")
    # Rule-9 self-check MOVED OUT of this loop (2026-06-12, standing decision after
    # hang #2): the hourly in-process check (reading BOTH prebuilts inside the
    # appender) was the last sign of life before BOTH daemon hangs at the 21-22Z
    # pause boundary (2026-06-12 00:05 and 22:00). It now runs as its own systemd
    # --user timer (gx1-rule9-selfcheck.timer → feature_liveness --live-tail) so a
    # stuck check can never stall data collection. The launch preflight remains
    # the hard gate; this daemon only appends.
    while True:
        try:
            stats = run_one_cycle()
            if stats["cv3_appended"] > 0:
                LOG.info(f"cycle stats: {stats}")
        except Exception as exc:
            LOG.exception(f"cycle failed: {exc}")
        _time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
