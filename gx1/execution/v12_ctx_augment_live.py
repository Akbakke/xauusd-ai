#!/usr/bin/env python3
"""V12 live context augmentation for the model-native feature stack.

Background: an older stack was trained against a prebuilt parquet that contained
canonical_v3 columns + ~32 augmented ctx-cont / ctx-cat features computed
by `add_ctx_cont_columns_to_prebuilt.py`. Today the canonical_v3 prebuilt
on disk has been regenerated without those augmentations, so any live
inference must compute them from scratch.

Production context augmentation accepts only the scalar projection owned by
``gx1.features.htf_features`` through the two ``*_from_v4`` entrypoints below.
The private ``_add_htf_features`` block remains temporarily because historical
model-state/dataset consumers still call it directly; it is not an alternative
public augmentation route and must be migrated to the canonical V4 owner before
that compatibility block can be deleted.

Features added by the local context owner include:

  Spread / ATR derivations (2):
    - atr_bps                              # canonical_v2 atr / mid * 1e4
    - spread_bps                           # (ask - bid) / bid * 1e4

  Session features — see gx1.time.session_detector for the SSoT:
    - session_id                           # 0/1/2/3 = ASIA/EU/OVERLAP/US
    - minutes_since_session_open
    - minutes_to_next_session_boundary
    - session_change_flag                  # 1 if session changed vs prev bar

  Session flag (1):
    - is_ASIA

  HTF derivations:
    - D1_dist_from_ema200_atr              # (D1 mid - D1 EMA200) / D1 ATR14
    - d1_dist_change_1bar_atr_v4           # raw native-D1 first difference
    - h4_mid_ema50_dist_atr_canon_v2       # raw pre-sign H4 distance

  Microstructure on M5 close (5):
    - close_return_3_bps                   # exact 3-bar close return
    - close_return_5_bps                   # exact 5-bar close return
    - close_return_acceleration_1_bps      # change in consecutive returns
    - close_distance_below_high_range_fraction
    - close_range_observed                 # exact high>low availability mask
    - close_distance_from_ema5_bps         # classic SMA-seeded EMA5

  Quote/spread dynamics (3 — V30 package 4, 2026-08-13; abstention and
  execution-regime evidence, NOT a direction signal):
    - spread_bps_delta_1                   # 1-bar change of the ctx spread_bps
    - spread_intrabar_range_bps            # (ask_high - bid_low) / close * 1e4
    - quote_range_asymmetry_bps            # ((ask_high-ask_low) - (bid_high-bid_low)) / close * 1e4

  Swing structure (14— the 5 V1 fields + the 9 V29 event additions adopted
  into the ctx contract by V30 package 2, 2026-08-13):
    - dist_last_swing_high_atr             # (close - last pivot-high) / ATR14
    - dist_last_swing_low_atr              # (close - last pivot-low)  / ATR14
    - bars_since_swing_high
    - bars_since_swing_low
    - retracement_from_last_impulse        # 0..1 retracement
    - swing_high/low_break_event, swing_break_displacement_atr,
      bars_since_swing_high/low_break, swing_high/low_sequence_delta_atr,
      consecutive_higher_lows_count, consecutive_lower_highs_count
      # exact names/semantics owned by swing_structure_v1

ATR and spread remain continuous model evidence. No row is converted into an
operator-selected rolling regime or TRAIN-quintile bucket.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from gx1.features.htf_features import multi_tf_resample
from gx1.features.micro_structure_v1 import (
    MICRO_WARMUP_PREFIX_FIELDS_V1,
    SPREAD_DYNAMICS_WARMUP_PREFIX_FIELDS_V1,
    compute_micro_structure_features,
    compute_spread_dynamics_features,
)
from gx1.features.model_native_market_context_v1 import (
    derive_model_native_atr_spread_bps,
)
from gx1.features.swing_structure_v1 import (
    SWING_ATR_PERIOD_V1,
    SWING_LOOKBACK_V1,
    SWING_V29_ADDITION_NAMES_V1,
    compute_swing_structure_features,
)
from gx1.time.session_detector import (
    ASIA_SESSION_ID,
    get_session_id_vectorized,
    get_session_minutes_since_open_vectorized,
    get_session_minutes_to_next_boundary_vectorized,
    get_session_vectorized,
    decision_availability,
    M5_BAR_DURATION,
)

LOG = logging.getLogger("v12_ctx_augment_live")

ATR_EPS = 1e-9


# ── HTF resampling helpers ────────────────────────────────────────────────


def _resample_ohlc(df_m5: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample M5 OHLC to one declared higher timeframe (M15/H1/H4/D1).

    Input df_m5 must be DatetimeIndex'd with columns open/high/low/close.
    Returns DataFrame with same columns, indexed at the start of each HTF bar.

    V30 package 3 (2026-08-13): keyed on the declared TIMEFRAME and routed
    through ``htf_features.multi_tf_resample``, the one cadence+origin owner.
    Before this, the local literal ``"1D"`` kept its own midnight-UTC daily
    clock that could not follow the trading-day origin decision, so the live
    ``D1_dist_from_ema200_atr`` would have been
    computed on different bars than the offline surface (rule 6).
    """
    return pd.DataFrame(
        {
            "open": multi_tf_resample(df_m5["open"], timeframe).first(),
            "high": multi_tf_resample(df_m5["high"], timeframe).max(),
            "low": multi_tf_resample(df_m5["low"], timeframe).min(),
            "close": multi_tf_resample(df_m5["close"], timeframe).last(),
        }
    ).dropna()


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # A1 2026-06-04: STRICT min_periods=n (matches gx1.features.htf_features._atr).
    # The loose max(2, n//2) emitted an unconverged ATR on short serve/rescore windows;
    # used ONLY by _add_htf_features (HTF-block-local helper), so this is contained.
    return tr.rolling(window=n, min_periods=n).mean()


def _align_last_closed(
    target_idx: pd.DatetimeIndex,
    htf_series: pd.Series,
    shift: pd.Timedelta,
    *,
    base_bar_duration: pd.Timedelta = M5_BAR_DURATION,
) -> pd.Series:
    """For each M5 timestamp, return the value of the last fully-closed HTF
    bar (no lookahead). The HTF bar at time T closes at T + shift.
    """
    shifted = htf_series.copy()
    shifted.index = shifted.index + shift
    decision_idx = target_idx + base_bar_duration
    aligned = shifted.reindex(decision_idx, method="ffill")
    aligned.index = target_idx
    return aligned


# ── per-feature-group computations ────────────────────────────────────────


def _add_session_features(
    cv3: pd.DataFrame,
    *,
    base_bar_duration: pd.Timedelta = M5_BAR_DURATION,
) -> None:
    """Add the exact session fields consumed by the current context contract.

    The input is M5-cadence canonical state labelled by bar start.  Session
    evidence describes the decision instant when that row closes, ``T+5min``.
    """
    idx = decision_availability(
        cv3.index,
        bar_duration=base_bar_duration,
        context="CTX_SESSION",
    )
    session_id = get_session_id_vectorized(idx).to_numpy(dtype=np.int64)
    cv3["session_id"] = session_id
    cv3["is_ASIA"] = (
        cv3["session_id"] == ASIA_SESSION_ID
    ).astype(np.int64)
    cv3["minutes_since_session_open"] = (
        get_session_minutes_since_open_vectorized(idx)
        .to_numpy(dtype=np.float32)
    )
    cv3["minutes_to_next_session_boundary"] = (
        get_session_minutes_to_next_boundary_vectorized(idx)
        .to_numpy(dtype=np.float32)
    )
    sess_tag = get_session_vectorized(idx)
    cv3["session_change_flag"] = (
        sess_tag.ne(sess_tag.shift(1)).to_numpy(dtype=np.int64)
    )
def _add_spread_atr_bps(cv3: pd.DataFrame) -> None:
    """Mutate ``cv3`` with the shared model-native ATR/spread formula."""

    derived = derive_model_native_atr_spread_bps(cv3)
    for name in ("atr_bps", "spread_bps"):
        cv3[name] = derived[name].to_numpy(dtype=np.float64)


def _add_htf_features(
    cv3: pd.DataFrame,
    df_m5: pd.DataFrame,
    *,
    base_bar_duration: pd.Timedelta = M5_BAR_DURATION,
    context_bar_duration: pd.Timedelta = M5_BAR_DURATION,
) -> None:
    """Mutate cv3 with raw D1 and H4 continuous trend evidence.

    Existing derived columns are overwritten. Indicator convergence is exposed
    as one NaN prefix and must be trimmed by the owning common-history contract.
    """
    _htf_cols = (
        "D1_dist_from_ema200_atr",
        "d1_dist_change_1bar_atr_v4",
        "h4_mid_ema50_dist_atr_canon_v2",
    )
    if not isinstance(cv3.index, pd.DatetimeIndex) or cv3.index.tz is None:
        raise RuntimeError(
            "[LIVE_HTF_SOURCE] target requires a timezone-aware UTC DatetimeIndex"
        )
    if any(pd.Timestamp(ts).utcoffset() != pd.Timedelta(0) for ts in cv3.index[:1]):
        raise RuntimeError("[LIVE_HTF_SOURCE] target index must be UTC")
    # Non-empty means ROWS: a zero-column output container with a populated
    # index is a legitimate target (DataFrame.empty is True on any zero axis).
    if (
        len(cv3.index) == 0
        or cv3.index.hasnans
        or not cv3.index.is_unique
        or not cv3.index.is_monotonic_increasing
    ):
        raise RuntimeError(
            "[LIVE_HTF_SOURCE] target must be non-empty, unique and chronological"
        )
    m5 = df_m5.copy(deep=False)
    if "time" in m5.columns and not isinstance(m5.index, pd.DatetimeIndex):
        m5["time"] = pd.to_datetime(m5["time"], utc=True, errors="coerce")
        m5 = m5.set_index("time")
    from gx1.features.htf_features import (
        D1_EMA200_MIN_BARS,
        H4_EMA50_MIN_BARS,
        validate_causal_feature_matrix,
    )
    if not isinstance(base_bar_duration, pd.Timedelta) or base_bar_duration <= pd.Timedelta(0):
        raise RuntimeError("[LIVE_HTF_SOURCE] base bar duration must be positive")
    if (
        m5.empty
        or not isinstance(m5.index, pd.DatetimeIndex)
        or m5.index.tz is None
        or m5.index.hasnans
        or not m5.index.is_unique
        or not m5.index.is_monotonic_increasing
        or not isinstance(context_bar_duration, pd.Timedelta)
        or context_bar_duration != M5_BAR_DURATION
        or np.any(m5.index.asi8 % int(context_bar_duration.value) != 0)
    ):
        raise RuntimeError("[LIVE_HTF_SOURCE] raw source timestamp geometry invalid")
    missing_ohlc = [name for name in ("open", "high", "low", "close") if name not in m5.columns]
    if missing_ohlc:
        raise RuntimeError(f"[LIVE_HTF_SOURCE] raw source columns missing: {missing_ohlc}")
    _validate_values = m5.loc[:, ["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(_validate_values).all():
        raise RuntimeError("[LIVE_HTF_SOURCE] raw source OHLC must be finite")
    final_m5_cutoff = int(
        cv3.index[-1].value
        + base_bar_duration.value
        - context_bar_duration.value
    )
    if int(np.searchsorted(m5.index.asi8, final_m5_cutoff, side="right")) == 0:
        raise RuntimeError(
            "[LIVE_HTF_SOURCE] true M5 context does not cover the decision lane"
        )

    df_d1 = _resample_ohlc(m5, "D1")
    d1_mid = (df_d1["high"] + df_d1["low"]) * 0.5
    d1_ema200 = _ema(d1_mid, 200)
    d1_atr14 = _atr(df_d1["high"], df_d1["low"], df_d1["close"], 14)
    d1_dist = (d1_mid - d1_ema200) / np.maximum(d1_atr14, ATR_EPS)
    d1_dist.iloc[: D1_EMA200_MIN_BARS - 1] = np.nan
    cv3["D1_dist_from_ema200_atr"] = _align_last_closed(
        cv3.index,
        d1_dist,
        pd.Timedelta(days=1),
        base_bar_duration=base_bar_duration,
    ).to_numpy(dtype=float)
    cv3["d1_dist_change_1bar_atr_v4"] = _align_last_closed(
        cv3.index,
        d1_dist.diff(),
        pd.Timedelta(days=1),
        base_bar_duration=base_bar_duration,
    ).to_numpy(dtype=float)

    df_h4 = _resample_ohlc(m5, "H4")
    h4_mid = (df_h4["high"] + df_h4["low"]) * 0.5
    h4_ema50 = _ema(h4_mid, 50)
    h4_atr14 = _atr(df_h4["high"], df_h4["low"], df_h4["close"], 14)
    h4_atr_safe = np.maximum(
        h4_atr14,
        np.maximum(df_h4["close"] * 1e-4, 1e-3),
    )
    h4_distance = (h4_mid - h4_ema50) / h4_atr_safe
    h4_distance.iloc[: H4_EMA50_MIN_BARS - 1] = np.nan
    cv3["h4_mid_ema50_dist_atr_canon_v2"] = _align_last_closed(
        cv3.index,
        h4_distance,
        pd.Timedelta(hours=4),
        base_bar_duration=base_bar_duration,
    ).to_numpy(dtype=float)

    htf_values = cv3.loc[:, _htf_cols].to_numpy(dtype=np.float64)
    warmup_rows = validate_causal_feature_matrix(
        htf_values,
        expected_width=len(_htf_cols),
        context="LIVE_HTF_CAUSAL",
    )
    cv3.attrs["causal_htf_warmup_rows"] = warmup_rows


def _add_micro_features(cv3: pd.DataFrame) -> None:
    """Add the exact local-clock price primitives to ``cv3``.

    The same owner receives independently closed M5 history for Entry and M1
    history for Exit. Its honest lag/EMA NaN prefix is trimmed below.
    """
    features = compute_micro_structure_features(
        cv3["high"].to_numpy(dtype=np.float64),
        cv3["low"].to_numpy(dtype=np.float64),
        cv3["close"].to_numpy(dtype=np.float64),
    )
    for name, values in features.items():
        cv3[name] = values


def _add_spread_dynamics_features(cv3: pd.DataFrame) -> None:
    """Mutates cv3: spread_bps_delta_1, spread_intrabar_range_bps,
    quote_range_asymmetry_bps (V30 package 4, 2026-08-13).

    Delegates to the ONE owner ``micro_structure_v1``; the frame must carry the
    decision bar's own closed quotes (bid/ask close plus the four extremes,
    all members of CANONICAL_NATIVE_REQUIRED_COLUMNS). Missing columns fail
    closed there — this block has no fallback source and never substitutes the
    mid OHLC for a quote.

    ``spread_bps_delta_1`` has an honest 1-row NaN prefix which the causal
    warmup trim below removes, exactly like the swing pivot-delta prefixes."""
    for name, values in compute_spread_dynamics_features(cv3).items():
        cv3[name] = values


def _add_swing_features(cv3: pd.DataFrame) -> None:
    """Mutates cv3 with the swing-structure ctx features (dist_last_swing_high/low_atr,
    bars_since_swing_high/low, retracement_from_last_impulse + the V29/V30 event
    additions adopted by V30 package 2). Delegates to the ONE-TRUTH
    helper gx1.features.swing_structure_v1 (lookahead-safe confirmation lag) — do NOT
    re-implement the math here (2026-06-24 unification; live decision bar stays causal).

    ``include_v29_additions=True`` is the call-site contract switch: the
    complete source-owned swing tuple is emitted on every context producer at
    the same rebuild boundary (rule 6)."""
    feats = compute_swing_structure_features(
        cv3["high"].to_numpy(dtype=np.float64),
        cv3["low"].to_numpy(dtype=np.float64),
        cv3["close"].to_numpy(dtype=np.float64),
        lookback=SWING_LOOKBACK_V1,
        atr_period=SWING_ATR_PERIOD_V1,
        include_v29_additions=True,
    )
    for _name, _arr in feats.items():
        cv3[_name] = _arr


# ── public API ────────────────────────────────────────────────────────────


def _ctx_rss_mark(label: str) -> None:
    """Report resident memory inside the context completion.

    The M1 enriched stage dies between the coarse stage marks, so the peak is
    invisible without a mark per helper. Cheap, valid anywhere, no effect on
    values.
    """

    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    gib = float(line.split()[1]) / (1024.0 * 1024.0)
                    print(f"[ctx_rss] {label} rss_gib={gib:.2f}", flush=True)
                    return
    except OSError:
        return


def _finish_canonical_v3_context(
    out: pd.DataFrame,
    *,
    base_bar_duration: pd.Timedelta,
) -> pd.DataFrame:
    """Complete local context after an explicit MTF owner has attached fields."""

    _ctx_rss_mark("finish_start")
    _add_spread_atr_bps(out)
    _ctx_rss_mark("spread_atr")
    _add_session_features(out, base_bar_duration=base_bar_duration)
    _ctx_rss_mark("session")
    _add_micro_features(out)
    _ctx_rss_mark("micro")
    _add_spread_dynamics_features(out)
    _ctx_rss_mark("spread_dynamics")
    _add_swing_features(out)
    _ctx_rss_mark("swing")
    # Volume / order-flow per-bar features — SAME helper the V10 builder uses, so
    # the seq's vol_z_20/vol_ratio_5_20/vol_pct_96 are identical
    # train↔serve. Computed on the full `out` frame (full history) so trailing
    # windows match training (no window-edge skew). Fail-closed-neutral if no vol.
    from gx1.features.volume_features import add_volume_features

    add_volume_features(out)
    _ctx_rss_mark("volume")
    from gx1.scripts.augment_forward_outcome_v2 import trim_causal_context_warmup_prefix

    htf_required = [
        "D1_dist_from_ema200_atr",
        "d1_dist_change_1bar_atr_v4",
        "h4_mid_ema50_dist_atr_canon_v2",
    ]
    out = trim_causal_context_warmup_prefix(
        out,
        htf_required
        # V30 (2026-08-13): the adopted V29 swing fields carry their own honest
        # NaN warmup (no pivot-sequence delta exists until a SECOND confirmed
        # pivot per side), so they join the trim list instead of being left as
        # a non-finite ctx prefix. They are far shorter than the D1 warmup
        # above; listing the owner tuple keeps the trim correct if another
        # addition gains a prefix.
        + list(SWING_V29_ADDITION_NAMES_V1)
        # The local price owner needs five preceding rows for change-5 and a
        # classic SMA-seeded EMA5.  Missing history is NaN, never a parked
        # zero or a first-observation EMA.
        + list(MICRO_WARMUP_PREFIX_FIELDS_V1)
        # V30 package 4 (2026-08-13): spread_bps_delta_1 has no value on the
        # first row of any frame (SPREAD_DYNAMICS_CAUSAL_WARMUP_ROWS_V1 = 1).
        # Trim it by contract instead of relying on the much longer D1 warmup
        # above to cover it incidentally. The other two spread-dynamics fields
        # are defined on every row and carry no prefix.
        + list(SPREAD_DYNAMICS_WARMUP_PREFIX_FIELDS_V1),
    )
    return out


def augment_canonical_v3_model_agnostic_from_v4(
    cv3: pd.DataFrame,
    *,
    base_bar_duration: pd.Timedelta,
) -> pd.DataFrame:
    """Complete model-agnostic offline context after exact V4 projection."""

    if cv3.empty:
        raise RuntimeError("[MODEL_AGNOSTIC_CANONICAL_V4_SOURCE_EMPTY]")
    from gx1.features.htf_features import (
        require_model_native_mtf_owner_marker_v4,
    )

    require_model_native_mtf_owner_marker_v4(
        cv3,
        decision_bar_duration=base_bar_duration,
    )
    out = _finish_canonical_v3_context(
        cv3.copy(deep=False),
        base_bar_duration=base_bar_duration,
    )
    return out
