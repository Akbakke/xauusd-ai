#!/usr/bin/env python3
"""V12 live context augmentation for the model-native feature stack.

Background: an older stack was trained against a prebuilt parquet that contained
canonical_v3 columns + ~32 augmented ctx-cont / ctx-cat features computed
by `add_ctx_cont_columns_to_prebuilt.py`. Today the canonical_v3 prebuilt
on disk has been regenerated without those augmentations, so any live
inference must compute them from scratch.

Features added (32 total):

  Spread / ATR derivations (2):
    - atr_bps                              # canonical_v2 atr / mid * 1e4
    - spread_bps                           # (ask - bid) / bid * 1e4

  Session features (5) — see gx1.time.session_detector for the SSoT:
    - session_id                           # 0/1/2/3 = ASIA/EU/OVERLAP/US
    - minutes_since_session_open
    - minutes_to_next_session_boundary
    - session_change_flag                  # 1 if session changed vs prev bar
    - session_tradable                     # 1 if session_id != 0 (not ASIA)

  Session flags (3) — for backward-compat with _v1_is_* shifted columns
  that basic_v1.py only computes when a 'ts' column is present:
    - is_ASIA
    - _v1_is_EU                            # is_EU shifted by 1 bar
    - _v1_is_US                            # is_US shifted by 1 bar

  Session-interaction (3) — products with _v1_is_US:
    - _v1_int_ema_us                       # _v1_ema_diff * _v1_is_US
    - _v1_int_range_us                     # _v1_range_z * _v1_is_US
    - _v1_int_slope_h1_us                  # _v1h1_slope3 * _v1_is_US

  Regime / bucket categoricals (4):
    - trend_regime_id                      # 0/1/2 from exact D1 distance (legacy cat)
    - vol_regime_id                        # 0..4 causal trailing-288-bar ATR rank
    - atr_bucket                           # 0..4 frozen TRAIN-distribution ATR rank
    - spread_bucket                        # 0..4 percentile rank of spread_bps

  H4 trend sign (1):
    - H4_trend_sign_cat                    # 0/1/2 from sign(H4 mid - H4 EMA50)

  HTF derivations (4):
    - D1_dist_from_ema200_atr              # (D1 mid - D1 EMA200) / D1 ATR14
    - H1_range_compression_ratio           # H1 ATR14 / H1 ATR100
    - D1_atr_percentile_252                # rolling 252-day percentile of D1 ATR14
    - M15_range_compression_ratio          # M15 ATR14 / M15 ATR100

  Microstructure on M5 close (5):
    - micro_momentum_3                     # close - close.shift(3)
    - micro_momentum_5                     # close - close.shift(5)
    - micro_acceleration                   # diff of diff
    - wick_ratio                           # (high - close) / range
    - distance_ema_fast                    # close - EMA5

  Swing structure (5):
    - dist_last_swing_high_atr             # (close - last pivot-high) / ATR14
    - dist_last_swing_low_atr              # (close - last pivot-low)  / ATR14
    - bars_since_swing_high
    - bars_since_swing_low
    - retracement_from_last_impulse        # 0..1 retracement

The local volatility regime is causal over a fixed 288-bar window. Absolute
ATR and spread buckets require an explicit immutable TRAIN-only rank reference;
there is no global edge file or frame-relative fallback.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_state_v2 import (
    TrainRankReferenceV2,
    bucket_against_train_reference,
    causal_vol_regime_bucket,
    compute_causal_market_rank_inputs,
)
from gx1.features.micro_structure_v1 import compute_micro_structure_features
from gx1.features.model_native_market_context_v1 import (
    derive_model_native_trend_regime_id,
)
from gx1.features.swing_structure_v1 import (
    SWING_ATR_PERIOD_V1,
    SWING_LOOKBACK_V1,
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


def _resample_ohlc(df_m5: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample M5 OHLC to a higher timeframe (1H/4H/15min/1D).

    Input df_m5 must be DatetimeIndex'd with columns open/high/low/close.
    Returns DataFrame with same columns, indexed at the start of each HTF bar.
    """
    out = pd.DataFrame(
        {
            "open": df_m5["open"].resample(rule).first(),
            "high": df_m5["high"].resample(rule).max(),
            "low": df_m5["low"].resample(rule).min(),
            "close": df_m5["close"].resample(rule).last(),
        }
    ).dropna()
    return out


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
    """Mutates cv3: adds session_id, is_ASIA, _v1_is_EU, _v1_is_US,
    minutes_since/to, session_change_flag, session_tradable.

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
    cv3["session_tradable"] = (
        cv3["session_id"] != ASIA_SESSION_ID
    ).astype(np.int64)
    # Session membership is calendar evidence known at the decision instant;
    # shifting it would make these two fields disagree with session_id for one
    # complete M5 bar at every boundary.
    is_eu = (sess_tag == "EU").astype(np.float64)
    is_us = (sess_tag == "US").astype(np.float64)
    cv3["_v1_is_EU"] = is_eu.to_numpy(dtype=np.float64)
    cv3["_v1_is_US"] = is_us.to_numpy(dtype=np.float64)


def _add_session_interactions(cv3: pd.DataFrame) -> None:
    """Mutates cv3: _v1_int_ema_us, _v1_int_range_us, _v1_int_slope_h1_us.

    Requires _v1_is_US already set, plus _v1_ema_diff / _v1_range_z /
    _v1h1_slope3 from canonical_v2.
    """
    is_us = cv3["_v1_is_US"].to_numpy(dtype=np.float64)
    if "_v1_ema_diff" in cv3.columns:
        cv3["_v1_int_ema_us"] = cv3["_v1_ema_diff"].to_numpy(dtype=np.float64) * is_us
    if "_v1_range_z" in cv3.columns:
        cv3["_v1_int_range_us"] = cv3["_v1_range_z"].to_numpy(dtype=np.float64) * is_us
    if "_v1h1_slope3" in cv3.columns:
        cv3["_v1_int_slope_h1_us"] = (
            cv3["_v1h1_slope3"].to_numpy(dtype=np.float64) * is_us
        )
def _add_spread_atr_bps(cv3: pd.DataFrame) -> None:
    """Mutate ``cv3`` with the shared model-native ATR/spread formula."""

    derived = compute_causal_market_rank_inputs(cv3)
    # ``atr`` is an existing canonical Wilder-ATR feature.  The rank contract
    # uses a distinct simple TR14 formula; only its normalized rank input is
    # shared here, otherwise unrelated model evidence silently changes meaning.
    for name in ("atr_bps", "spread_bps"):
        cv3[name] = derived[name].to_numpy(dtype=np.float64)


def _add_htf_features(
    cv3: pd.DataFrame,
    df_m5: pd.DataFrame,
    *,
    base_bar_duration: pd.Timedelta = M5_BAR_DURATION,
    context_bar_duration: pd.Timedelta = M5_BAR_DURATION,
) -> None:
    """Mutates cv3: D1_dist_from_ema200_atr, H1_range_compression_ratio,
    D1_atr_percentile_252, M15_range_compression_ratio, H4_trend_sign_cat.

    Existing derived columns are overwritten. Indicator convergence is exposed
    as one NaN prefix and must be trimmed by the owning common-history contract.
    """
    _htf_cols = (
        "D1_dist_from_ema200_atr",
        "D1_atr_percentile_252",
        "H1_range_compression_ratio",
        "M15_range_compression_ratio",
        "H4_trend_sign_cat",
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
        D1_PCTL252_MIN_BARS,
        H1_ATR100_MIN_BARS,
        H4_EMA50_MIN_BARS,
        M15_ATR100_MIN_BARS,
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

    df_d1 = _resample_ohlc(m5, "1D")
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

    def _pctl_last(arr):
        values = np.asarray(arr, dtype=float)
        if not np.isfinite(values).all():
            return float("nan")
        return float((values <= values[-1]).mean())

    atr_pctl = d1_atr14.rolling(252, min_periods=252).apply(_pctl_last, raw=True)
    atr_pctl.iloc[: D1_PCTL252_MIN_BARS - 1] = np.nan
    cv3["D1_atr_percentile_252"] = _align_last_closed(
        cv3.index,
        atr_pctl,
        pd.Timedelta(days=1),
        base_bar_duration=base_bar_duration,
    ).to_numpy(dtype=float)

    # H1 features (H1_ATR100_MIN_BARS=120 — ATR100 converged)
    df_h1 = _resample_ohlc(m5, "1h")
    h1_atr14 = _atr(df_h1["high"], df_h1["low"], df_h1["close"], 14)
    h1_atr100 = _atr(df_h1["high"], df_h1["low"], df_h1["close"], 100)
    h1_comp = h1_atr14 / np.maximum(h1_atr100, ATR_EPS)
    h1_comp.iloc[: H1_ATR100_MIN_BARS - 1] = np.nan
    cv3["H1_range_compression_ratio"] = _align_last_closed(
        cv3.index,
        h1_comp,
        pd.Timedelta(hours=1),
        base_bar_duration=base_bar_duration,
    ).to_numpy(dtype=float)

    # M15 features (M15_ATR100_MIN_BARS=200)
    df_m15 = _resample_ohlc(m5, "15min")
    m15_atr14 = _atr(df_m15["high"], df_m15["low"], df_m15["close"], 14)
    m15_atr100 = _atr(df_m15["high"], df_m15["low"], df_m15["close"], 100)
    m15_comp = m15_atr14 / np.maximum(m15_atr100, ATR_EPS)
    m15_comp.iloc[: M15_ATR100_MIN_BARS - 1] = np.nan
    cv3["M15_range_compression_ratio"] = _align_last_closed(
        cv3.index,
        m15_comp,
        pd.Timedelta(minutes=15),
        base_bar_duration=base_bar_duration,
    ).to_numpy(dtype=float)

    # H4 trend sign categorical (H4_EMA50_MIN_BARS=80 — EMA50 converged)
    df_h4 = _resample_ohlc(m5, "4h")
    h4_mid = (df_h4["high"] + df_h4["low"]) * 0.5
    h4_ema50 = _ema(h4_mid, 50)
    diff = h4_mid - h4_ema50
    sign_series = (np.sign(diff) + 1.0).astype(np.float64)
    sign_series.iloc[: H4_EMA50_MIN_BARS - 1] = np.nan
    cv3["H4_trend_sign_cat"] = _align_last_closed(
        cv3.index,
        sign_series,
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
    """Mutates cv3: micro_momentum_3/5, micro_acceleration, wick_ratio,
    distance_ema_fast. Computed on M5 close/high/low (already in cv3)."""
    features = compute_micro_structure_features(
        cv3["high"].to_numpy(dtype=np.float64),
        cv3["low"].to_numpy(dtype=np.float64),
        cv3["close"].to_numpy(dtype=np.float64),
    )
    for name, values in features.items():
        cv3[name] = values


def _add_swing_features(cv3: pd.DataFrame) -> None:
    """Mutates cv3 with the 5 swing-structure ctx features (dist_last_swing_high/low_atr,
    bars_since_swing_high/low, retracement_from_last_impulse). Delegates to the ONE-TRUTH
    helper gx1.features.swing_structure_v1 (lookahead-safe confirmation lag) — do NOT
    re-implement the math here (2026-06-24 unification; live decision bar stays causal)."""
    feats = compute_swing_structure_features(
        cv3["high"].to_numpy(dtype=np.float64),
        cv3["low"].to_numpy(dtype=np.float64),
        cv3["close"].to_numpy(dtype=np.float64),
        lookback=SWING_LOOKBACK_V1,
        atr_period=SWING_ATR_PERIOD_V1,
    )
    for _name, _arr in feats.items():
        cv3[_name] = _arr


def _add_regime_categoricals(
    cv3: pd.DataFrame,
    *,
    rank_reference: TrainRankReferenceV2 | None = None,
) -> None:
    """Mutates cv3: trend_regime_id, vol_regime_id, atr_bucket, spread_bucket."""
    # The legacy categorical has one immutable source: exact D1 distance. The
    # price-vs-EMA signal remains separate continuous model evidence.
    if "D1_dist_from_ema200_atr" not in cv3.columns:
        raise RuntimeError("[REGIME] exact D1_dist_from_ema200_atr source missing")
    from gx1.features.htf_features import validate_causal_feature_matrix

    d = pd.to_numeric(cv3["D1_dist_from_ema200_atr"], errors="coerce").to_numpy(
        np.float64
    )
    d_warmup = validate_causal_feature_matrix(
        d[:, None],
        expected_width=1,
        context="TREND_REGIME_D1_SOURCE",
    )
    trend = np.full(len(d), np.nan, dtype=np.float64)
    trend[d_warmup:] = derive_model_native_trend_regime_id(d[d_warmup:])
    _ctx_rss_mark("  rc.trend_computed")
    cv3["trend_regime_id"] = trend
    _ctx_rss_mark("  rc.trend_assigned")
    # vol_regime_id is a distinct causal local-volatility coordinate. Absolute
    # ATR/spread buckets belong only to an explicit immutable TRAIN reference.
    # A model-agnostic canonical build intentionally omits those two fields.
    if "atr_bps" not in cv3.columns or "spread_bps" not in cv3.columns:
        raise RuntimeError("[REGIME] exact atr_bps/spread_bps sources missing")
    _av = cv3["atr_bps"].to_numpy(dtype=float)
    _sv = cv3["spread_bps"].to_numpy(dtype=float)
    _ctx_rss_mark("  rc.av_sv_extracted")
    vol = causal_vol_regime_bucket(_av)
    _ctx_rss_mark("  rc.vol_computed")
    cv3["vol_regime_id"] = vol.astype(np.int64)
    _ctx_rss_mark("  rc.vol_assigned")
    cv3.drop(columns=["atr_bucket", "spread_bucket"], errors="ignore", inplace=True)
    _ctx_rss_mark("  rc.dropped")
    if rank_reference is None:
        return
    cv3["atr_bucket"] = bucket_against_train_reference(
        _av,
        rank_reference.atr_bps_sorted,
    )
    cv3["spread_bucket"] = bucket_against_train_reference(
        _sv,
        rank_reference.spread_bps_sorted,
    )
    _ctx_rss_mark("  rc.buckets_assigned")


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
    rank_reference: TrainRankReferenceV2 | None,
    base_bar_duration: pd.Timedelta,
) -> pd.DataFrame:
    """Complete local context after an explicit MTF owner has attached fields."""

    _ctx_rss_mark("finish_start")
    _add_spread_atr_bps(out)
    _ctx_rss_mark("spread_atr")
    _add_session_features(out, base_bar_duration=base_bar_duration)
    _ctx_rss_mark("session")
    _add_session_interactions(out)
    _ctx_rss_mark("session_interactions")
    _add_micro_features(out)
    _ctx_rss_mark("micro")
    _add_swing_features(out)
    _ctx_rss_mark("swing")
    _add_regime_categoricals(out, rank_reference=rank_reference)
    _ctx_rss_mark("regime_categoricals")
    # Volume / order-flow per-bar features — SAME helper the V10 builder uses, so
    # the seq's vol_z_20/vol_ratio_5_20/vol_pct_96/signed_vol_z_20 are identical
    # train↔serve. Computed on the full `out` frame (full history) so trailing
    # windows match training (no window-edge skew). Fail-closed-neutral if no vol.
    from gx1.features.volume_features import add_volume_features

    add_volume_features(out)
    _ctx_rss_mark("volume")
    # REGIME_V4 (2026-06-03): multi-TF regime CONDITIONING + 'regime is shifting' CHANGE-
    # DETECTION features. ONE-TRUTH: identical gx1.features.regime_v4_features helper as the
    # build-side (add_ctx_cont_columns_to_prebuilt.py) — cannot drift. Sources are already on `out`: per-TF
    # regime/trend-age/ema-stack from the shared V4 MTF cache, D1_dist from
    # _add_htf_features above. `out` is full-history time-ordered (same as the volume helper).
    from gx1.features.regime_v4_features import (
        REGIME_V4_DERIVED_COLS,
        REGIME_V4_SOURCE_COLS,
        add_regime_v4_features,
    )

    _ctx_rss_mark("before_regime_v4")
    add_regime_v4_features(
        out,
        base_bar_duration=base_bar_duration,
    )
    from gx1.scripts.augment_forward_outcome_v2 import trim_causal_context_warmup_prefix

    htf_required = [
        "D1_dist_from_ema200_atr",
        "D1_atr_percentile_252",
        "H1_range_compression_ratio",
        "M15_range_compression_ratio",
        "H4_trend_sign_cat",
        "trend_regime_id",
    ]
    out = trim_causal_context_warmup_prefix(
        out,
        htf_required + list(REGIME_V4_SOURCE_COLS) + list(REGIME_V4_DERIVED_COLS),
    )
    out["trend_regime_id"] = out["trend_regime_id"].astype(np.int64)
    out["H4_trend_sign_cat"] = out["H4_trend_sign_cat"].astype(np.int64)
    return out


def _augment_canonical_v3(
    cv3: pd.DataFrame,
    df_m5: pd.DataFrame,
    *,
    rank_reference: TrainRankReferenceV2 | None,
    base_bar_duration: pd.Timedelta = M5_BAR_DURATION,
    context_bar_duration: pd.Timedelta = M5_BAR_DURATION,
) -> pd.DataFrame:
    """Historical live-only augmentation; offline producers never call it."""

    if cv3.empty:
        return cv3
    out = cv3.copy(deep=False)
    _add_htf_features(
        out,
        df_m5,
        base_bar_duration=base_bar_duration,
        context_bar_duration=context_bar_duration,
    )
    return _finish_canonical_v3_context(
        out,
        rank_reference=rank_reference,
        base_bar_duration=base_bar_duration,
    )


def augment_canonical_v3_from_v4(
    cv3: pd.DataFrame,
    *,
    rank_reference: TrainRankReferenceV2,
    base_bar_duration: pd.Timedelta,
) -> pd.DataFrame:
    """Complete offline context only after the exact V4 MTF projection."""

    if not isinstance(rank_reference, TrainRankReferenceV2):
        raise RuntimeError(
            "[CANONICAL_CTX_TRAIN_RANK_REQUIRED] explicit immutable reference required"
        )
    if cv3.empty:
        raise RuntimeError("[CANONICAL_CTX_V4_SOURCE_EMPTY]")
    from gx1.features.htf_features import (
        require_model_native_mtf_owner_marker_v4,
    )

    require_model_native_mtf_owner_marker_v4(
        cv3,
        decision_bar_duration=base_bar_duration,
    )
    out = _finish_canonical_v3_context(
        cv3.copy(deep=False),
        rank_reference=rank_reference,
        base_bar_duration=base_bar_duration,
    )
    missing = sorted({"atr_bucket", "spread_bucket"} - set(out.columns))
    if missing:
        raise RuntimeError(f"[CANONICAL_CTX_TRAIN_RANK_OUTPUT_MISSING] {missing}")
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
        rank_reference=None,
        base_bar_duration=base_bar_duration,
    )
    forbidden = sorted({"atr_bucket", "spread_bucket"} & set(out.columns))
    if forbidden:
        raise RuntimeError(
            f"[MODEL_AGNOSTIC_CANONICAL] TRAIN-fit fields leaked into raw state: {forbidden}"
        )
    return out


def augment_canonical_v3_model_agnostic(
    cv3: pd.DataFrame,
    df_m5: pd.DataFrame,
    *,
    base_bar_duration: pd.Timedelta = M5_BAR_DURATION,
    context_bar_duration: pd.Timedelta = M5_BAR_DURATION,
) -> pd.DataFrame:
    """Build causal shared context without dataset-fitted bucket categories."""

    out = _augment_canonical_v3(
        cv3,
        df_m5,
        rank_reference=None,
        base_bar_duration=base_bar_duration,
        context_bar_duration=context_bar_duration,
    )
    forbidden = sorted({"atr_bucket", "spread_bucket"} & set(out.columns))
    if forbidden:
        raise RuntimeError(
            f"[MODEL_AGNOSTIC_CANONICAL] TRAIN-fit fields leaked into raw state: {forbidden}"
        )
    return out


def augment_canonical_v3(
    cv3: pd.DataFrame,
    df_m5: pd.DataFrame,
    *,
    rank_reference: TrainRankReferenceV2,
    base_bar_duration: pd.Timedelta = M5_BAR_DURATION,
    context_bar_duration: pd.Timedelta = M5_BAR_DURATION,
) -> pd.DataFrame:
    """Build the complete causal context under one exact TRAIN reference."""

    if not isinstance(rank_reference, TrainRankReferenceV2):
        raise RuntimeError(
            "[CANONICAL_CTX_TRAIN_RANK_REQUIRED] explicit immutable reference required"
        )
    out = _augment_canonical_v3(
        cv3,
        df_m5,
        rank_reference=rank_reference,
        base_bar_duration=base_bar_duration,
        context_bar_duration=context_bar_duration,
    )
    missing = sorted({"atr_bucket", "spread_bucket"} - set(out.columns))
    if missing:
        raise RuntimeError(f"[CANONICAL_CTX_TRAIN_RANK_OUTPUT_MISSING] {missing}")
    return out
