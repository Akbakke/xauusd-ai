#!/usr/bin/env python3
"""V12 live context augmentation for the model-native feature stack.

Background: an older stack was trained against a prebuilt parquet that contained
canonical_v3 columns + ~32 augmented ctx-cont / ctx-cat features computed
by `add_ctx_cont_columns_to_prebuilt.py`. Today the canonical_v3 prebuilt
on disk has been regenerated without those augmentations, so any live
inference must compute them from scratch.

Production context augmentation accepts only scalar projections owned by
``gx1.features.htf_features``.  ``_add_htf_features`` is a compatibility
call-site for historical model-state/dataset consumers, not a formula owner:
it delegates its complete HTF block to that canonical projection.

Features added by the local context owner include:

  Spread / ATR derivations (2):
    - atr_bps                              # Wilder-14 ATR / close * 1e4 (the ONE
                                           #   ATR owner, technical_indicators_v1)
    - spread_bps                           # (ask - bid) / bid * 1e4

  Session features — see gx1.time.session_detector for the SSoT:
    - session_id                           # 0/1/2/3 = ASIA/EU/OVERLAP/US
    - minutes_since_session_open
    - session_change_flag                  # 1 if session changed vs prev bar

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
    - spread_extremes_sum_bps              # ((ask_high-bid_high) + (ask_low-bid_low)) / close * 1e4
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

from gx1.features.micro_structure_v1 import (
    MICRO_WARMUP_PREFIX_FIELDS_V1,
    SPREAD_DYNAMICS_WARMUP_PREFIX_FIELDS_V1,
    compute_micro_structure_features,
    compute_spread_dynamics_features,
)
from gx1.features.model_native_market_context_v1 import (
    MODEL_NATIVE_ATR_WARMUP_PREFIX_FIELDS_V1,
    derive_model_native_atr_spread_bps,
)
from gx1.features.entry_model_native_feature_layers_v1 import (
    SWING_EVENT_LAYER_FEATURE_NAMES,
)
from gx1.features.swing_structure_v1 import (
    SWING_ATR_PERIOD_V1,
    SWING_LOOKBACK_V1,
    compute_swing_structure_features,
)
from gx1.time.session_detector import (
    get_session_id_vectorized,
    get_session_minutes_since_open_vectorized,
    get_session_vectorized,
    decision_availability,
    M5_BAR_DURATION,
)

LOG = logging.getLogger("v12_ctx_augment_live")

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
    # V30 wave 2 (2026-08-18): `is_ASIA` and `minutes_to_next_session_boundary`
    # left MODEL_NATIVE_CTX_CONT_SESSION_FIELDS -- both are exactly recoverable
    # per bar inside session_regime_encoder from fields that stay (the injective
    # (hour_sin, hour_cos) pair, and minutes_since_session_open plus the session
    # length that pair determines). This producer existed to satisfy the ctx
    # contract, so the derivations go with it (rule 10).
    cv3["minutes_since_session_open"] = (
        get_session_minutes_since_open_vectorized(idx)
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
    from gx1.features.htf_features import validate_causal_feature_matrix
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

    from gx1.features.htf_features import (
        MODEL_NATIVE_HTF_CONTEXT_FIELDS_V4,
        project_model_native_htf_context_from_m5_v4,
    )

    projected = project_model_native_htf_context_from_m5_v4(
        m5,
        cv3.index.asi8.astype(np.int64, copy=False),
        decision_bar_duration=base_bar_duration,
    )
    if tuple(projected) != MODEL_NATIVE_HTF_CONTEXT_FIELDS_V4:
        raise RuntimeError("[LIVE_HTF_SOURCE] canonical context field order invalid")
    for name in MODEL_NATIVE_HTF_CONTEXT_FIELDS_V4:
        cv3[name] = projected[name]

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
    """Mutates cv3: spread_bps_delta_1, spread_extremes_sum_bps,
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
        # V30 wave 2 (2026-08-18): these fourteen are NO LONGER ctx contract
        # fields (entry_model_native_signal_v1 v33 retired the ctx copies in
        # favour of their mandatory swing_structure_event_layer twins), but this
        # producer still emits them and this lane still trims on them -- for a
        # different, now explicitly stated reason. Unlike the offline builder,
        # which consumes a precomputed M5 feature surface whose materializer
        # measured the layer warmup on the declared source bytes, this lane
        # hands its OWN (already trimmed) frame to
        # _build_inline_seq_structure_extension as a temp parquet, so the
        # swing-layer warmup has no other owner here. The name tuple is the
        # producer's, so the trim stays correct if another addition gains a
        # prefix.
        + list(SWING_EVENT_LAYER_FEATURE_NAMES)
        # The local price owner needs five preceding rows for change-5 and a
        # classic SMA-seeded EMA5.  Missing history is NaN, never a parked
        # zero or a first-observation EMA.
        + list(MICRO_WARMUP_PREFIX_FIELDS_V1)
        # V30 package 4 (2026-08-13): spread_bps_delta_1 has no value on the
        # first row of any frame (SPREAD_DYNAMICS_CAUSAL_WARMUP_ROWS_V1 = 1).
        # Trim it by contract instead of relying on the much longer D1 warmup
        # above to cover it incidentally. The other two spread-dynamics fields
        # are defined on every row and carry no prefix.
        + list(SPREAD_DYNAMICS_WARMUP_PREFIX_FIELDS_V1)
        # ctx_cont.atr_bps is the classic Wilder-14 ATR (rule 19: the same and
        # only ATR owner every other field on the surface uses), so its first
        # MODEL_NATIVE_ATR_CAUSAL_WARMUP_ROWS_V1 rows are honest NaN instead of
        # the retired partial-window mean. Trim it by contract instead of
        # relying on the much longer D1 warmup above to cover it incidentally.
        + list(MODEL_NATIVE_ATR_WARMUP_PREFIX_FIELDS_V1),
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
