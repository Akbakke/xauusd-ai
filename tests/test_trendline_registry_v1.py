"""Tests for gx1/features/trendline_registry_v1.py (V29 Phase A, stage 1).

Evidence class of every test here: proven from source / synthetic execution.
Synthetic series prove mechanism and causality only (rule 2c); no test makes
a claim about real-tape behaviour.  Real-tape registry cost is a pre-adoption
red gate (design doc §6, measurement 1).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SEQ_LEN
from gx1.features.smc_v1 import SWING_LOOKBACK, _detect_swing_pivots
from gx1.features.trendline_registry_v1 import (
    TRENDLINE_REGISTRY_CHANNEL_FEATURE_NAMES_V1,
    TRENDLINE_REGISTRY_EVENT_FEATURE_NAMES_V1,
    TRENDLINE_REGISTRY_FEATURE_COUNT_V1,
    TRENDLINE_REGISTRY_FEATURE_NAMES_SHA256_V1,
    TRENDLINE_REGISTRY_FEATURE_NAMES_V1,
    TRENDLINE_REGISTRY_SLOT_FEATURE_NAMES_V1,
    TRENDLINE_RETEST_WINDOW_BARS_V1,
    TRENDLINE_SIDE_RESISTANCE,
    TRENDLINE_STATE_ACTIVE,
    TRENDLINE_STATE_CANDIDATE,
    compute_trendline_registry_features_v1,
    fit_trendline_tolerance,
)

# Stage-2 wiring contract: the exact 33-name tuple (design doc B.5 + the V30
# 2026-08-13 additions: per-side ACTIVE counts beside the masks and the
# geomline_bars_since_break memory) and its sha.  Any drift in name, order or
# count must fail here first.
EXPECTED_TRENDLINE_REGISTRY_FEATURE_NAMES_V1 = (
    "geomline_above_active",
    "geomline_above_active_count",
    "geomline_above_dist_atr",
    "geomline_above_slope_atr_per_bar",
    "geomline_above_touch_count",
    "geomline_above_age_bars",
    "geomline_above_last_touch_age_bars",
    "geomline_above_max_dev_atr",
    "geomline_below_active",
    "geomline_below_active_count",
    "geomline_below_dist_atr",
    "geomline_below_slope_atr_per_bar",
    "geomline_below_touch_count",
    "geomline_below_age_bars",
    "geomline_below_last_touch_age_bars",
    "geomline_below_max_dev_atr",
    "geomline_touch_above",
    "geomline_touch_below",
    "geomline_break_up",
    "geomline_break_down",
    "geomline_break_line_touch_count",
    "geomline_break_line_age_bars",
    "geomline_retest_hold_up",
    "geomline_retest_fail_up",
    "geomline_retest_hold_down",
    "geomline_retest_fail_down",
    "geomline_bars_since_break",
    "geomchan_active",
    "geomchan_width_atr",
    "geomchan_pos_0_1",
    "geomchan_slope_atr_per_bar",
    "geomchan_converging",
    "geomchan_apex_proximity",
)
EXPECTED_TRENDLINE_REGISTRY_FEATURE_NAMES_SHA256_V1 = (
    "75086616223a022b3b88e3917cefc6c401ab91be724308459d7f8e8e4eacd84d"
)

WARMUP = 2 * SWING_LOOKBACK + 2  # structural NaN prefix (module contract)


def _support_line_frame(
    n_bars: int,
    pivot_devs: dict,
    *,
    slope: float = 0.05,
    base: float = 100.0,
    offset: float = 1.0,
) -> pd.DataFrame:
    """Rising support line through carved swing lows at proj(bar)+dev."""
    t = np.arange(n_bars, dtype=np.float64)
    proj = base + slope * (t - 10.0)
    close = proj + offset
    high = close + 0.2
    low = close - 0.2
    for bar, dev in pivot_devs.items():
        low[bar] = proj[bar] + dev
    return pd.DataFrame(
        {"high": high, "low": low, "close": close, "atr": np.ones(n_bars)}
    )


def _resistance_line_frame(n_bars: int, pivot_devs: dict) -> pd.DataFrame:
    """Falling resistance line through carved swing highs."""
    t = np.arange(n_bars, dtype=np.float64)
    proj = 106.0 - 0.05 * (t - 10.0)
    close = proj - 1.0
    high = close + 0.2
    low = close - 0.2
    for bar, dev in pivot_devs.items():
        high[bar] = proj[bar] + dev
    return pd.DataFrame(
        {"high": high, "low": low, "close": close, "atr": np.ones(n_bars)}
    )


def _random_walk_frame(n_bars: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 500.0 + np.cumsum(rng.normal(0.0, 0.3, n_bars))
    high = close + np.abs(rng.normal(0.0, 0.2, n_bars)) + 0.05
    low = close - np.abs(rng.normal(0.0, 0.2, n_bars)) - 0.05
    return pd.DataFrame(
        {"high": high, "low": low, "close": close, "atr": np.ones(n_bars)}
    )


def _compute(df, *, band=0.3, seq_len=200, state=None):
    return compute_trendline_registry_features_v1(
        df, timeframe="TEST", seq_len=seq_len, band_atr=band, state=state
    )


# ---------------------------------------------------------------------------
# Name-tuple drift guard
# ---------------------------------------------------------------------------


def test_feature_name_tuple_and_sha_drift_guard():
    # V30 (2026-08-13): 33 = 30 + 2 occupancy counts + 1 break memory.
    assert TRENDLINE_REGISTRY_FEATURE_COUNT_V1 == 33
    assert TRENDLINE_REGISTRY_FEATURE_COUNT_V1 == len(
        TRENDLINE_REGISTRY_FEATURE_NAMES_V1
    )
    assert (
        TRENDLINE_REGISTRY_FEATURE_NAMES_V1
        == EXPECTED_TRENDLINE_REGISTRY_FEATURE_NAMES_V1
    )
    assert (
        TRENDLINE_REGISTRY_FEATURE_NAMES_SHA256_V1
        == EXPECTED_TRENDLINE_REGISTRY_FEATURE_NAMES_SHA256_V1
    )
    assert (
        TRENDLINE_REGISTRY_SLOT_FEATURE_NAMES_V1
        + TRENDLINE_REGISTRY_EVENT_FEATURE_NAMES_V1
        + TRENDLINE_REGISTRY_CHANNEL_FEATURE_NAMES_V1
        == TRENDLINE_REGISTRY_FEATURE_NAMES_V1
    )
    assert len(TRENDLINE_REGISTRY_SLOT_FEATURE_NAMES_V1) == 16
    assert len(TRENDLINE_REGISTRY_EVENT_FEATURE_NAMES_V1) == 11
    assert len(TRENDLINE_REGISTRY_CHANNEL_FEATURE_NAMES_V1) == 6
    assert TRENDLINE_RETEST_WINDOW_BARS_V1 == 2 * SWING_LOOKBACK + 1


# ---------------------------------------------------------------------------
# Pivot one-truth parity
# ---------------------------------------------------------------------------


def test_incremental_pivot_stream_matches_smc_v1_batch_masks():
    df = _random_walk_frame(600, seed=11)
    _, state = _compute(df, band=0.3, seq_len=4096)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    sh_mask, sl_mask = _detect_swing_pivots(high, low, SWING_LOOKBACK)
    expected_support = [(int(j), float(low[j])) for j in np.flatnonzero(sl_mask)]
    expected_resistance = [
        (int(j), float(high[j])) for j in np.flatnonzero(sh_mask)
    ]
    assert state.support_pivots == expected_support
    assert state.resistance_pivots == expected_resistance


# ---------------------------------------------------------------------------
# 3-touch validation + warmup + output contract
# ---------------------------------------------------------------------------


def test_three_touch_validation_and_promotion_bar():
    df = _support_line_frame(60, {10: 0.0, 30: 0.0, 50: 0.0})
    feats, state = _compute(df)
    assert tuple(feats.columns) == TRENDLINE_REGISTRY_FEATURE_NAMES_V1
    assert feats.dtypes.eq(np.float32).all()
    assert feats.index.equals(df.index)
    # structural NaN warmup prefix, then a single fully-finite region
    assert feats.iloc[:WARMUP].isna().all().all()
    assert feats.iloc[WARMUP:].notna().all().all()
    below_active = feats["geomline_below_active"].to_numpy()
    # third pivot lies at bar 50 but participates only from its confirmation
    # bar 53 (= 50 + SWING_LOOKBACK): nothing is ACTIVE before that
    assert (below_active[WARMUP:53] == 0.0).all()
    assert below_active[53] == 1.0
    row = feats.iloc[53]
    assert row["geomline_touch_below"] == 1.0
    assert row["geomline_below_touch_count"] == 3.0
    assert float(row["geomline_below_dist_atr"]) == pytest.approx(1.0)
    assert float(row["geomline_below_slope_atr_per_bar"]) == pytest.approx(0.05)
    assert row["geomline_below_age_bars"] == 40.0  # 53 - (10 + 3)
    assert row["geomline_below_last_touch_age_bars"] == 3.0
    assert float(row["geomline_below_max_dev_atr"]) == pytest.approx(0.0)
    assert feats["geomline_touch_below"].iloc[54] == 0.0
    active = [
        ln for ln in state.active_lines if ln.state == TRENDLINE_STATE_ACTIVE
    ]
    assert len(active) == 1
    assert (active[0].anchor1_bar, active[0].anchor2_bar) == (10, 30)


def test_two_touches_never_activate():
    df = _support_line_frame(60, {10: 0.0, 30: 0.0})
    feats, state = _compute(df)
    assert (feats["geomline_below_active"].iloc[WARMUP:] == 0.0).all()
    assert (feats["geomline_above_active"].iloc[WARMUP:] == 0.0).all()
    assert not state.active_lines
    assert len(state.cand_support) > 0  # the 2-anchor candidate exists


# ---------------------------------------------------------------------------
# Causality / future-invariance
# ---------------------------------------------------------------------------


def test_future_invariance_prefix_rows_never_change():
    df = _random_walk_frame(1200, seed=7)
    full, _ = _compute(df, band=0.3, seq_len=512)
    part, _ = _compute(df.iloc[:800], band=0.3, seq_len=512)
    np.testing.assert_array_equal(
        part.to_numpy(), full.to_numpy()[:800], strict=True
    )


# ---------------------------------------------------------------------------
# Line identity immutability
# ---------------------------------------------------------------------------


def test_active_line_anchors_immutable_under_later_touches():
    df = _support_line_frame(80, {10: 0.0, 30: 0.0, 50: 0.0, 70: 0.1})
    _, state = _compute(df.iloc[:55])
    active = [
        ln for ln in state.active_lines if ln.state == TRENDLINE_STATE_ACTIVE
    ]
    assert len(active) == 1
    snapshot = (
        active[0].line_id,
        active[0].anchor1_bar,
        active[0].anchor1_price,
        active[0].anchor2_bar,
        active[0].anchor2_price,
        active[0].slope,
    )
    feats2, state = _compute(df.iloc[55:], state=state)
    active_after = [
        ln for ln in state.active_lines if ln.state == TRENDLINE_STATE_ACTIVE
    ]
    line = min(active_after, key=lambda ln: ln.line_id)
    assert (
        line.line_id,
        line.anchor1_bar,
        line.anchor1_price,
        line.anchor2_bar,
        line.anchor2_price,
        line.slope,
    ) == snapshot
    # the 4th pivot (bar 70, confirmed at 73) updates evidence, not identity
    assert line.touch_count == 4
    assert line.max_dev_atr == pytest.approx(0.1)
    assert feats2.loc[73, "geomline_touch_below"] == 1.0
    assert feats2.loc[73, "geomline_below_touch_count"] == 4.0
    slopes = feats2.loc[55:, "geomline_below_slope_atr_per_bar"]
    assert slopes.nunique() == 1  # projection geometry never re-fitted


# ---------------------------------------------------------------------------
# First break fires once; retest hold / fail / expiry
# ---------------------------------------------------------------------------


def _break_frame(retest: str) -> pd.DataFrame:
    df = _support_line_frame(70, {10: 0.0, 30: 0.0, 50: 0.0})
    t = np.arange(70, dtype=np.float64)
    proj = 100.0 + 0.05 * (t - 10.0)
    for bar in (60, 61):
        df.loc[bar, "close"] = proj[bar] - 0.8
        df.loc[bar, "high"] = proj[bar] - 0.6
        df.loc[bar, "low"] = proj[bar] - 1.0
    if retest == "hold":
        df.loc[62, "close"] = proj[62] - 0.5
        df.loc[62, "high"] = proj[62] - 0.2
        df.loc[62, "low"] = proj[62] - 0.7
    elif retest == "fail":
        df.loc[62, "close"] = proj[62] + 0.5
        df.loc[62, "high"] = proj[62] + 0.7
        df.loc[62, "low"] = proj[62] - 0.1
    elif retest == "expiry":
        for bar in range(62, 70):
            df.loc[bar, "close"] = proj[bar] - 0.8
            df.loc[bar, "high"] = proj[bar] - 0.6
            df.loc[bar, "low"] = proj[bar] - 1.0
    return df


def test_first_break_fires_exactly_once_with_broken_line_attributes():
    feats, _ = _compute(_break_frame("hold"))
    break_down = feats["geomline_break_down"]
    assert break_down.iloc[60] == 1.0
    assert break_down.sum() == 1.0
    # V30 package 8A (2026-08-13): the held retest at bar 62 FLIPS the broken
    # support into an ACTIVE resistance instead of deleting the line, so the
    # very next bar — whose close returns to proj + 1.0, beyond the 0.3 band —
    # is a genuine first break of the line in its NEW role.  Before the flip
    # repair the line no longer existed and this event could not fire.
    assert feats["geomline_break_up"].sum() == 1.0
    assert feats["geomline_break_up"].iloc[63] == 1.0
    assert feats["geomline_break_line_touch_count"].iloc[60] == 3.0
    assert feats["geomline_break_line_age_bars"].iloc[60] == 47.0  # 60-(10+3)
    assert feats["geomline_break_line_touch_count"].iloc[61] == 0.0
    # a BROKEN line leaves the nearest-ACTIVE slots immediately
    assert feats["geomline_below_active"].iloc[60] == 0.0
    assert feats["geomline_above_active"].iloc[60] == 0.0


def test_retest_hold_after_break():
    feats, state = _compute(_break_frame("hold"))
    hold = feats["geomline_retest_hold_down"]
    assert hold.iloc[62] == 1.0
    assert hold.sum() == 1.0
    assert feats["geomline_retest_fail_down"].sum() == 0.0
    assert feats["geomline_retest_hold_up"].sum() == 0.0
    # resolved retest retires the line
    assert not [ln for ln in state.active_lines if ln.break_bar == 60]


def test_retest_fail_after_break():
    feats, _ = _compute(_break_frame("fail"))
    fail = feats["geomline_retest_fail_down"]
    assert fail.iloc[62] == 1.0
    assert fail.sum() == 1.0
    assert feats["geomline_retest_hold_down"].sum() == 0.0


def test_retest_window_expiry_retires_line_silently():
    feats, state = _compute(_break_frame("expiry"))
    assert feats["geomline_retest_hold_down"].sum() == 0.0
    assert feats["geomline_retest_fail_down"].sum() == 0.0
    assert not [ln for ln in state.active_lines if ln.break_bar == 60]


def test_resistance_mirror_break_up_and_retest_hold_up():
    df = _resistance_line_frame(70, {10: 0.0, 30: 0.0, 50: 0.0})
    t = np.arange(70, dtype=np.float64)
    proj = 106.0 - 0.05 * (t - 10.0)
    for bar in (60, 61):
        df.loc[bar, "close"] = proj[bar] + 0.8
        df.loc[bar, "high"] = proj[bar] + 1.0
        df.loc[bar, "low"] = proj[bar] + 0.6
    df.loc[62, "close"] = proj[62] + 0.5
    df.loc[62, "high"] = proj[62] + 0.7
    df.loc[62, "low"] = proj[62] + 0.2
    feats, _ = _compute(df)
    row53 = feats.iloc[53]
    assert row53["geomline_above_active"] == 1.0
    assert row53["geomline_touch_above"] == 1.0
    assert float(row53["geomline_above_slope_atr_per_bar"]) == pytest.approx(
        -0.05
    )
    assert float(row53["geomline_above_dist_atr"]) == pytest.approx(1.0)
    assert feats["geomline_break_up"].iloc[60] == 1.0
    assert feats["geomline_break_up"].sum() == 1.0
    assert feats["geomline_break_line_touch_count"].iloc[60] == 3.0
    assert feats["geomline_retest_hold_up"].iloc[62] == 1.0
    assert feats["geomline_retest_hold_up"].sum() == 1.0
    assert feats["geomline_retest_fail_up"].sum() == 0.0


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


def test_parallel_channel_pairing_via_return_rail():
    n_bars = 68
    t = np.arange(n_bars, dtype=np.float64)
    proj_s = 100.0 + 0.05 * (t - 10.0)
    close = proj_s + 1.5
    high = close + 0.2
    low = close - 0.2
    for bar in (10, 30, 50):
        low[bar] = proj_s[bar]
    for bar in (20, 40, 60):
        high[bar] = proj_s[bar] + 3.0
    df = pd.DataFrame(
        {"high": high, "low": low, "close": close, "atr": np.ones(n_bars)}
    )
    feats, _ = _compute(df)
    # before any ACTIVE line: no channel
    assert feats["geomchan_active"].iloc[50] == 0.0
    # support ACTIVE at 53 + two parallel opposite pivots stored => channel,
    # even though the above slot is still empty (route: parallel return rail)
    row55 = feats.iloc[55]
    assert row55["geomline_above_active"] == 0.0
    assert row55["geomchan_active"] == 1.0
    assert float(row55["geomchan_width_atr"]) == pytest.approx(3.0)
    assert float(row55["geomchan_pos_0_1"]) == pytest.approx(0.5)
    assert float(row55["geomchan_slope_atr_per_bar"]) == pytest.approx(0.05)
    assert row55["geomchan_converging"] == 0.0
    assert row55["geomchan_apex_proximity"] == 0.0
    # resistance line promotes at 63 (its own 3rd touch)
    assert feats["geomline_touch_above"].iloc[63] == 1.0
    row64 = feats.iloc[64]
    assert row64["geomline_above_active"] == 1.0
    assert row64["geomline_above_touch_count"] == 3.0
    assert row64["geomchan_active"] == 1.0
    assert float(row64["geomchan_width_atr"]) == pytest.approx(3.0)
    assert row64["geomchan_converging"] == 0.0


def test_converging_pair_triangle_with_apex_proximity():
    n_bars = 66
    t = np.arange(n_bars, dtype=np.float64)
    proj_s = 100.0 + 0.05 * (t - 10.0)
    proj_r = 106.0 - 0.05 * (t - 20.0)
    close = 103.25 + 0.002 * t
    high = close + 0.2
    low = close - 0.2
    for bar in (10, 30, 50):
        low[bar] = proj_s[bar]
    for bar in (20, 40, 60):
        high[bar] = proj_r[bar]
    df = pd.DataFrame(
        {"high": high, "low": low, "close": close, "atr": np.ones(n_bars)}
    )
    feats, _ = _compute(df, band=0.15)
    row = feats.iloc[64]
    assert row["geomline_below_active"] == 1.0
    assert row["geomline_above_active"] == 1.0
    assert row["geomchan_active"] == 1.0
    assert row["geomchan_converging"] == 1.0
    assert float(row["geomchan_width_atr"]) == pytest.approx(1.1)
    expected_pos = (float(close[64]) - float(proj_s[64])) / 1.1
    assert float(row["geomchan_pos_0_1"]) == pytest.approx(expected_pos)
    assert float(row["geomchan_slope_atr_per_bar"]) == pytest.approx(0.0)
    # rails intersect at bar 75 -> apex proximity 1 - (75-64)/seq_len
    assert float(row["geomchan_apex_proximity"]) == pytest.approx(
        1.0 - 11.0 / 200.0
    )


# ---------------------------------------------------------------------------
# Tolerance fit
# ---------------------------------------------------------------------------


def test_tolerance_fit_deterministic_with_provenance():
    df = _random_walk_frame(1500, seed=3)
    first = fit_trendline_tolerance(df, timeframe="TEST", seq_len=512)
    second = fit_trendline_tolerance(df, timeframe="TEST", seq_len=512)
    assert first == second
    assert first["band_atr"] > 0.0
    assert first["n_candidates_measured"] > 0
    assert first["n_support_pivots"] > 0
    assert first["n_resistance_pivots"] > 0
    assert first["n_bars"] == 1500
    assert first["statistic"] == "median_abs_first_subsequent_pivot_deviation_atr"
    assert first["schema_version"] == "trendline_tolerance_fit_v1"
    assert len(first["contract_sha256"]) == 64


def test_tolerance_fit_publishes_the_implied_validation_rate():
    """The band is the median of the population it then judges, so the share of
    arbitrary 2-pivot pairs it promotes to a "validated" line is >= 0.5 by the
    definition of a median.  Publishing the measured rate (2026-08-13) makes
    that degeneracy visible in every frozen constants manifest instead of
    provable only from source (audit §0b)."""

    df = _random_walk_frame(1500, seed=3)
    payload = fit_trendline_tolerance(df, timeframe="TEST", seq_len=512)
    rate = payload["implied_validation_rate"]
    assert 0.5 <= rate <= 1.0
    assert "implied_validation_rate_definition" in payload
    # It is exactly the share of the reported population under the band.
    n_measured = payload["n_candidates_measured"]
    assert abs(rate * n_measured - round(rate * n_measured)) < 1e-9


def test_tolerance_fit_fails_closed_on_empty_population():
    n_bars = 120
    t = np.arange(n_bars, dtype=np.float64)
    close = 100.0 + 0.1 * t  # strictly trending: no swing pivots at all
    df = pd.DataFrame(
        {
            "high": close + 0.1,
            "low": close - 0.05,
            "close": close,
            "atr": np.ones(n_bars),
        }
    )
    with pytest.raises(RuntimeError, match="TRENDLINE_TOLERANCE_FIT_EMPTY"):
        fit_trendline_tolerance(df, timeframe="TEST", seq_len=64)


# ---------------------------------------------------------------------------
# Chunk-carry exactness
# ---------------------------------------------------------------------------


def test_chunked_processing_bit_identical_to_one_shot():
    df = _random_walk_frame(1200, seed=7)
    band = fit_trendline_tolerance(df, timeframe="TEST", seq_len=512)[
        "band_atr"
    ]
    one_shot, _ = _compute(df, band=band, seq_len=512)
    state = None
    pieces = []
    # boundaries deliberately inside pivot-confirmation windows
    for chunk in (df.iloc[:400], df.iloc[400:407], df.iloc[407:]):
        feats, state = _compute(chunk, band=band, seq_len=512, state=state)
        pieces.append(feats)
    np.testing.assert_array_equal(
        pd.concat(pieces).to_numpy(), one_shot.to_numpy(), strict=True
    )
    # the fitted band must actually exercise the registry on this tape
    assert one_shot["geomline_below_active"].sum() > 0.0 or (
        one_shot["geomline_above_active"].sum() > 0.0
    )


# ---------------------------------------------------------------------------
# ATR warmup prefix
# ---------------------------------------------------------------------------


def test_atr_warmup_prefix_is_nan_and_registry_starts_after():
    df = _support_line_frame(80, {10: 0.0, 30: 0.0, 50: 0.0, 70: 0.0})
    atr = np.ones(80)
    atr[:30] = np.nan
    df["atr"] = atr
    feats, _ = _compute(df)
    assert feats.iloc[:30].isna().all().all()
    assert feats.iloc[30:].notna().all().all()
    # pivots inside the ATR-unavailable prefix never enter the registry:
    # the first ACTIVE line is (30, 50) validated by pivot 70 at bar 73
    assert (feats["geomline_below_active"].iloc[30:73] == 0.0).all()
    assert feats["geomline_below_active"].iloc[73] == 1.0


# ---------------------------------------------------------------------------
# Fail-closed validation
# ---------------------------------------------------------------------------


def test_fail_closed_inputs():
    df = _support_line_frame(40, {10: 0.0})
    with pytest.raises(RuntimeError, match="TRENDLINE_BAND_INVALID"):
        _compute(df, band=0.0)
    with pytest.raises(RuntimeError, match="TRENDLINE_BAND_INVALID"):
        _compute(df, band=float("nan"))
    with pytest.raises(RuntimeError, match="TRENDLINE_BAND_INVALID"):
        _compute(df, band=True)
    with pytest.raises(RuntimeError, match="TRENDLINE_SEQ_LEN_INVALID"):
        _compute(df, seq_len=0)
    with pytest.raises(RuntimeError, match="TRENDLINE_TIMEFRAME_INVALID"):
        compute_trendline_registry_features_v1(
            df, timeframe="", seq_len=200, band_atr=0.3
        )
    with pytest.raises(RuntimeError, match="TRENDLINE_SOURCE_MISSING"):
        _compute(df.drop(columns=["close"]))
    bad = df.copy()
    bad.loc[5, "close"] = np.nan
    with pytest.raises(RuntimeError, match="TRENDLINE_SOURCE_NONFINITE"):
        _compute(bad)
    gap = df.copy()
    gap_atr = np.ones(40)
    gap_atr[10] = np.nan
    gap["atr"] = gap_atr
    with pytest.raises(
        RuntimeError, match="TRENDLINE_ATR_AVAILABILITY_INVALID"
    ):
        _compute(gap)
    with pytest.raises(RuntimeError, match="TRENDLINE_SOURCE_EMPTY"):
        _compute(df.iloc[:0])


def test_fail_closed_state_carry():
    df = _support_line_frame(60, {10: 0.0, 30: 0.0, 50: 0.0})
    _, state = _compute(df.iloc[:30])
    # config drift between chunks is terminal
    with pytest.raises(RuntimeError, match="TRENDLINE_STATE_CONFIG_MISMATCH"):
        _compute(df.iloc[30:], band=0.4, state=state)
    # non-contiguous / rewound chunk is terminal
    with pytest.raises(
        RuntimeError, match="TRENDLINE_STATE_INDEX_DISCONTINUITY"
    ):
        _compute(df.iloc[:30], state=state)
    with pytest.raises(RuntimeError, match="TRENDLINE_STATE_TYPE_INVALID"):
        _compute(df.iloc[30:], state=object())


# ---------------------------------------------------------------------------
# Bounded compute (synthetic micro-benchmark)
# ---------------------------------------------------------------------------


def test_v30_package_8a_retest_hold_flips_polarity_and_keeps_the_line():
    """V30 package 8A (2026-08-13) — "old support is new resistance".

    Before this repair a RETEST_HOLD emitted its one-bar impulse and DELETED
    the line on the very bar it proved itself as flipped resistance (fidelity
    audit §5), so the most-used S/R construct existed as an event and never as
    an object.  A HOLD now returns the line to ACTIVE with its identity,
    anchors and touch history intact and only its side flipped.  A FAIL is
    unchanged: the break stands and the line is retired.
    """

    feats, state = _compute(_break_frame("hold"))
    line = next(ln for ln in state.active_lines if ln.anchor1_bar == 10)
    # Identity is untouched by the flip: same anchors, hence same projection.
    assert (line.anchor1_bar, line.anchor2_bar) == (10, 30)
    assert line.side == TRENDLINE_SIDE_RESISTANCE  # flipped from support
    # The retest bar counts as a touch through the same touch bookkeeping the
    # intra-band path uses, so the staleness clock is renewed rather than the
    # line dying on the next bar.
    assert 62 in line.touch_bars
    assert line.touch_count == 4
    assert line.last_touch_bar == 62
    # The flipped line is a real ACTIVE object on the bar of the flip: it
    # projects above the close, so it occupies the ABOVE slot.
    assert feats["geomline_retest_hold_down"].iloc[62] == 1.0
    assert feats["geomline_above_active"].iloc[62] == 1.0
    assert feats["geomline_above_active_count"].iloc[62] == 1.0
    assert feats["geomline_below_active"].iloc[62] == 0.0
    # The generic touch impulse is deliberately NOT raised on the flip bar:
    # the retest-hold event is that bar's specific impulse.
    assert feats["geomline_touch_above"].iloc[62] == 0.0
    assert feats["geomline_touch_below"].iloc[62] == 0.0

    # A FAILED retest still retires the line (no flip, no surviving object).
    feats_fail, state_fail = _compute(_break_frame("fail"))
    assert feats_fail["geomline_retest_fail_down"].iloc[62] == 1.0
    assert not [
        ln for ln in state_fail.active_lines if ln.anchor1_bar == 10
    ]


def test_micro_benchmark_per_bar_update_under_loose_bound():
    """Loose synthetic cost guard only.

    Bound origin (rule 2a, UNCHANGED): the chart report B.7 algebraic estimate
    declares 'minutes' of overhead on the ~6-year M5 tape (~450k bars);
    10 minutes / 450k bars ~= 1.3 ms/bar, rounded up to a loose 1.5 ms/bar
    guard.  This is measured on synthetic data and therefore proves only that
    the code runs within the bound here (rule 2c); the real-tape registry cost
    during the Phase-A build is a pre-adoption red gate (design doc §6,
    measurement 1).

    Window origin (rule 2g, moved 2026-08-13 by V30 package 8A): the
    measurement is taken at the window the registry is actually run with.
    ``seq_len`` is the per-TF model sequence length (module docstring), and
    every recipe/fixture in this repository binds ``trendline_seq_len`` to
    ``MODEL_NATIVE_SEQ_LEN`` for the entry M5/513 lane, while the per-TF lanes
    use their pyramid seq_lens (M5 = 16).  This test previously measured at 512
    — 5.3x the largest declared window and a window no lane uses — so it was
    not measuring where the decision is made.

    Measured on this fixture, 2026-08-13 `[M-synthetic]`, before -> after the
    package-8A retest-hold POLARITY FLIP (ms/bar, final ACTIVE-line count):
        seq_len  16: 0.040 (0 lines)    -> 0.042 (1 line)
        seq_len  96: 0.109 (41 lines)   -> 0.169 (65 lines)
        seq_len 512: 0.954 (1110 lines) -> 2.546 (2584 lines)
    Stated uninvited (rule 25a): the flip keeps held-retest lines alive instead
    of deleting them, so the ACTIVE-line population grows ~1.6x at the declared
    window and ~2.3x at 512, and per-bar cost is superlinear in seq_len because
    both the per-bar update and the emission are O(active lines).  At 512 the
    result is 2.55 ms/bar — ABOVE this guard.  No lane runs there, but if one
    ever does, this cost is real and the guard must be re-derived from a
    declared budget rather than relaxed.
    """
    seq_len = MODEL_NATIVE_SEQ_LEN
    df = _random_walk_frame(5000, seed=13)
    band = fit_trendline_tolerance(df, timeframe="TEST", seq_len=seq_len)[
        "band_atr"
    ]
    start = time.perf_counter()
    _compute(df, band=band, seq_len=seq_len)
    elapsed = time.perf_counter() - start
    per_bar = elapsed / 5000.0
    assert per_bar < 1.5e-3, f"per-bar update {per_bar * 1e3:.3f} ms"


# ---------------------------------------------------------------------------
# Forward-realized line-hold labels (stage 2; chart report B.8)
# ---------------------------------------------------------------------------


def test_touch_hold_labels_are_event_masked_and_forward_realized():
    from gx1.features.trendline_registry_v1 import (
        TRENDLINE_TOUCH_HOLD_LABEL_COLUMNS_V1,
        compute_trendline_touch_hold_labels_v1,
    )

    # Rising support validated by its 3rd pivot touch (bar 50, confirmed at
    # 53) and BROKEN at bar 60 — within a 10-bar forward horizon of the
    # touch, so the touch did NOT hold.
    broken = _break_frame("expiry")
    labels = compute_trendline_touch_hold_labels_v1(
        broken, seq_len=200, band_atr=0.3, horizon_bars=10
    )
    assert tuple(labels.columns) == TRENDLINE_TOUCH_HOLD_LABEL_COLUMNS_V1
    support_mask = labels["y_line_support_touch_mask"].to_numpy()
    support_held = labels["y_line_support_touch_held"].to_numpy()
    assert support_mask[53] == 1.0
    assert support_held[53] == 0.0
    # Labels exist ONLY on touch-event bars (the y_side_mask pattern).
    assert support_mask.sum() == 1.0
    assert labels["y_line_resistance_touch_mask"].to_numpy().sum() == 0.0

    # Same line, no break within the horizon: the touch held.
    intact = _support_line_frame(70, {10: 0.0, 30: 0.0, 50: 0.0})
    held_labels = compute_trendline_touch_hold_labels_v1(
        intact, seq_len=200, band_atr=0.3, horizon_bars=10
    )
    assert held_labels["y_line_support_touch_mask"].to_numpy()[53] == 1.0
    assert held_labels["y_line_support_touch_held"].to_numpy()[53] == 1.0

    # An unobserved forward window is undecidable: masked out, never a
    # placeholder outcome (rule 2e).
    undecidable = compute_trendline_touch_hold_labels_v1(
        intact, seq_len=200, band_atr=0.3, horizon_bars=30
    )
    assert undecidable["y_line_support_touch_mask"].to_numpy().sum() == 0.0
