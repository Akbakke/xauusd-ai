"""Tests for gx1/features/trendline_registry_v1.py (V29 Phase A, stage 1).

Evidence class of every test here: proven from source / synthetic execution.
Synthetic series prove mechanism and causality only (rule 2c); no test makes
a claim about real-tape behaviour.  Real-tape registry cost is a pre-adoption
red gate (design doc §6, measurement 1).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SEQ_LEN
from gx1.contracts.entry_exit_production_architecture_v1 import (
    PRODUCTION_EXIT_SEQUENCE_BARS,
)
from gx1.features.smc_v1 import SWING_LOOKBACK, _detect_swing_pivots
from gx1.features.trendline_registry_v1 import (
    TRENDLINE_REGISTRY_CHANNEL_FEATURE_NAMES_V1,
    TRENDLINE_REGISTRY_EVENT_FEATURE_NAMES_V1,
    TRENDLINE_REGISTRY_FEATURE_COUNT_V1,
    TRENDLINE_REGISTRY_FEATURE_NAMES_SHA256_V1,
    TRENDLINE_REGISTRY_FEATURE_NAMES_V1,
    TRENDLINE_REGISTRY_SLOT_FEATURE_NAMES_V1,
    TRENDLINE_SIDE_RESISTANCE,
    TRENDLINE_SIDE_SUPPORT,
    TRENDLINE_STATE_ACTIVE,
    TrendlineV1,
    compute_trendline_registry_features_v1 as _compute_trendline,
    fit_trendline_registry_hyperparameters_v1,
)

# Stage-2 wiring contract: the exact declared name tuple (design doc B.5, the
# V30 2026-08-13 additions — per-side ACTIVE counts and the
# geomline_bars_since_break memory — and the 2026-08-15 retirement of the
# geomline_{above,below}_active presence masks, which were by construction the
# ">= 1" indicator of the count beside them) and its sha.  Any drift in name,
# order or count must fail here first.
EXPECTED_TRENDLINE_REGISTRY_FEATURE_NAMES_V1 = (
    "geomline_above_active_count",
    "geomline_above_dist_atr",
    "geomline_above_slope_atr_per_bar",
    "geomline_above_touch_count",
    "geomline_above_age_bars",
    "geomline_above_last_touch_age_bars",
    "geomline_above_max_dev_atr",
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
    "9415d0e2757579b717e2c5335fdbb8c3fab41a68c642c7054b1836214379ad0c"
)

WARMUP = 2 * SWING_LOOKBACK + 2  # structural NaN prefix (module contract)


def compute_trendline_registry_features_v1(df, *, seq_len, **kwargs):
    kwargs.setdefault("identity_expiry_bars", seq_len)
    return _compute_trendline(df, seq_len=seq_len, **kwargs)


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


def _fit_source(tmp_path: Path, *, clock: str) -> dict:
    paths = {}
    for name in ("source", "tape", "pair"):
        path = (tmp_path / f"{name}.json").resolve()
        path.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
        paths[name] = path
    return {
        "source_artifact": str(paths["source"]),
        "source_sha256": hashlib.sha256(paths["source"].read_bytes()).hexdigest(),
        "source_schema_version": "synthetic_closed_ohlcv_v1",
        "source_lane": clock,
        "tape_manifest_artifact": str(paths["tape"]),
        "tape_manifest_sha256": hashlib.sha256(paths["tape"].read_bytes()).hexdigest(),
        "pair_manifest_artifact": str(paths["pair"]),
        "pair_manifest_sha256": hashlib.sha256(paths["pair"].read_bytes()).hexdigest(),
        "train_split_id": "synthetic:TRAIN",
        "declared_train_window_start": "2020-01-01T00:00:00+00:00",
        "declared_train_window_end": "2020-01-06T04:55:00+00:00",
    }


def _compute(df, *, band=0.3, seq_len=200, state=None):
    return compute_trendline_registry_features_v1(
        df, timeframe="TEST", seq_len=seq_len, band_atr=band, state=state
    )


# ---------------------------------------------------------------------------
# Name-tuple drift guard
# ---------------------------------------------------------------------------


def test_feature_name_tuple_and_sha_drift_guard():
    assert TRENDLINE_REGISTRY_FEATURE_COUNT_V1 == 31
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
    assert len(TRENDLINE_REGISTRY_SLOT_FEATURE_NAMES_V1) == 14
    assert len(TRENDLINE_REGISTRY_EVENT_FEATURE_NAMES_V1) == 11
    assert len(TRENDLINE_REGISTRY_CHANNEL_FEATURE_NAMES_V1) == 6


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
    assert feats.iloc[WARMUP:].drop(
        columns=["geomline_bars_since_break"]
    ).notna().all().all()
    below_active_count = feats["geomline_below_active_count"].to_numpy()
    # third pivot lies at bar 50 but participates only from its confirmation
    # bar 53 (= 50 + SWING_LOOKBACK): nothing is ACTIVE before that
    assert (below_active_count[WARMUP:53] == 0.0).all()
    assert below_active_count[53] == 1.0
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


def test_exact_line_violation_death_prevents_later_candidate_promotion():
    intact = _support_line_frame(60, {10: 0.0, 30: 0.0, 50: 0.0})
    intact_population: list[float] = []
    _intact_features, intact_state = compute_trendline_registry_features_v1(
        intact,
        timeframe="TEST",
        seq_len=200,
        band_atr=0.3,
        candidate_deviation_log=intact_population,
    )
    assert intact_population == [0.0]
    assert any(
        line.state == TRENDLINE_STATE_ACTIVE
        and (line.anchor1_bar, line.anchor2_bar) == (10, 30)
        for line in intact_state.active_lines
    )

    violated = intact.copy()
    projection = 100.0 + 0.05 * (40.0 - 10.0)
    violated.loc[40, ["close", "high", "low"]] = [
        projection - 1.0,
        projection - 0.8,
        projection - 1.2,
    ]
    violated_population: list[float] = []
    _violated_features, violated_state = (
        compute_trendline_registry_features_v1(
            violated,
            timeframe="TEST",
            seq_len=200,
            band_atr=0.3,
            candidate_deviation_log=violated_population,
        )
    )
    # The break bar itself becomes a later confirmed swing low and creates new
    # two-anchor identities.  Their later deviations remain observable, but
    # the violated (10, 30) identity's otherwise exact 0.0 validation is gone.
    assert violated_population
    assert 0.0 not in violated_population
    assert all(value > 0.3 for value in violated_population)
    assert not [
        line
        for line in violated_state.active_lines
        if (line.anchor1_bar, line.anchor2_bar) == (10, 30)
    ]


def test_two_touches_never_activate():
    df = _support_line_frame(60, {10: 0.0, 30: 0.0})
    feats, state = _compute(df)
    assert (feats["geomline_below_active_count"].iloc[WARMUP:] == 0.0).all()
    assert (feats["geomline_above_active_count"].iloc[WARMUP:] == 0.0).all()
    assert not state.active_lines
    assert len(state.cand_support) > 0  # the 2-anchor candidate exists


def test_expired_trendline_candidate_cannot_promote_before_prune():
    """A later matching pivot cannot revive anchors outside ``seq_len``."""

    seed = pd.DataFrame(
        {"high": [102.0], "low": [101.0], "close": [101.5], "atr": [1.0]},
        index=[0],
    )
    _, state = _compute(seed, seq_len=20)
    state.bar_count = 100
    state.last_index = 0
    # Six carried bars plus the call's bar form the confirmation window.  Its
    # center (registry bar 97) is a support pivot exactly on the old line.
    state.buf_high[:] = [102.0] * 6
    state.buf_low[:] = [101.0, 101.0, 101.0, 100.0, 101.0, 101.0]
    state.buf_atr[:] = [1.0] * 6
    state.cand_support.extend_pairs(
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([100.0], dtype=np.float64),
        5,
        100.0,
    )
    state.next_line_id = 1

    row = pd.DataFrame(
        {"high": [102.0], "low": [101.0], "close": [101.5], "atr": [1.0]},
        index=[100],
    )
    features, state = _compute(row, seq_len=20, state=state)

    assert not [line for line in state.active_lines if line.line_id == 0]
    assert features.loc[100, "geomline_touch_below"] == 0.0


def test_parallel_channel_ignores_opposite_pivots_outside_seq_len():
    """The parallel-rail route cannot consume stale opposite-side pivots."""

    seed = pd.DataFrame(
        {"high": [102.0], "low": [101.0], "close": [101.5], "atr": [1.0]},
        index=[0],
    )
    _, state = _compute(seed, seq_len=20)
    state.bar_count = 100
    state.last_index = 0
    state.buf_high.clear()
    state.buf_low.clear()
    state.buf_atr.clear()
    state.resistance_pivots[:] = [(0, 102.0), (5, 102.0)]
    state.active_lines[:] = [
        TrendlineV1(
            line_id=0,
            side=TRENDLINE_SIDE_SUPPORT,
            anchor1_bar=90,
            anchor1_price=100.0,
            anchor2_bar=95,
            anchor2_price=100.0,
            state=TRENDLINE_STATE_ACTIVE,
            touch_count=3,
            last_touch_bar=99,
            max_dev_atr=0.0,
            touch_bars={90, 95, 99},
        )
    ]
    row = pd.DataFrame(
        {"high": [101.2], "low": [100.8], "close": [101.0], "atr": [1.0]},
        index=[100],
    )

    features, state = _compute(row, seq_len=20, state=state)

    assert state.resistance_pivots == []
    assert features.loc[100, "geomline_below_active_count"] == 1.0
    assert features.loc[100, "geomchan_active"] == 0.0


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


def test_runtime_candidate_population_is_prefix_causal():
    df = _random_walk_frame(1200, seed=17)
    full_population: list[float] = []
    prefix_population: list[float] = []
    compute_trendline_registry_features_v1(
        df,
        timeframe="TEST",
        seq_len=512,
        band_atr=0.3,
        candidate_deviation_log=full_population,
    )
    compute_trendline_registry_features_v1(
        df.iloc[:800],
        timeframe="TEST",
        seq_len=512,
        band_atr=0.3,
        candidate_deviation_log=prefix_population,
    )
    np.testing.assert_array_equal(
        np.asarray(prefix_population, dtype=np.float64),
        np.asarray(full_population[: len(prefix_population)], dtype=np.float64),
        strict=True,
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
    assert np.isnan(feats["geomline_bars_since_break"].iloc[59])
    assert feats["geomline_bars_since_break"].iloc[60] == 0.0
    assert feats["geomline_bars_since_break"].iloc[61] == 1.0
    # a BROKEN line leaves the nearest-ACTIVE slots immediately
    assert feats["geomline_below_active_count"].iloc[60] == 0.0
    assert feats["geomline_above_active_count"].iloc[60] == 0.0


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


def test_retest_stays_armed_past_old_seven_bar_window():
    feats, state = _compute(_break_frame("expiry"))
    assert feats["geomline_retest_hold_down"].sum() == 0.0
    assert feats["geomline_retest_fail_down"].sum() == 0.0
    pending = [ln for ln in state.active_lines if ln.break_bar == 60]
    assert len(pending) == 1
    assert state.bar_count - 1 - pending[0].break_bar == 9


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
    assert row53["geomline_above_active_count"] == 1.0
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
    assert row55["geomline_above_active_count"] == 0.0
    assert row55["geomchan_active"] == 1.0
    assert float(row55["geomchan_width_atr"]) == pytest.approx(3.0)
    assert float(row55["geomchan_pos_0_1"]) == pytest.approx(0.5)
    assert float(row55["geomchan_slope_atr_per_bar"]) == pytest.approx(0.05)
    assert row55["geomchan_converging"] == 0.0
    assert row55["geomchan_apex_proximity"] == 0.0
    # resistance line promotes at 63 (its own 3rd touch)
    assert feats["geomline_touch_above"].iloc[63] == 1.0
    row64 = feats.iloc[64]
    assert row64["geomline_above_active_count"] == 1.0
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
    assert row["geomline_below_active_count"] == 1.0
    assert row["geomline_above_active_count"] == 1.0
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


@pytest.mark.parametrize(
    "seq_len",
    (MODEL_NATIVE_SEQ_LEN, PRODUCTION_EXIT_SEQUENCE_BARS),
    ids=("entry_m5_96", "exit_m1_480"),
)
def test_chunked_processing_bit_identical_to_one_shot(seq_len: int):
    df = _random_walk_frame(1200, seed=7)
    band = 0.3
    one_shot, _ = _compute(df, band=band, seq_len=seq_len)
    state = None
    pieces = []
    # boundaries deliberately inside pivot-confirmation windows
    for chunk in (df.iloc[:400], df.iloc[400:407], df.iloc[407:]):
        feats, state = _compute(chunk, band=band, seq_len=seq_len, state=state)
        pieces.append(feats)
    np.testing.assert_array_equal(
        pd.concat(pieces).to_numpy(), one_shot.to_numpy(), strict=True
    )
    # the fitted band must actually exercise the registry on this tape
    assert one_shot["geomline_below_active_count"].sum() > 0.0 or (
        one_shot["geomline_above_active_count"].sum() > 0.0
    )


def test_hyperfit_binds_runtime_population_and_learned_expiry(tmp_path: Path):
    df = _random_walk_frame(1500, seed=3)
    df.index = pd.date_range(
        "2020-01-01", periods=len(df), freq="5min", tz="UTC"
    )
    payload = fit_trendline_registry_hyperparameters_v1(
        df,
        timeframe="M5",
        seq_len=512,
        inner_fit_end_exclusive=900,
        source_provenance=_fit_source(tmp_path, clock="M5"),
    )
    assert payload["selected_threshold_atr"] > 0.0
    assert payload["learned_expiry_bars"] > 0
    assert payload["population_configuration"] == {
        "owner": "trendline_exact_runtime_candidate_population_v1",
        "seq_len": 512,
        "swing_lookback": SWING_LOOKBACK,
        "identity_expiry_bars": payload["learned_expiry_bars"],
    }
    features, _state = compute_trendline_registry_features_v1(
        df,
        timeframe="M5",
        seq_len=512,
        band_atr=payload["selected_threshold_atr"],
        identity_expiry_bars=payload["learned_expiry_bars"],
    )
    assert features.shape == (len(df), TRENDLINE_REGISTRY_FEATURE_COUNT_V1)


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
    assert feats.iloc[30:].drop(
        columns=["geomline_bars_since_break"]
    ).notna().all().all()
    # pivots inside the ATR-unavailable prefix never enter the registry:
    # the first ACTIVE line is (30, 50) validated by pivot 70 at bar 73
    assert (feats["geomline_below_active_count"].iloc[30:73] == 0.0).all()
    assert feats["geomline_below_active_count"].iloc[73] == 1.0


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
    assert feats["geomline_above_active_count"].iloc[62] == 1.0
    assert feats["geomline_below_active_count"].iloc[62] == 0.0
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


@pytest.mark.parametrize(
    "seq_len",
    (MODEL_NATIVE_SEQ_LEN, PRODUCTION_EXIT_SEQUENCE_BARS),
    ids=("entry_m5_96", "exit_m1_480"),
)
def test_micro_benchmark_per_bar_update_under_loose_bound(seq_len: int):
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
    ``seq_len`` is the exact native-lane receptive field.  Both production
    local clocks are measured here: Entry M5 uses ``MODEL_NATIVE_SEQ_LEN``
    (96), while Exit M1 uses ``PRODUCTION_EXIT_SEQUENCE_BARS`` (480).  The
    per-TF lanes retain their separately declared pyramid lengths.

    Measured on this fixture, 2026-08-13 `[M-synthetic]`, before -> after the
    package-8A retest-hold POLARITY FLIP (ms/bar, final ACTIVE-line count):
        seq_len  16: 0.040 (0 lines)    -> 0.042 (1 line)
        seq_len  96: 0.109 (41 lines)   -> 0.169 (65 lines)
        seq_len 512: 0.954 (1110 lines) -> 2.546 (2584 lines)
    The historical 512 result is close to the production Exit window, so the
    old 96-only guard did not measure where the Exit decision is made.  The
    same existing per-bar bound is now enforced at 480; no new threshold is
    introduced.
    """
    df = _random_walk_frame(5000, seed=13)
    band = 0.3
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
        broken, seq_len=200, band_atr=0.3, identity_expiry_bars=200, horizon_bars=10
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
        intact, seq_len=200, band_atr=0.3, identity_expiry_bars=200, horizon_bars=10
    )
    assert held_labels["y_line_support_touch_mask"].to_numpy()[53] == 1.0
    assert held_labels["y_line_support_touch_held"].to_numpy()[53] == 1.0

    # An unobserved forward window is undecidable: masked out, never a
    # placeholder outcome (rule 2e).
    undecidable = compute_trendline_touch_hold_labels_v1(
        intact, seq_len=200, band_atr=0.3, identity_expiry_bars=200, horizon_bars=30
    )
    assert undecidable["y_line_support_touch_mask"].to_numpy().sum() == 0.0


def test_touch_hold_label_observes_break_after_polarity_flip():
    from gx1.features.trendline_registry_v1 import (
        compute_trendline_touch_hold_labels_v1,
    )

    # One rising support line completes two role flips with the same line_id:
    # support --break(60)/hold(62)--> resistance
    #         --break(63)/hold(65)--> support --break(67)--> BROKEN.
    # The support touch at the second flip must therefore be judged against
    # break(67), not the already-past first break(60) for that line identity.
    frame = _support_line_frame(75, {10: 0.0, 30: 0.0, 50: 0.0})
    t = np.arange(len(frame), dtype=np.float64)
    proj = 100.0 + 0.05 * (t - 10.0)
    for bar in (60, 61):
        frame.loc[bar, ["close", "high", "low"]] = (
            proj[bar] - 0.8,
            proj[bar] - 0.6,
            proj[bar] - 1.0,
        )
    frame.loc[62, ["close", "high", "low"]] = (
        proj[62] - 0.5,
        proj[62] - 0.2,
        proj[62] - 0.7,
    )
    frame.loc[65, ["close", "high", "low"]] = (
        proj[65] + 0.5,
        proj[65] + 0.7,
        proj[65] + 0.2,
    )
    frame.loc[67, ["close", "high", "low"]] = (
        proj[67] - 0.8,
        proj[67] - 0.6,
        proj[67] - 1.0,
    )

    features, _ = _compute(frame, seq_len=200, band=0.3)
    assert features.loc[60, "geomline_break_down"] == 1.0
    assert features.loc[62, "geomline_retest_hold_down"] == 1.0
    assert features.loc[63, "geomline_break_up"] == 1.0
    assert features.loc[65, "geomline_retest_hold_up"] == 1.0
    assert features.loc[67, "geomline_break_down"] == 1.0

    labels = compute_trendline_touch_hold_labels_v1(
        frame,
        seq_len=200,
        band_atr=0.3,
        identity_expiry_bars=200,
        horizon_bars=5,
    )
    assert labels.loc[65, "y_line_support_touch_mask"] == 1.0
    assert labels.loc[65, "y_line_support_touch_held"] == 0.0
