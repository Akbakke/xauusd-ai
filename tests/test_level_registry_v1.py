"""Contract tests for the V29 Phase A level registry owner.

Covers: causality (future-row invariance), touch dedup, break-once semantics,
reaction accounting (signed), tolerance-fit determinism + provenance,
chunk-carry bit-exactness, name-tuple drift guards, absence/expiry encoding,
retest hold/fail/none lifecycle, pivot one-truth vs smc_v1, and the
fail-closed input/state/kind validation standard.

All series here are synthetic: per rule 2c they prove code properties
(algebraic identities, causal invariances, declared contract shapes), never
production behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.features.level_registry_v1 import (
    LEVEL_KIND_PIVOT_CLUSTER,
    LEVEL_KIND_ROUND_NUMBER,
    LEVEL_KIND_SESSION_ANCHORED,
    LEVEL_KINDS,
    LEVEL_KINDS_IMPLEMENTED_V1,
    LEVEL_REGISTRY_AGE_CAP_BARS,
    LEVEL_REGISTRY_COUNT_AGE_CAP,
    LEVEL_REGISTRY_DIST_SATURATION_ATR,
    LEVEL_REGISTRY_M5_FEATURE_NAMES,
    LEVEL_REGISTRY_MTF_FEATURE_NAMES,
    LEVEL_REGISTRY_REACTION_WINDOW_BARS,
    LEVEL_REGISTRY_RETEST_WINDOW_BARS,
    LEVEL_REGISTRY_STATE_KEYS,
    compute_level_registry_m5_block_v1,
    compute_level_registry_mtf_block_v1,
    fit_level_registry_tolerance,
    require_level_kind_implemented,
)
from gx1.features.smc_v1 import SWING_LOOKBACK, _detect_swing_pivots


TOL = 0.5  # explicit test input for the engine (production uses the TRAIN fit)
SHIFT = 4000.0  # XAU-magnitude translation; pivot/zone geometry is invariant


def _mk_df(high, low, close, atr=None) -> pd.DataFrame:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    if atr is None:
        atr = np.ones_like(high)
    return pd.DataFrame({"high": high, "low": low, "close": close, "atr": np.asarray(atr, dtype=np.float64)})


def _col(matrix: np.ndarray, names: list[str], name: str) -> np.ndarray:
    return matrix[:, names.index(name)]


def _series_s1() -> pd.DataFrame:
    """One resistance level (pivot j=5, center 12+SHIFT), touches/merges only."""
    n = 28
    high = SHIFT + 10.0 + 0.01 * np.arange(n)
    high[5] = SHIFT + 12.0
    high[12:16] = SHIFT + np.array([11.6, 11.65, 11.7, 11.75])
    high[16:20] = SHIFT + np.array([11.4, 11.35, 11.3, 11.25])
    high[20] = SHIFT + 11.55
    high[21:] = SHIFT + np.array([11.0, 10.9, 10.8, 10.7, 10.6, 10.5, 10.4])
    low = SHIFT + 9.0 - 0.01 * np.arange(n)
    close = SHIFT + 9.5 - 0.005 * np.arange(n)
    return _mk_df(high, low, close)


def _series_s2() -> pd.DataFrame:
    """Support level (pivot j=5, center 7+SHIFT): touch, break down, retest hold,
    one positive and one negative completed reaction."""
    n = 30
    idx = np.arange(n)
    high = SHIFT + 14.0 + 0.01 * idx
    low = SHIFT + 9.0 - 0.01 * idx
    low[5] = SHIFT + 7.0
    close = low + 0.3
    high[12], low[12], close[12] = SHIFT + 7.6, SHIFT + 7.4, SHIFT + 7.5
    tail = idx[13:]
    high[13:] = SHIFT + 6.9 - 0.1 * (tail - 13)
    low[13:] = SHIFT + 6.5 - 0.1 * (tail - 13)
    close[13:] = low[13:] + 0.05
    close[13] = SHIFT + 6.6
    close[14] = SHIFT + 6.45  # first close below zone edge 6.5 -> break at t=14
    return _mk_df(high, low, close)


def _series_s3() -> pd.DataFrame:
    """S2 variant: after the down-break at t=14 the retest at t=15 FAILS
    (close back above center)."""
    df = _series_s2()
    high = df["high"].to_numpy().copy()
    low = df["low"].to_numpy().copy()
    close = df["close"].to_numpy().copy()
    high[15], low[15], close[15] = SHIFT + 7.3, SHIFT + 6.5, SHIFT + 7.2
    for i in range(16, len(high)):
        high[i] = SHIFT + 7.25 - 0.05 * (i - 16)
        low[i] = SHIFT + 6.6 - 0.05 * (i - 16)
        close[i] = 0.5 * (high[i] + low[i])
    return _mk_df(high, low, close)


def _series_s4_no_reentry() -> pd.DataFrame:
    """Support break at t=13 with NO zone re-entry: retest window expires to
    the none-state (no event)."""
    n = 42
    idx = np.arange(n)
    high = SHIFT + 14.0 + 0.01 * idx
    low = SHIFT + 9.0 - 0.01 * idx
    low[5] = SHIFT + 7.0
    close = low + 0.3
    high[12], low[12], close[12] = SHIFT + 7.6, SHIFT + 7.4, SHIFT + 7.5
    tail = idx[13:]
    high[13:] = SHIFT + 6.4 - 0.1 * (tail - 13)
    low[13:] = SHIFT + 6.0 - 0.1 * (tail - 13)
    close[13:] = 0.5 * (high[13:] + low[13:])
    return _mk_df(high, low, close)


def _series_expiry() -> pd.DataFrame:
    """One resistance level admitted at t=8 and never touched again: expiry at
    t - last_touch_bar(=5) > AGE_CAP['m5'] = 240, i.e. removal at t=246."""
    n = 260
    idx = np.arange(n)
    high = SHIFT + 10.0 + 0.001 * idx
    high[5] = SHIFT + 12.0
    low = SHIFT + 9.0 - 0.001 * idx
    close = SHIFT + 9.3 - 0.0005 * idx
    return _mk_df(high, low, close)


def _rng_df(n: int = 200, seed: int = 20260811) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mid = SHIFT + np.cumsum(rng.normal(0.0, 1.0, n))
    hi_off = np.abs(rng.normal(0.0, 1.0, n)) + 0.1
    lo_off = np.abs(rng.normal(0.0, 1.0, n)) + 0.1
    high = mid + hi_off
    low = mid - lo_off
    close = low + rng.uniform(0.0, 1.0, n) * (high - low)
    atr = 1.0 + np.abs(rng.normal(0.0, 0.2, n))
    return _mk_df(high, low, close, atr)


# ---------------------------------------------------------------------------
# Declared contracts / drift guards
# ---------------------------------------------------------------------------

def test_declared_name_tuples_match_design_doc_verbatim():
    assert LEVEL_REGISTRY_M5_FEATURE_NAMES == (
        "level_above_dist_atr",
        "level_above_touch_count",
        "level_above_age_bars",
        "level_above_bars_since_touch",
        "level_above_mean_reaction_atr",
        "level_above_max_reaction_atr",
        "level_above_last_reaction_atr",
        "level_below_dist_atr",
        "level_below_touch_count",
        "level_below_age_bars",
        "level_below_bars_since_touch",
        "level_below_mean_reaction_atr",
        "level_below_max_reaction_atr",
        "level_below_last_reaction_atr",
        "level_break_up_event",
        "level_break_down_event",
        "level_broken_touch_count",
        "level_bars_since_break",
        "level_retest_hold_signed",
        "level_retest_fail_signed",
        "level_round_50_dist_atr",
        "level_round_100_dist_atr",
    )
    assert LEVEL_REGISTRY_MTF_FEATURE_NAMES == (
        "mtf_level_above_dist_atr",
        "mtf_level_below_dist_atr",
        "mtf_level_above_touch_count",
        "mtf_level_below_touch_count",
        "mtf_level_above_mean_reaction_atr",
        "mtf_level_below_mean_reaction_atr",
        "mtf_level_break_up_event",
        "mtf_level_break_down_event",
        "mtf_level_bars_since_break",
        "mtf_level_retest_hold_signed",
        "mtf_level_retest_fail_signed",
    )
    assert len(LEVEL_REGISTRY_M5_FEATURE_NAMES) == 22
    assert len(LEVEL_REGISTRY_MTF_FEATURE_NAMES) == 11
    assert all(n.startswith("level_") for n in LEVEL_REGISTRY_M5_FEATURE_NAMES)
    assert all(n.startswith("mtf_level_") for n in LEVEL_REGISTRY_MTF_FEATURE_NAMES)


def test_emitted_names_match_declared_tuples():
    df = _series_s1()
    m5, names_m5 = compute_level_registry_m5_block_v1(df, tol_level_atr=TOL)
    assert tuple(names_m5) == LEVEL_REGISTRY_M5_FEATURE_NAMES
    assert m5.shape == (len(df), 22)
    assert m5.dtype == np.float32
    mtf, names_mtf = compute_level_registry_mtf_block_v1(df, tf="m5", tol_level_atr=TOL)
    assert tuple(names_mtf) == LEVEL_REGISTRY_MTF_FEATURE_NAMES
    assert mtf.shape == (len(df), 11)


def test_convention_constants_pinned_to_origins():
    # Origins: _liquidity_zones lookbacks; tau-12/tau-24 conventions; the
    # exp(-min(x,20)) cap; the 999 sentinel (module docstring cites each).
    assert LEVEL_REGISTRY_AGE_CAP_BARS == {"m5": 240, "m15": 192, "h1": 168, "h4": 168, "d1": 60}
    assert LEVEL_REGISTRY_REACTION_WINDOW_BARS == 12
    assert LEVEL_REGISTRY_RETEST_WINDOW_BARS == 24
    assert LEVEL_REGISTRY_DIST_SATURATION_ATR == 20.0
    assert LEVEL_REGISTRY_COUNT_AGE_CAP == 999.0


def test_kind_enum_phase_a_boundary():
    assert LEVEL_KINDS == ("pivot_cluster", "session_anchored", "round_number")
    assert LEVEL_KINDS_IMPLEMENTED_V1 == ("pivot_cluster", "round_number")
    assert require_level_kind_implemented(LEVEL_KIND_PIVOT_CLUSTER) == "pivot_cluster"
    assert require_level_kind_implemented(LEVEL_KIND_ROUND_NUMBER) == "round_number"
    with pytest.raises(RuntimeError, match="KIND_NOT_IMPLEMENTED"):
        require_level_kind_implemented(LEVEL_KIND_SESSION_ANCHORED)
    with pytest.raises(RuntimeError, match="KIND_UNKNOWN"):
        require_level_kind_implemented("bogus")


def test_mtf_block_values_equal_m5_registry_fields_on_same_clock():
    df = _rng_df()
    m5, names_m5 = compute_level_registry_m5_block_v1(df, tol_level_atr=0.6)
    mtf, names_mtf = compute_level_registry_mtf_block_v1(df, tf="m5", tol_level_atr=0.6)
    pairs = {
        "mtf_level_above_dist_atr": "level_above_dist_atr",
        "mtf_level_below_dist_atr": "level_below_dist_atr",
        "mtf_level_above_touch_count": "level_above_touch_count",
        "mtf_level_below_touch_count": "level_below_touch_count",
        "mtf_level_above_mean_reaction_atr": "level_above_mean_reaction_atr",
        "mtf_level_below_mean_reaction_atr": "level_below_mean_reaction_atr",
        "mtf_level_break_up_event": "level_break_up_event",
        "mtf_level_break_down_event": "level_break_down_event",
        "mtf_level_bars_since_break": "level_bars_since_break",
        "mtf_level_retest_hold_signed": "level_retest_hold_signed",
        "mtf_level_retest_fail_signed": "level_retest_fail_signed",
    }
    for mtf_name, m5_name in pairs.items():
        np.testing.assert_array_equal(
            _col(mtf, names_mtf, mtf_name), _col(m5, names_m5, m5_name)
        )


# ---------------------------------------------------------------------------
# S1: warmup, creation, touch dedup, merges, positive reaction accounting
# ---------------------------------------------------------------------------

def test_s1_warmup_prefix_and_creation():
    df = _series_s1()
    m5, names = compute_level_registry_m5_block_v1(df, tol_level_atr=TOL)
    assert np.isnan(m5[:8]).all()          # NaN until first admitted level (t=8)
    assert np.isfinite(m5[8:]).all()       # no mid-series NaN afterwards
    dist = _col(m5, names, "level_above_dist_atr")
    touch = _col(m5, names, "level_above_touch_count")
    age = _col(m5, names, "level_above_age_bars")
    bst = _col(m5, names, "level_above_bars_since_touch")
    assert dist[8] == pytest.approx(12.0 - (9.5 - 0.04), abs=1e-3)  # 2.54
    assert touch[8] == 1.0
    assert age[8] == 0.0
    assert bst[8] == 3.0  # creation sets last_touch_bar = pivot bar j = 5


def test_s1_touch_dedup_and_merge_accounting():
    df = _series_s1()
    m5, names, state = compute_level_registry_m5_block_v1(
        df, tol_level_atr=TOL, return_registry_state=True
    )
    touch = _col(m5, names, "level_above_touch_count")
    bst = _col(m5, names, "level_above_bars_since_touch")
    # zone entry at 12 (one touch), hover 13-15 deduplicated, merge of pivot
    # j=15 at t=18, re-entry touch at 20, merge of pivot j=20 at t=23.
    assert touch[11] == 1.0
    assert touch[12] == 2.0
    assert (touch[13:16] == 2.0).all()
    assert touch[16] == 2.0
    assert touch[18] == 3.0
    assert touch[20] == 4.0
    assert touch[23] == 5.0
    assert bst[14] == 2.0   # last touch at 12
    assert bst[18] == 3.0   # merge sets last_touch_bar = pivot bar j = 15
    # merge center evolution: 12.0 -> 11.875 -> 11.766666...
    lv = state["levels"][0]
    assert lv["center_price"] == pytest.approx(SHIFT + 11.7666666667, abs=1e-6)
    assert lv["member_pivot_count"] == 3
    assert lv["member_pivot_bars"] == [5, 15, 20]
    # no breaks anywhere in S1
    assert _col(m5, names, "level_break_up_event")[8:].sum() == 0.0
    assert _col(m5, names, "level_break_down_event")[8:].sum() == 0.0
    assert (_col(m5, names, "level_bars_since_break")[8:] == 999.0).all()


def test_s1_reaction_accounting_and_event_gated_zero():
    df = _series_s1()
    m5, names = compute_level_registry_m5_block_v1(df, tol_level_atr=TOL)
    mean_r = _col(m5, names, "level_above_mean_reaction_atr")
    max_r = _col(m5, names, "level_above_max_reaction_atr")
    last_r = _col(m5, names, "level_above_last_reaction_atr")
    # event-gated 0 before the first completed window
    assert (mean_r[8:20] == 0.0).all()
    # creation window t0=8 completes at 20: (12.0 - min(low[9..20]))/1 = 3.2
    assert mean_r[20] == pytest.approx(3.2, abs=1e-3)
    assert max_r[20] == pytest.approx(3.2, abs=1e-3)
    assert last_r[20] == pytest.approx(3.2, abs=1e-3)
    # touch window t0=12 completes at 24: 12.0 - 8.76 = 3.24
    assert last_r[24] == pytest.approx(3.24, abs=1e-3)
    assert max_r[24] == pytest.approx(3.24, abs=1e-3)
    assert mean_r[24] == pytest.approx(3.22, abs=1e-3)


def test_s1_absent_side_and_round_grid_encoding():
    df = _series_s1()
    m5, names = compute_level_registry_m5_block_v1(df, tol_level_atr=TOL)
    # no low-side level ever exists: absent-side encoding
    assert (_col(m5, names, "level_below_dist_atr")[8:] == 20.0).all()
    assert (_col(m5, names, "level_below_touch_count")[8:] == 0.0).all()
    assert (_col(m5, names, "level_below_age_bars")[8:] == 999.0).all()
    assert (_col(m5, names, "level_below_bars_since_touch")[8:] == 999.0).all()
    assert (_col(m5, names, "level_below_mean_reaction_atr")[8:] == 0.0).all()
    # round grid: close[20] = 4009.40, nearest 50/100 gridline = 4000
    assert _col(m5, names, "level_round_50_dist_atr")[20] == pytest.approx(9.4, abs=1e-3)
    assert _col(m5, names, "level_round_100_dist_atr")[20] == pytest.approx(9.4, abs=1e-3)


# ---------------------------------------------------------------------------
# S2/S3/S4: break-once, retest hold/fail/none, signed (negative) reactions
# ---------------------------------------------------------------------------

def test_s2_break_once_retest_hold_and_signed_reactions():
    df = _series_s2()
    m5, names, state = compute_level_registry_m5_block_v1(
        df, tol_level_atr=TOL, return_registry_state=True
    )
    down = _col(m5, names, "level_break_down_event")
    up = _col(m5, names, "level_break_up_event")
    btc = _col(m5, names, "level_broken_touch_count")
    bsb = _col(m5, names, "level_bars_since_break")
    hold = _col(m5, names, "level_retest_hold_signed")
    fail = _col(m5, names, "level_retest_fail_signed")
    below_dist = _col(m5, names, "level_below_dist_atr")
    # break-once: exactly one down-break, at t=14, and no up-breaks
    assert down[14] == 1.0
    assert down[8:].sum() == 1.0
    assert up[8:].sum() == 0.0
    assert btc[14] == 2.0 and btc[8:].sum() == 2.0  # event-gated 0 off-event
    assert (bsb[8:14] == 999.0).all()
    assert bsb[14] == 0.0 and bsb[15] == 1.0
    # broken level leaves the below slot at the break bar
    assert below_dist[9] == pytest.approx((9.0 - 0.09 + 0.3) - 7.0, abs=1e-3)  # 2.21
    assert below_dist[14] == 20.0
    # retest: break bar excluded; first re-entry at t=15 holds (close < center)
    assert hold[15] == -1.0 and hold[8:].sum() == -1.0
    assert fail[8:].sum() == 0.0
    lv = next(l for l in state["levels"] if l["side_of_origin"] == "low_pivot")
    assert lv["status"] == "broken"
    assert lv["break_bar"] == 14 and lv["break_side"] == -1
    assert lv["retest_state"] == "held"
    # signed reactions on the support level: creation window t0=8 completes at
    # 20 with +7.11 (price lifted off); touch window t0=12 completes at 24
    # with -0.1 (touch never lifted off -> negative, sign-blindness repaired)
    assert lv["completed_reaction_count"] == 2
    assert lv["reaction_max_atr"] == pytest.approx(7.11, abs=1e-3)
    assert lv["reaction_last_atr"] == pytest.approx(-0.1, abs=1e-6)
    assert lv["reaction_sum_atr"] == pytest.approx(7.01, abs=1e-3)


def test_s3_retest_fail_signed_carries_break_direction():
    df = _series_s3()
    m5, names, state = compute_level_registry_m5_block_v1(
        df, tol_level_atr=TOL, return_registry_state=True
    )
    hold = _col(m5, names, "level_retest_hold_signed")
    fail = _col(m5, names, "level_retest_fail_signed")
    assert fail[15] == -1.0          # sign = break direction (down)
    assert fail[8:].sum() == -1.0
    assert hold[8:].sum() == 0.0
    lv = next(l for l in state["levels"] if l["break_bar"] == 14)
    assert lv["retest_state"] == "failed"


def test_s4_retest_window_expiry_emits_no_event():
    df = _series_s4_no_reentry()
    m5, names, state = compute_level_registry_m5_block_v1(
        df, tol_level_atr=TOL, return_registry_state=True
    )
    assert _col(m5, names, "level_break_down_event")[8:].sum() == 1.0
    assert _col(m5, names, "level_retest_hold_signed")[8:].sum() == 0.0
    assert _col(m5, names, "level_retest_fail_signed")[8:].sum() == 0.0
    lv = next(l for l in state["levels"] if l["break_side"] == -1)
    assert lv["retest_state"] == "none"  # window expired without re-entry


def test_expiry_removes_stale_level_with_finite_absence_encoding():
    df = _series_expiry()
    m5, names = compute_level_registry_m5_block_v1(df, tol_level_atr=TOL)
    dist = _col(m5, names, "level_above_dist_atr")
    bst = _col(m5, names, "level_above_bars_since_touch")
    age = _col(m5, names, "level_above_age_bars")
    assert dist[245] == pytest.approx(12.0 - (9.3 - 0.0005 * 245), abs=1e-3)
    assert bst[245] == 240.0
    assert age[245] == 237.0
    # strict expiry: removed at t=246 (t - last_touch_bar > 240); rows stay
    # finite with the absence encoding (no mid-series NaN)
    assert dist[246] == 20.0 and age[246] == 999.0 and bst[246] == 999.0
    assert np.isfinite(m5[8:]).all()


# ---------------------------------------------------------------------------
# Causality, chunk-carry, one pivot truth
# ---------------------------------------------------------------------------

def test_causality_future_row_invariance():
    df = _rng_df(n=200)
    full, _ = compute_level_registry_m5_block_v1(df, tol_level_atr=0.6)
    for k in (23, 61, 137, 190):
        prefix, _ = compute_level_registry_m5_block_v1(df.iloc[:k], tol_level_atr=0.6)
        np.testing.assert_array_equal(full[:k], prefix)


def test_chunk_carry_bit_exactness_m5():
    df = _rng_df(n=200)
    full, _, full_state = compute_level_registry_m5_block_v1(
        df, tol_level_atr=0.6, return_registry_state=True
    )
    parts = []
    state = None
    for lo, hi in ((0, 7), (7, 97), (97, 141), (141, 200)):
        block, _, state = compute_level_registry_m5_block_v1(
            df.iloc[lo:hi],
            tol_level_atr=0.6,
            registry_state=state,
            return_registry_state=True,
        )
        parts.append(block)
    chunked = np.vstack(parts)
    np.testing.assert_array_equal(full, chunked)
    assert state == full_state


def test_chunk_carry_bit_exactness_mtf_and_atr_prefix():
    df = _rng_df(n=180, seed=7)
    atr = df["atr"].to_numpy().copy()
    atr[:37] = np.nan  # declared contiguous ATR warmup prefix
    df = df.assign(atr=atr)
    full, _, full_state = compute_level_registry_mtf_block_v1(
        df, tf="h1", tol_level_atr=0.7, return_registry_state=True
    )
    # warmup prefix is all-NaN; finite rows form a single suffix
    finite = np.isfinite(full).all(axis=1)
    assert finite.any()
    first = int(np.argmax(finite))
    assert first >= 37
    assert finite[first:].all() and not finite[:first].any()
    parts = []
    state = None
    for lo, hi in ((0, 20), (20, 44), (44, 180)):  # split inside the NaN prefix too
        block, _, state = compute_level_registry_mtf_block_v1(
            df.iloc[lo:hi],
            tf="h1",
            tol_level_atr=0.7,
            registry_state=state,
            return_registry_state=True,
        )
        parts.append(block)
    np.testing.assert_array_equal(full, np.vstack(parts))
    assert state == full_state


def test_one_pivot_truth_every_smc_v1_pivot_is_a_member():
    df = _rng_df(n=200)  # < AGE_CAP['m5'] so no level can expire
    _, _, state = compute_level_registry_m5_block_v1(
        df, tol_level_atr=0.6, return_registry_state=True
    )
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    sh, sl = _detect_swing_pivots(high, low, SWING_LOOKBACK)
    expected = sorted(list(np.flatnonzero(sh)) + list(np.flatnonzero(sl)))
    members = sorted(
        bar for lv in state["levels"] for bar in lv["member_pivot_bars"]
    )
    assert members == expected


def test_emitted_value_domains_on_generic_series():
    df = _rng_df(n=200)
    m5, names = compute_level_registry_m5_block_v1(df, tol_level_atr=0.6)
    finite = np.isfinite(m5).all(axis=1)
    rows = m5[finite]
    for name in ("level_break_up_event", "level_break_down_event"):
        assert set(np.unique(_col(rows, names, name))) <= {0.0, 1.0}
    for name in ("level_retest_hold_signed", "level_retest_fail_signed"):
        assert set(np.unique(_col(rows, names, name))) <= {-1.0, 0.0, 1.0}
    for name in ("level_above_dist_atr", "level_below_dist_atr"):
        col = _col(rows, names, name)
        assert (col >= 0.0).all() and (col <= 20.0).all()
    for name in (
        "level_above_touch_count",
        "level_below_touch_count",
        "level_above_age_bars",
        "level_below_age_bars",
        "level_above_bars_since_touch",
        "level_below_bars_since_touch",
        "level_broken_touch_count",
        "level_bars_since_break",
    ):
        col = _col(rows, names, name)
        assert (col >= 0.0).all() and (col <= 999.0).all()


# ---------------------------------------------------------------------------
# Tolerance fit
# ---------------------------------------------------------------------------

def test_fit_determinism_and_provenance():
    df = _rng_df(n=300, seed=11)
    tol1, prov1 = fit_level_registry_tolerance(
        df, q=0.35, tf="m5", declared_train_window="TEST_WINDOW_SYNTHETIC"
    )
    tol2, prov2 = fit_level_registry_tolerance(
        df, q=0.35, tf="m5", declared_train_window="TEST_WINDOW_SYNTHETIC"
    )
    assert tol1 == tol2
    assert prov1 == prov2
    assert tol1 > 0.0
    # rule 2f: sample size and sampling bound stated
    for key in (
        "sample_size",
        "quantile_prob_se",
        "tol_bracket_lo",
        "tol_bracket_hi",
        "quantile_q",
        "quantile_method",
        "tf",
        "declared_train_window",
        "swing_lookback",
        "n_pivots_admitted",
        "n_pivots_dropped_atr_unavailable",
        "tol_level_atr",
    ):
        assert key in prov1
    assert prov1["tol_level_atr"] == tol1
    assert prov1["swing_lookback"] == SWING_LOOKBACK
    assert prov1["tol_bracket_lo"] <= tol1 <= prov1["tol_bracket_hi"]
    # independent sample-size check: every detected pivot except the first is
    # one nearest-earlier distance (all-finite ATR here)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    sh, sl = _detect_swing_pivots(high, low, SWING_LOOKBACK)
    n_pivots = int(sh.sum() + sl.sum())
    assert prov1["n_pivots_admitted"] == n_pivots
    assert prov1["sample_size"] == n_pivots - 1
    assert prov1["n_pivots_dropped_atr_unavailable"] == 0


def test_fit_quantile_monotone_in_q():
    df = _rng_df(n=300, seed=11)
    tol_lo, _ = fit_level_registry_tolerance(
        df, q=0.15, tf="m5", declared_train_window="W"
    )
    tol_hi, _ = fit_level_registry_tolerance(
        df, q=0.85, tf="m5", declared_train_window="W"
    )
    assert tol_lo <= tol_hi


def test_fit_fail_closed():
    df = _rng_df(n=300, seed=11)
    with pytest.raises(RuntimeError, match="Q_INVALID"):
        fit_level_registry_tolerance(df, q=0.0, tf="m5", declared_train_window="W")
    with pytest.raises(RuntimeError, match="Q_INVALID"):
        fit_level_registry_tolerance(df, q=1.0, tf="m5", declared_train_window="W")
    with pytest.raises(RuntimeError, match="WINDOW_UNDECLARED"):
        fit_level_registry_tolerance(df, q=0.5, tf="m5", declared_train_window="")
    with pytest.raises(RuntimeError, match="TF_INVALID"):
        fit_level_registry_tolerance(df, q=0.5, tf="m30", declared_train_window="W")
    with pytest.raises(TypeError):
        fit_level_registry_tolerance(df, tf="m5", declared_train_window="W")  # q required
    # zero pivots (monotone highs / lows) -> insufficient sample
    n = 40
    flatless = _mk_df(
        SHIFT + 10.0 + 0.01 * np.arange(n),
        SHIFT + 9.0 - 0.01 * np.arange(n),
        SHIFT + 9.5 + 0.001 * np.arange(n),
    )
    with pytest.raises(RuntimeError, match="TOL_FIT_INSUFFICIENT"):
        fit_level_registry_tolerance(flatless, q=0.5, tf="m5", declared_train_window="W")


# ---------------------------------------------------------------------------
# Fail-closed input/state validation
# ---------------------------------------------------------------------------

def test_input_validation_fail_closed():
    df = _series_s1()
    with pytest.raises(RuntimeError, match="SOURCE_EMPTY"):
        compute_level_registry_m5_block_v1(df.iloc[0:0], tol_level_atr=TOL)
    with pytest.raises(RuntimeError, match="SOURCE_MISSING"):
        compute_level_registry_m5_block_v1(df.drop(columns=["atr"]), tol_level_atr=TOL)
    bad = df.copy()
    bad.loc[bad.index[4], "high"] = np.nan
    with pytest.raises(RuntimeError, match="SOURCE_NONFINITE"):
        compute_level_registry_m5_block_v1(bad, tol_level_atr=TOL)
    bad = df.copy()
    bad.loc[bad.index[4], "low"] = bad.loc[bad.index[4], "high"] + 1.0
    with pytest.raises(RuntimeError, match="GEOMETRY_INVALID"):
        compute_level_registry_m5_block_v1(bad, tol_level_atr=TOL)
    bad = df.copy()
    bad.loc[bad.index[4], "atr"] = 0.0
    with pytest.raises(RuntimeError, match="GEOMETRY_INVALID"):
        compute_level_registry_m5_block_v1(bad, tol_level_atr=TOL)
    bad = df.copy()  # mid-series ATR gap is not a prefix
    bad.loc[bad.index[10], "atr"] = np.nan
    with pytest.raises(RuntimeError, match="ATR_AVAILABILITY_INVALID"):
        compute_level_registry_m5_block_v1(bad, tol_level_atr=TOL)
    for bad_tol in (0.0, -1.0, float("nan"), float("inf"), None):
        with pytest.raises(RuntimeError, match="TOL_INVALID"):
            compute_level_registry_m5_block_v1(df, tol_level_atr=bad_tol)
    with pytest.raises(RuntimeError, match="TF_INVALID"):
        compute_level_registry_mtf_block_v1(df, tf="m30", tol_level_atr=TOL)


def test_state_contract_fail_closed():
    df = _rng_df(n=120)
    _, _, state = compute_level_registry_m5_block_v1(
        df.iloc[:60], tol_level_atr=0.6, return_registry_state=True
    )
    assert set(state) == set(LEVEL_REGISTRY_STATE_KEYS)
    # tf mismatch (m5 state into an h1 lane)
    with pytest.raises(RuntimeError, match="STATE_CONTRACT_MISMATCH"):
        compute_level_registry_mtf_block_v1(
            df.iloc[60:], tf="h1", tol_level_atr=0.6, registry_state=state
        )
    # tolerance mismatch: carried state must bind the exact frozen constant
    with pytest.raises(RuntimeError, match="STATE_CONTRACT_MISMATCH"):
        compute_level_registry_m5_block_v1(
            df.iloc[60:], tol_level_atr=0.61, registry_state=state
        )
    # schema drift
    broken = dict(state)
    broken.pop("levels")
    with pytest.raises(RuntimeError, match="STATE_SCHEMA_INVALID"):
        compute_level_registry_m5_block_v1(
            df.iloc[60:], tol_level_atr=0.6, registry_state=broken
        )
    broken = dict(state)
    broken["state_version"] = "level_registry_v0_state"
    with pytest.raises(RuntimeError, match="STATE_VERSION_MISMATCH"):
        compute_level_registry_m5_block_v1(
            df.iloc[60:], tol_level_atr=0.6, registry_state=broken
        )
    # a carried started-state forbids new unavailable-ATR rows
    started_state = state
    assert started_state["atr_started"] is True
    tail_df = df.iloc[60:].copy()
    tail_df.loc[tail_df.index[0], "atr"] = np.nan
    with pytest.raises(RuntimeError, match="ATR_AVAILABILITY_INVALID"):
        compute_level_registry_m5_block_v1(
            tail_df, tol_level_atr=0.6, registry_state=started_state
        )
