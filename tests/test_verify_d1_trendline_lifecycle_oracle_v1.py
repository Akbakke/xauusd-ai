"""Small OHLC fixtures for the independent D1 lifecycle replay."""
import numpy as np
import pytest

from gx1.audit.verify_d1_trendline_lifecycle_oracle_v1 import (
    LIFECYCLE_NAMES,
    NAMES,
    confirmed_center,
    oracle_rows,
)


def test_confirmed_pivot_needs_three_future_bars_and_accepts_exact_ties():
    high = np.array([11., 11., 11., 12., 11., 11., 11.])
    low = np.array([10., 10., 10., 9., 10., 10., 10.])
    assert confirmed_center(high, low, 5) == (False, False)
    assert confirmed_center(high, low, 6) == (True, True)
    high[4] = 12.
    low[4] = 9.
    assert confirmed_center(high, low, 6) == (True, True)
    high[4] = 12.01
    low[4] = 8.99
    assert confirmed_center(high, low, 6) == (False, False)


def test_support_promotion_break_and_held_retest_flip():
    high = 12. + np.arange(17) * .1
    low = np.full(17, 11.)
    low[[3, 7, 11]] = 10.
    close = np.full(17, 11.)
    low[15] = close[15] = 9.
    low[16] = close[16] = 9.5
    rows, counts = oracle_rows(
        high, low, close, np.ones(17), seq_len=16,
        band_atr=.2, witness_positions=[14, 15, 16],
    )
    ix = {name: i for i, name in enumerate(NAMES)}
    assert rows[0, ix['geomline_below_active_count']] == 1.
    assert rows[0, ix['geomline_below_touch_count']] == 3.
    assert rows[1, ix['geomline_break_down']] == 1.
    assert rows[1, ix['geomline_break_line_touch_count']] == 3.
    assert rows[1, ix['geomline_break_line_age_bars']] == 9.
    assert rows[2, ix['geomline_retest_hold_down']] == 1.
    assert rows[2, ix['geomline_above_active_count']] == 1.
    assert rows[2, ix['geomline_above_touch_count']] == 4.
    assert rows[2, ix['geomline_bars_since_break_signed']] == -1.
    assert counts == {
        'confirmed_pivots': 3, 'promotions': 1, 'first_breaks': 1,
        'held_retests': 1, 'failed_retests': 0,
    }


def test_two_independent_sloping_lines_produce_converging_channel():
    high = np.full(15, 11.5)
    low = np.full(15, 10.5)
    for bar, lo, hi in ((3, 9., 13.), (7, 9.4, 12.6), (11, 9.8, 12.2)):
        low[bar], high[bar] = lo, hi
    rows, counts = oracle_rows(
        high, low, np.full(15, 11.), np.ones(15), seq_len=16,
        band_atr=.2, witness_positions=[14],
    )
    ix = {name: i for i, name in enumerate(NAMES)}
    row = rows[0]
    assert row[ix['geomline_above_active_count']] == 1.
    assert row[ix['geomline_below_active_count']] == 1.
    assert row[ix['geomchan_active']] == 1.
    assert row[ix['geomchan_converging']] == 1.
    assert row[ix['geomchan_width_atr']] == pytest.approx(1.8)
    assert row[ix['geomchan_pos_0_1']] == pytest.approx(.5)
    assert row[ix['geomchan_apex_proximity']] == pytest.approx(.4375)
    assert counts['promotions'] == 2


def test_warmup_and_invalid_witnesses_cannot_silently_become_evidence():
    high = 12. + np.arange(10) * .1
    low = np.full(10, 10.)
    close = np.full(10, 11.)
    atr = np.ones(10)
    rows, _ = oracle_rows(
        high, low, close, atr, seq_len=16,
        band_atr=.2, witness_positions=[7, 8],
    )
    assert np.isnan(rows[0]).all()
    assert len(NAMES) == 31 and len(LIFECYCLE_NAMES) == 19
    assert np.isnan(rows[1, NAMES.index('geomline_bars_since_break_signed')])
    with pytest.raises(ValueError, match='WITNESSES'):
        oracle_rows(high, low, close, atr, seq_len=16,
                    band_atr=.2, witness_positions=[8, 7])


def test_physical_input_domain_matches_producer():
    high = np.full(10, 12.)
    low = np.full(10, 10.)
    close = np.full(10, 11.)
    atr = np.ones(10)
    with pytest.raises(ValueError, match="PRICE"):
        oracle_rows(high - 20., low - 20., close - 20., atr,
                    seq_len=16, band_atr=.2, witness_positions=[9])
    atr_hole = atr.copy()
    atr_hole[:2] = np.nan
    atr_hole[4] = np.nan
    with pytest.raises(ValueError, match="ATR"):
        oracle_rows(high, low, close, atr_hole,
                    seq_len=16, band_atr=.2, witness_positions=[9])
    atr_prefix = atr.copy()
    atr_prefix[:2] = np.nan
    rows, _ = oracle_rows(high, low, close, atr_prefix,
                          seq_len=16, band_atr=.2, witness_positions=[9])
    assert rows.shape == (1, 31)
