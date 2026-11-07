"""Synthetic confirmed-pivot divergence checks, no cache or physical data."""
import numpy as np
import pandas as pd
import pytest

from gx1.audit.market_state_remaining_mtf_divergence10_oracle_v1 import (
    CACHE_COLUMNS, DIVERGENCE10_CACHE_NAMES, SWING_LOOKBACK,
    _pivot_masks, divergence10_from_closed_hlc_rsi_atr,
)
from gx1.features.htf_features import (
    MULTI_TF_PER_BAR_FEATURES_V4, _compute_v29_momentum_event_frame,
)
from gx1.features.smc_v1 import (
    SWING_LOOKBACK as OWNER_LOOKBACK, _detect_swing_pivots,
)


def _regular_bars(*, second_low_at_40=True):
    n = 80
    i = np.arange(n, dtype=np.float64)
    high, low = 120 + .01 * i, 100 + .01 * i
    high[[20, 40, 50]] = [150, 160, 170]
    low[30] = 80
    if second_low_at_40:
        low[40] = 70
    low[60] = 60 if second_low_at_40 else 70
    close = (high + low) / 2
    rsi = np.full(n, 50., dtype=np.float64)
    rsi[:14] = np.nan
    rsi[[20, 30, 40, 50, 60]] = [70, 50, 60, 50, 70 if second_low_at_40 else 60]
    atr = np.full(n, 2., dtype=np.float64)
    atr[:13] = np.nan
    return high, low, close, rsi, atr


def _hidden_bars():
    n = 60
    i = np.arange(n, dtype=np.float64)
    high, low = 120 + .01 * i, 100 + .01 * i
    high[[20, 40]] = [160, 150]
    low[[30, 40]] = [70, 80]
    close = (high + low) / 2
    rsi = np.full(n, 50., dtype=np.float64)
    rsi[:14] = np.nan
    rsi[[20, 30, 40]] = [60, 80, 70]
    atr = np.full(n, 2., dtype=np.float64)
    atr[:13] = np.nan
    return high, low, close, rsi, atr


def _owner_values(inputs):
    high, low, close, rsi, atr = inputs
    index = pd.date_range('2024-01-01', periods=len(close), freq='5min', tz='UTC')
    series = lambda values: pd.Series(values, index=index, dtype=np.float64)
    frame = _compute_v29_momentum_event_frame(
        high=series(high), low=series(low), close=series(close),
        rsi=series(rsi), mom_20_atr=series(np.zeros(len(close))),
        atr_positive=series(atr),
    )
    return frame.loc[:, DIVERGENCE10_CACHE_NAMES].to_numpy(dtype=np.float32)


def test_cache_order_and_named_confirmation_lag():
    assert CACHE_COLUMNS == tuple(range(113, 123))
    assert DIVERGENCE10_CACHE_NAMES == tuple(MULTI_TF_PER_BAR_FEATURES_V4[113:123])
    assert SWING_LOOKBACK == OWNER_LOOKBACK == 3



def test_centered_tie_tolerance_and_unconfirmed_edges():
    high = np.linspace(90., 80., 14)
    low = np.linspace(70., 80., 14)
    high[[3, 4, 7, 8]] = [100., 100. + 5e-13, 100., 100. + 2e-12]
    ours_high, ours_low = _pivot_masks(high, low)
    owner_high, owner_low = _detect_swing_pivots(high, low, OWNER_LOOKBACK)
    assert np.array_equal(ours_high, owner_high)
    assert np.array_equal(ours_low, owner_low)
    assert ours_high[3] and ours_high[4] and not ours_high[7] and ours_high[8]
    assert not ours_high[:3].any() and not ours_high[-3:].any()


def test_simultaneous_regular_events_strength_age_and_held_side():
    out = divergence10_from_closed_hlc_rsi_atr(*_regular_bars())
    assert out.shape == (80, 10) and out.dtype == np.float32
    assert np.isnan(out[:43, :]).all()  # pivot 40 is invisible until row 43
    assert out[43].tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert out[44].tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    assert out[52, 4:6].tolist() == [9, 0]
    assert out[53].tolist() == [1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    assert out[54, 4:6].tolist() == [1, 1]
    assert out[63].tolist() == [0, 1, 0, 1, 0, -1, 0, 0, 0, 0]
    assert out[64, 4:6].tolist() == [1, -1]


def test_hidden_events_have_separate_strength_and_do_not_reset_regular_age():
    out = divergence10_from_closed_hlc_rsi_atr(*_hidden_bars())
    assert np.isnan(out[:43]).all()
    assert out[43, [0, 1, 2, 3, 6, 7, 8, 9]].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert np.isnan(out[43:, 4:6]).all()  # no regular event has happened
    assert out[44, [6, 7, 8, 9]].tolist() == [0, 0, 0, 0]


def test_one_sided_definition_does_not_fabricate_merged_age():
    out = divergence10_from_closed_hlc_rsi_atr(*_regular_bars(second_low_at_40=False))
    assert out[43, 0] == out[43, 2] == 1
    assert np.isnan(out[43, [1, 3, 4, 5, 7, 9]]).all()
    assert out[63, 1] == out[63, 3] == 1
    assert out[63, 4:6].tolist() == [0, -1]  # earlier bear was not a merged event


def test_causal_prefix_and_selected_observed_positions():
    inputs = _regular_bars()
    full = divergence10_from_closed_hlc_rsi_atr(*inputs)
    positions = np.array([0, 42, 43, 44, 53, 63, 79], dtype=np.int64)
    selected = divergence10_from_closed_hlc_rsi_atr(*inputs, positions=positions)
    assert np.array_equal(selected.view(np.uint32), full[positions].view(np.uint32))
    for cutoff in (6, 7, 43, 44, 45, 54, 64, 80):
        prefix = divergence10_from_closed_hlc_rsi_atr(
            *(source[:cutoff] for source in inputs))
        assert np.array_equal(prefix.view(np.uint32), full[:cutoff].view(np.uint32))


def test_invalid_source_and_selection_fail_closed():
    high, low, close, rsi, atr = _regular_bars()
    with pytest.raises(ValueError, match='FLOAT64_INPUTS'):
        divergence10_from_closed_hlc_rsi_atr(high.astype(np.float32), low, close, rsi, atr)
    with pytest.raises(ValueError, match='HLC_GEOMETRY'):
        divergence10_from_closed_hlc_rsi_atr(high, close + 1, close, rsi, atr)
    bad_rsi = rsi.copy(); bad_rsi[30] = np.nan
    with pytest.raises(ValueError, match='RSI_DOMAIN'):
        divergence10_from_closed_hlc_rsi_atr(high, low, close, bad_rsi, atr)
    bad_atr = atr.copy(); bad_atr[30] = 0
    with pytest.raises(ValueError, match='ATR_DOMAIN'):
        divergence10_from_closed_hlc_rsi_atr(high, low, close, rsi, bad_atr)
    with pytest.raises(ValueError, match='POSITIONS'):
        divergence10_from_closed_hlc_rsi_atr(
            high, low, close, rsi, atr, positions=np.array([43, 42], dtype=np.int64))


def test_synthetic_owner_parity_for_regular_and_hidden_cases():
    for inputs in (_regular_bars(), _hidden_bars(),
                   _regular_bars(second_low_at_40=False)):
        independent = divergence10_from_closed_hlc_rsi_atr(*inputs)
        owner = _owner_values(inputs)
        assert np.array_equal(independent.view(np.uint32), owner.view(np.uint32))
