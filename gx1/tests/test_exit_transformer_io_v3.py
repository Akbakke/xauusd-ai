#!/usr/bin/env python3
"""
Unit tests for Exit Transformer IO V3 (IOV3_CLEAN).

IOV3_CLEAN = IOV2 (31) + 4 extras = 35 dims. One representation per concept (no duplicates).
Philosophy: TRUTH / ONE UNIVERSE — no fallback; missing/NaN/wrong dim → RuntimeError.
"""

from __future__ import annotations

import numpy as np
import pytest

from gx1.contracts.exit_io import (
    EXIT_IO_FEATURE_COUNT,
    FEATURE_DIM_V3,
    IOV2_DIM,
    MIN_ATR_BPS,
    ORDERED_EXIT_EXTRA_FIELDS_V3,
    ORDERED_EXIT_FEATURES_V3,
    ordered_feature_names_v3,
    row_to_feature_vector_v3,
    validate_row_v3,
    validate_window_v3,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_row(
    *,
    atr_bps_now: float = 10.0,
    entry_price: float = 1.0,
    price_now: float = 1.0002,
) -> dict:
    """Return a fully valid synthetic row for IOV3 tests."""
    return {
        "signals": {
            "p_long_now": 0.6,
            "p_short_now": 0.4,
            "p_hat_now": 0.6,
            "uncertainty_score": 0.2,
            "margin_top1_top2": 0.15,
            "entropy": 0.5,
        },
        "entry_snapshot": {
            "p_long_entry": 0.55,
            "p_hat_entry": 0.55,
            "uncertainty_entry": 0.25,
            "entropy_entry": 0.48,
            "margin_entry": 0.12,
        },
        "state": {
            "pnl_bps": 2.0,
            "mfe_bps": 5.0,
            "mae_bps": -1.0,
            "dd_from_mfe_bps": 3.0,
            "bars_held": 10,
            "time_since_mfe_bars": 3,
            "atr_bps_now": atr_bps_now,
            "entry_price": entry_price,
            "price_now": price_now,
            "spread_bps_now": 2.0,
        },
        "context": {
            "ctx_cont": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "ctx_cat": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        },
    }


# ---------------------------------------------------------------------------
# Dimension & ordering
# ---------------------------------------------------------------------------

def test_feature_dim_v3():
    """IOV3_CLEAN dim = IOV2 (31) + 4 extras = 35."""
    assert IOV2_DIM == 31
    assert EXIT_IO_FEATURE_COUNT == 35
    assert FEATURE_DIM_V3 == EXIT_IO_FEATURE_COUNT
    assert len(ORDERED_EXIT_EXTRA_FIELDS_V3) == 4


def test_ordered_names_match_contract():
    """ordered_feature_names_v3() length and content match EXIT_IO_FEATURE_COUNT and ORDERED_EXIT_FEATURES_V3."""
    names = ordered_feature_names_v3()
    assert len(names) == EXIT_IO_FEATURE_COUNT
    assert names == ORDERED_EXIT_FEATURES_V3
    assert names[-4:] == ORDERED_EXIT_EXTRA_FIELDS_V3


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_row_to_feature_vector_v3_dim_and_finiteness():
    """Valid row → vector shape (EXIT_IO_FEATURE_COUNT,), dtype float32, no NaN/Inf."""
    row = _base_row()
    vec = row_to_feature_vector_v3(row)

    assert vec.shape == (EXIT_IO_FEATURE_COUNT,)
    assert vec.shape == (FEATURE_DIM_V3,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()


def test_v3_extras_semantics():
    """
    Verify V3 extras (4 only; no drawdown/bars_since_mfe duplicate):
    - pnl_over_atr = pnl_bps / atr_bps_now
    - spread_over_atr = spread_bps_now / atr_bps_now
    - price_dist_from_entry_bps = (price_now - entry_price) / entry_price * 10000
    - price_dist_from_entry_over_atr = price_dist_from_entry_bps / atr_bps_now
    """
    row = _base_row(
        atr_bps_now=10.0,
        entry_price=1.0,
        price_now=1.0002,
    )
    vec = row_to_feature_vector_v3(row)

    pnl_over_atr = vec[IOV2_DIM + 0]             # 2 / 10 = 0.2
    spread_over_atr = vec[IOV2_DIM + 1]          # 2 / 10 = 0.2
    price_dist_bps = vec[IOV2_DIM + 2]           # (1.0002 - 1.0) * 10000 = 2.0
    price_dist_over_atr = vec[IOV2_DIM + 3]      # 2 / 10 = 0.2

    assert pnl_over_atr == pytest.approx(0.2)
    assert spread_over_atr == pytest.approx(0.2)
    assert price_dist_bps == pytest.approx(2.0)
    assert price_dist_over_atr == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Negative paths (hard fail, no fallback)
# ---------------------------------------------------------------------------

def test_validate_row_v3_missing_entry_price_raises():
    """Missing state.entry_price → RuntimeError."""
    row = _base_row()
    del row["state"]["entry_price"]

    with pytest.raises(RuntimeError):
        validate_row_v3(row)


def test_validate_row_v3_missing_price_now_raises():
    """Missing state.price_now → RuntimeError."""
    row = _base_row()
    del row["state"]["price_now"]

    with pytest.raises(RuntimeError):
        validate_row_v3(row)


def test_row_to_feature_vector_v3_missing_entry_price_raises():
    """row_to_feature_vector_v3 must hard-fail if entry_price is missing."""
    row = _base_row()
    del row["state"]["entry_price"]

    with pytest.raises(RuntimeError):
        row_to_feature_vector_v3(row)


def test_atr_below_min_raises():
    """atr_bps_now < MIN_ATR_BPS must hard-fail (no clamping)."""
    row = _base_row(atr_bps_now=MIN_ATR_BPS * 0.1)

    with pytest.raises(RuntimeError):
        row_to_feature_vector_v3(row)


# ---------------------------------------------------------------------------
# Window validation
# ---------------------------------------------------------------------------

def test_validate_window_v3_accepts_correct_shape():
    """validate_window_v3 accepts (window_len, EXIT_IO_FEATURE_COUNT)."""
    window_len = 64
    x = np.zeros((window_len, EXIT_IO_FEATURE_COUNT), dtype=np.float32)

    validate_window_v3(
        x,
        window_len=window_len,
        feature_dim=EXIT_IO_FEATURE_COUNT,
        context="test",
    )


def test_validate_window_v3_rejects_wrong_dim():
    """validate_window_v3 rejects wrong feature_dim (must be EXIT_IO_FEATURE_COUNT)."""
    window_len = 64
    x = np.zeros((window_len, IOV2_DIM), dtype=np.float32)

    with pytest.raises(RuntimeError):
        validate_window_v3(
            x,
            window_len=window_len,
            feature_dim=IOV2_DIM,
            context="test",
        )


# ---------------------------------------------------------------------------
# Manual run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_feature_dim_v3()
    test_ordered_names_match_contract()
    test_row_to_feature_vector_v3_dim_and_finiteness()
    test_v3_extras_semantics()
    test_validate_row_v3_missing_entry_price_raises()
    test_validate_row_v3_missing_price_now_raises()
    test_row_to_feature_vector_v3_missing_entry_price_raises()
    test_atr_below_min_raises()
    test_validate_window_v3_accepts_correct_shape()
    test_validate_window_v3_rejects_wrong_dim()
    print("OK: Exit Transformer IOV3 tests passed")