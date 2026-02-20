#!/usr/bin/env python3
"""
Unit tests for Exit Transformer IO V2 contract.

ONE UNIVERSE: IOV2 = 19 (IOV1) + 6 (ctx_cont) + 6 (ctx_cat) = 31.
Verifies row_to_feature_vector_v2, dimension, no NaN/Inf, ordered fields.
"""

from __future__ import annotations

import numpy as np

from gx1.contracts.exit_transformer_io_v2 import (
    DEFAULT_CTX_CAT_DIM,
    DEFAULT_CTX_CONT_DIM,
    feature_dim_v2,
    IOV1_DIM,
    ordered_feature_names_v2,
    row_to_feature_vector_v2,
    validate_window_v2,
)


def test_iov2_dim():
    """IOV2 feature dim is 31 (19 + 6 + 6)."""
    dim = feature_dim_v2(ctx_cont_dim=6, ctx_cat_dim=6)
    assert dim == 31, f"expected 31, got {dim}"


def test_synthetic_row_31_dim_no_nan():
    """Build synthetic exits row with signals 7, entry_snapshot 5, state 7, context 6/6; vector dim 31, no NaN/Inf."""
    row = {
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
            "mfe_bps": 3.0,
            "mae_bps": -1.0,
            "dd_from_mfe_bps": 0.0,
            "bars_held": 5,
            "time_since_mfe_bars": 1,
            "atr_bps_now": 8.0,
        },
        "context": {
            "ctx_cont": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "ctx_cat": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        },
    }
    vec = row_to_feature_vector_v2(row, ctx_cont_dim=DEFAULT_CTX_CONT_DIM, ctx_cat_dim=DEFAULT_CTX_CAT_DIM)
    assert vec.shape == (31,), f"expected (31,), got {vec.shape}"
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all(), "expected no NaN/Inf"
    assert not np.any(np.isnan(vec)) and not np.any(np.isinf(vec))


def test_ordered_names_v2():
    """Ordered feature names: IOV1 then ctx_cont_0..5 then ctx_cat_0..5."""
    names = ordered_feature_names_v2(ctx_cont_dim=6, ctx_cat_dim=6)
    assert len(names) == 31
    assert names[IOV1_DIM] == "ctx_cont_0"
    assert names[IOV1_DIM + 5] == "ctx_cont_5"
    assert names[IOV1_DIM + 6] == "ctx_cat_0"
    assert names[30] == "ctx_cat_5"


def test_validate_window_v2():
    """validate_window_v2 accepts (window_len, 31) and rejects wrong shape / non-finite."""
    window_len = 64
    good = np.zeros((window_len, 31), dtype=np.float32)
    validate_window_v2(good, window_len=window_len, feature_dim=31, context="test")
    bad_shape = np.zeros((window_len, 19), dtype=np.float32)
    try:
        validate_window_v2(bad_shape, window_len=window_len, feature_dim=31, context="test")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "EXIT_TRANSFORMER_IO_V2" in str(e) or "shape" in str(e).lower()


if __name__ == "__main__":
    test_iov2_dim()
    test_synthetic_row_31_dim_no_nan()
    test_ordered_names_v2()
    test_validate_window_v2()
    print("OK: IOV2 tests passed")
