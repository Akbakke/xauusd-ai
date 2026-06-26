from __future__ import annotations

import numpy as np
import pandas as pd

from gx1.scripts.train_xgb_universal_multihead_v2 import (
    compute_class_weights,
    compute_fixed_horizon_spread_pnl_labels,
)


def test_fixed_horizon_spread_pnl_labels_match_entry_direction_contract() -> None:
    df = pd.DataFrame(
        {
            "bid_close": [100.00, 100.30, 99.60, 100.01, 100.02],
            "ask_close": [100.02, 100.32, 99.62, 100.03, 100.04],
        }
    )

    labels, valid, hit_code, pnl_long, pnl_short = compute_fixed_horizon_spread_pnl_labels(
        df,
        lookahead_bars=1,
        threshold_bps=15.0,
    )

    assert labels.tolist() == [0, 1, 0, 2, -1]
    assert hit_code.tolist() == [0, 1, 0, 2, -1]
    assert valid.tolist() == [True, True, True, True, False]
    assert pnl_long[0] > 15.0
    assert pnl_short[1] > 15.0
    assert pnl_long[3] < 15.0 and pnl_short[3] < 15.0


def test_fixed_horizon_spread_pnl_labels_reject_missing_bid_ask() -> None:
    df = pd.DataFrame({"close": [100.0, 101.0]})

    try:
        compute_fixed_horizon_spread_pnl_labels(df, lookahead_bars=1, threshold_bps=15.0)
    except KeyError as exc:
        assert "bid/ask close" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing bid/ask close columns to fail")


def test_fixed_horizon_spread_pnl_labels_invalid_prices_are_not_trainable() -> None:
    df = pd.DataFrame(
        {
            "bid_close": [100.00, np.nan, 100.00],
            "ask_close": [100.02, 100.04, 100.02],
        }
    )

    labels, valid, hit_code, _pnl_long, _pnl_short = compute_fixed_horizon_spread_pnl_labels(
        df,
        lookahead_bars=1,
        threshold_bps=15.0,
    )

    assert labels.tolist() == [-1, -1, -1]
    assert hit_code.tolist() == [-1, -1, -1]
    assert valid.tolist() == [False, False, False]


def test_compute_class_weights_power_interpolates_toward_unweighted() -> None:
    y = np.array([0, 1, 2, 2], dtype=np.int64)

    full = compute_class_weights(y, n_classes=3, power=1.0)
    mild = compute_class_weights(y, n_classes=3, power=0.5)
    off = compute_class_weights(y, n_classes=3, power=0.0)

    assert full[0] == full[1]
    assert full[0] > mild[0] > off[0]
    assert full[2] < mild[2] < off[2]
    assert off == {0: 1.0, 1: 1.0, 2: 1.0}
    for weights in (full, mild, off):
        sample_mean = np.mean([weights[int(cls)] for cls in y])
        assert np.isclose(sample_mean, 1.0)


def test_compute_class_weights_rejects_negative_power() -> None:
    try:
        compute_class_weights(np.array([0, 1, 2], dtype=np.int64), power=-0.1)
    except ValueError as exc:
        assert "power" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected negative class weight power to fail")
