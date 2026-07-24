from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts.materialize_canonical_v3_augment import (
    add_cross_tf_momentum,
    add_smc_premium_state_interaction,
)


def test_cross_tf_momentum_treats_zero_h1_atr_as_unavailable_not_tiny() -> None:
    close = np.arange(20, dtype=np.float64) + 2_000.0
    h1_atr = np.zeros(20, dtype=np.float64)
    h1_atr[13:] = 2.0

    out = add_cross_tf_momentum(
        pd.DataFrame({"close": close, "_v1h1_atr": h1_atr})
    )["m5h1_momentum"].to_numpy(dtype=np.float64)

    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[:13], np.zeros(13))
    np.testing.assert_allclose(out[13:], np.full(7, 6.0), rtol=0.0, atol=0.0)


def test_cross_tf_momentum_carries_causal_nan_warmup_prefix() -> None:
    # Causal HTF construction emits a NaN warmup prefix (not neutral zero).
    close = np.arange(20, dtype=np.float64) + 2_000.0
    h1_atr = np.full(20, np.nan, dtype=np.float64)
    h1_atr[13:] = 2.0

    out = add_cross_tf_momentum(
        pd.DataFrame({"close": close, "_v1h1_atr": h1_atr})
    )["m5h1_momentum"].to_numpy(dtype=np.float64)

    assert np.isnan(out[:13]).all()
    assert np.isfinite(out[13:]).all()
    np.testing.assert_allclose(out[13:], np.full(7, 6.0), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "column,value,error",
    (
        ("_v1h1_atr", -1.0, "H1 ATR must be non-negative"),
        ("_v1h1_atr", np.nan, "non-finite values after causal warmup"),
        ("close", np.nan, "close must be finite"),
    ),
)
def test_cross_tf_momentum_fails_closed_on_invalid_source(
    column: str,
    value: float,
    error: str,
) -> None:
    frame = pd.DataFrame(
        {
            "close": np.arange(20, dtype=np.float64) + 2_000.0,
            "_v1h1_atr": np.ones(20, dtype=np.float64),
        }
    )
    frame.loc[15, column] = value

    with pytest.raises(RuntimeError, match=error):
        add_cross_tf_momentum(frame)


def test_smc_premium_state_is_conditional_and_unknown_is_not_uptrend() -> None:
    frame = pd.DataFrame(
        {
            "smc_premium_discount": [0.8, 0.8, 0.8],
            "smc_swing_state": [0, 3, 4],
        }
    )

    out = add_smc_premium_state_interaction(frame)["smc_premium_state"]

    np.testing.assert_allclose(out, [0.8, 0.0, 0.0], rtol=0.0, atol=1e-7)


@pytest.mark.parametrize(
    "column,value,error",
    (
        ("smc_premium_discount", np.nan, "must be finite"),
        ("smc_premium_discount", 1.1, "within"),
        ("smc_swing_state", np.nan, "finite enum"),
        ("smc_swing_state", 5.0, "enum 0..4"),
    ),
)
def test_smc_premium_state_fails_closed_on_invalid_source(
    column: str,
    value: float,
    error: str,
) -> None:
    frame = pd.DataFrame(
        {
            "smc_premium_discount": [0.4, 0.7],
            "smc_swing_state": [0, 3],
        }
    )
    frame.loc[1, column] = value

    with pytest.raises(RuntimeError, match=error):
        add_smc_premium_state_interaction(frame)
