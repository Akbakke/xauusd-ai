import numpy as np
import pandas as pd

from gx1.scripts.materialize_build_canonical_features_v1 import add_high_level_basics
from gx1.scripts.add_ctx_cont_columns_to_prebuilt import _derive_spread_bps_from_available
from gx1.execution.v12_ctx_augment_live import _add_spread_atr_bps


def test_high_level_body_pct_is_clipped_for_degenerate_ohlc() -> None:
    n = 120
    df = pd.DataFrame(
        {
            "bid_close": np.full(n, 100.0),
            "ask_close": np.full(n, 100.1),
            "ask_high": np.full(n, 100.2),
            "bid_low": np.full(n, 99.8),
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.5),
            "low": np.full(n, 99.5),
            "close": np.full(n, 100.25),
        }
    )
    df.loc[5, ["open", "high", "low", "close"]] = [90.0, 100.0, 100.0, 110.0]

    out = add_high_level_basics(df.copy())

    assert np.isfinite(out["body_pct"]).all()
    assert out["body_pct"].between(0.0, 1.0).all()
    assert out.loc[5, "body_pct"] == 1.0


def test_add_ctx_derives_spread_bps_from_bid_ask_before_zero_fallback() -> None:
    df = pd.DataFrame(
        {
            "bid_close": [100.0, 200.0, 0.0, 100.0],
            "ask_close": [100.1, 200.4, 10.0, 99.5],
        }
    )

    spread = _derive_spread_bps_from_available(df)

    np.testing.assert_allclose(spread[:2], [10.0, 20.0])
    assert spread[2] == 0.0
    assert spread[3] == 0.0
    assert np.isfinite(spread).all()


def test_add_ctx_existing_spread_bps_wins_over_bid_ask() -> None:
    df = pd.DataFrame(
        {
            "spread_bps": [1.25, 1.5, -2.0],
            "bid_close": [100.0, 200.0, 300.0],
            "ask_close": [101.0, 202.0, 303.0],
        }
    )

    np.testing.assert_allclose(_derive_spread_bps_from_available(df), [1.25, 1.5, 0.0])


def test_add_ctx_spread_close_fallback_when_bid_ask_missing() -> None:
    df = pd.DataFrame({"spread": [0.05, 0.10], "close": [100.0, 200.0]})

    np.testing.assert_allclose(_derive_spread_bps_from_available(df), [5.0, 5.0])


def test_live_ctx_spread_bps_clips_negative_bid_ask_glitches() -> None:
    df = pd.DataFrame(
        {
            "_v1_atr14": [1.0, 1.0],
            "close": [100.0, 100.0],
            "bid_close": [100.0, 100.0],
            "ask_close": [100.1, 99.5],
        }
    )

    _add_spread_atr_bps(df)

    np.testing.assert_allclose(df["spread_bps"].to_numpy(), [10.0, 0.0])
