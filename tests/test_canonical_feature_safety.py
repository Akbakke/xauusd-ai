import numpy as np
import pandas as pd
import pytest

from gx1.features.basic_v1 import (
    _require_observed_execution_cost_inputs,
    _validate_causal_feature_column,
    build_basic_v1,
)
from gx1.features.model_native_market_context_v1 import derive_observed_spread_bps
from gx1.features.smc_v1 import compute_smc_features
from gx1.execution.v12_ctx_augment_live import _add_spread_atr_bps
from gx1.scripts.materialize_build_canonical_features_v1 import add_high_level_basics


def test_high_level_rejects_degenerate_ohlc_before_plus5_features() -> None:
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
            "volume": np.full(n, 10.0),
        }
    )
    df.loc[5, ["open", "high", "low", "close"]] = [90.0, 100.0, 100.0, 110.0]

    with pytest.raises(RuntimeError, match="PLUS5_OHLC_INVALID"):
        add_high_level_basics(df.copy())


def test_high_level_plus5_owner_recomputes_dependent_vwap_interaction() -> None:
    n = 120
    close = 100.0 + np.linspace(0.0, 2.0, n)
    frame = pd.DataFrame(
        {
            "bid_close": close - 0.05,
            "ask_close": close + 0.05,
            "ask_high": close + 0.6,
            "bid_low": close - 0.6,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.linspace(10.0, 30.0, n),
            "_v1h1_ema_diff": np.full(n, 2.0),
            "_v1_int_vwap_h1": np.full(n, 999.0),
        }
    )

    out = add_high_level_basics(frame)

    np.testing.assert_allclose(
        out["_v1_int_vwap_h1"],
        out["_v1_vwap_drift48"] * 2.0,
    )


def test_add_ctx_derives_spread_bps_from_valid_bid_ask() -> None:
    df = pd.DataFrame(
        {
            "bid_close": [100.0, 200.0],
            "ask_close": [100.1, 200.4],
        }
    )

    spread = derive_observed_spread_bps(df)

    np.testing.assert_allclose(spread, [10.0, 20.0])
    assert np.isfinite(spread).all()


@pytest.mark.parametrize(
    ("bid", "ask"),
    [(0.0, 10.0), (100.0, 99.5), (float("nan"), 100.0)],
)
def test_add_ctx_rejects_invalid_bid_ask_instead_of_zero_fill(
    bid: float, ask: float
) -> None:
    with pytest.raises(RuntimeError, match="MODEL_NATIVE_CONTEXT_(INVALID|NONFINITE)"):
        derive_observed_spread_bps(
            pd.DataFrame({"bid_close": [bid], "ask_close": [ask]})
        )


def test_add_ctx_existing_spread_bps_wins_over_bid_ask() -> None:
    df = pd.DataFrame(
        {
            "spread_bps": [1.25, 1.5],
            "bid_close": [100.0, 200.0],
            "ask_close": [101.0, 202.0],
        }
    )

    np.testing.assert_allclose(derive_observed_spread_bps(df), [1.25, 1.5])


def test_add_ctx_rejects_negative_existing_spread() -> None:
    with pytest.raises(RuntimeError, match="negative values"):
        derive_observed_spread_bps(pd.DataFrame({"spread_bps": [-2.0]}))


def test_add_ctx_spread_close_fallback_when_bid_ask_missing() -> None:
    df = pd.DataFrame({"spread": [0.05, 0.10], "close": [100.0, 200.0]})

    np.testing.assert_allclose(derive_observed_spread_bps(df), [5.0, 5.0])


def test_add_ctx_rejects_missing_spread_source() -> None:
    with pytest.raises(RuntimeError, match="observed spread requires"):
        derive_observed_spread_bps(pd.DataFrame({"close": [100.0]}))


def test_live_ctx_rejects_negative_bid_ask_glitches() -> None:
    df = pd.DataFrame(
        {
            "_v1_atr14": [1.0, 1.0],
            "close": [100.0, 100.0],
            "bid_close": [100.0, 100.0],
            "ask_close": [100.1, 99.5],
        }
    )

    with pytest.raises(RuntimeError, match="MODEL_NATIVE_CONTEXT_INVALID"):
        _add_spread_atr_bps(df)


def test_live_ctx_rejects_missing_atr_source_before_producing_features() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0],
            "bid_close": [100.0],
            "ask_close": [100.1],
        }
    )

    with pytest.raises(RuntimeError, match="LIVE_CTX_ATR_SOURCE_MISSING"):
        _add_spread_atr_bps(df)

    assert "atr_bps" not in df.columns
    assert "spread_bps" not in df.columns


def test_basic_v1_execution_cost_owner_rejects_missing_slippage() -> None:
    df = pd.DataFrame(
        {
            "open": [100.0, 100.1],
            "high": [100.2, 100.3],
            "low": [99.8, 99.9],
            "close": [100.0, 100.1],
            "bid_close": [99.9, 100.0],
            "ask_close": [100.1, 100.2],
        },
        index=pd.date_range("2026-07-20", periods=2, freq="5min", tz="UTC"),
    )

    with pytest.raises(RuntimeError, match="BASIC_V1_SLIPPAGE_SOURCE_MISSING"):
        build_basic_v1(df)

    assert not any(column.startswith("_v1") for column in df.columns)


def test_basic_v1_execution_cost_owner_uses_exact_observed_inputs() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0, 100.1],
            "bid_close": [100.0, 100.0],
            "ask_close": [100.1, 100.2],
            "slippage_bps": [0.5, 0.75],
        }
    )

    _require_observed_execution_cost_inputs(df)

    np.testing.assert_allclose(df["spread_pct"], [0.001, 0.002])
    np.testing.assert_allclose(df["slippage_bps"], [0.5, 0.75])


def test_basic_v1_final_pack_preserves_only_causal_nan_prefix() -> None:
    values = np.asarray([np.nan, np.nan, 1.0, 2.0])

    observed = _validate_causal_feature_column(values, name="_v1_test")

    np.testing.assert_array_equal(observed, values)
    with pytest.raises(RuntimeError, match="BASIC_V1_FEATURE_NONFINITE_GAP"):
        _validate_causal_feature_column(
            np.asarray([np.nan, 1.0, np.nan]),
            name="_v1_test",
        )


@pytest.mark.parametrize(
    ("atr_values", "error_code"),
    [
        ([1.0, float("nan"), 1.0], "SMC_SOURCE_NONFINITE"),
        ([1.0, 0.0, 1.0], "SMC_SOURCE_INVALID"),
    ],
)
def test_smc_rejects_unavailable_atr_instead_of_using_sentinel(
    atr_values: list[float],
    error_code: str,
) -> None:
    frame = pd.DataFrame(
        {
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "atr": atr_values,
        }
    )

    with pytest.raises(RuntimeError, match=error_code):
        compute_smc_features(frame)


def test_smc_rejects_missing_atr_instead_of_using_one() -> None:
    frame = pd.DataFrame(
        {
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
        }
    )

    with pytest.raises(RuntimeError, match="SMC_SOURCE_MISSING"):
        compute_smc_features(frame)
