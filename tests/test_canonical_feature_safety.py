from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.features.basic_v1 import (
    BASIC_V1_OBSERVED_SPREAD_FEATURES,
    _require_observed_spread_input,
    _validate_causal_feature_column,
)
from gx1.features.model_native_market_context_v1 import (
    derive_model_native_trend_regime_id,
    derive_observed_spread_bps,
)
from gx1.features.smc_v1 import compute_smc_features
from gx1.contracts.entry_model_native_state_v2 import TrainRankReferenceV2
from gx1.execution.v12_ctx_augment_live import (
    _add_regime_categoricals,
    _add_spread_atr_bps,
)
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
            "high": [100.5, 100.5],
            "low": [99.5, 99.5],
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

    with pytest.raises(RuntimeError, match="MODEL_NATIVE_RANK_SOURCE_FIELDS_MISSING"):
        _add_spread_atr_bps(df)


def test_live_ctx_rank_formula_does_not_overwrite_canonical_atr() -> None:
    frame = pd.DataFrame(
        {
            "atr": [9.0, 10.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "bid_close": [99.9, 100.9],
            "ask_close": [100.1, 101.1],
        }
    )

    _add_spread_atr_bps(frame)

    assert frame["atr"].tolist() == [9.0, 10.0]
    assert np.isfinite(frame[["atr_bps", "spread_bps"]]).all().all()


def test_regime_categories_omit_train_fit_buckets_without_reference() -> None:
    frame = pd.DataFrame(
        {
            "D1_dist_from_ema200_atr": [-2.0, 0.0, 2.0],
            "atr_bps": [1.0, 2.0, 3.0],
            "spread_bps": [0.1, 0.2, 0.3],
            "atr_bucket": [4, 4, 4],
            "spread_bucket": [4, 4, 4],
        }
    )

    _add_regime_categoricals(frame)

    assert "atr_bucket" not in frame
    assert "spread_bucket" not in frame
    assert frame["trend_regime_id"].tolist() == [0, 1, 2]


def test_shared_trend_regime_formula_is_strict_at_exact_boundaries() -> None:
    values = np.asarray([-2.0, -1.0, 1.0, 2.0], dtype=np.float64)
    assert derive_model_native_trend_regime_id(values).tolist() == [0, 1, 1, 2]

    with pytest.raises(RuntimeError, match="MODEL_NATIVE_CONTEXT_NONFINITE"):
        derive_model_native_trend_regime_id(np.asarray([np.nan]))
    with pytest.raises(RuntimeError, match="MODEL_NATIVE_TREND_REGIME_SHAPE"):
        derive_model_native_trend_regime_id(values[:, None])


def test_regime_categories_use_one_explicit_train_reference() -> None:
    reference = TrainRankReferenceV2(
        path=Path("unused.npz"),
        sha256="0" * 64,
        sidecar_sha256="1" * 64,
        sidecar={},
        fit_start_utc=pd.Timestamp("2026-01-01T00:00:00Z"),
        fit_end_utc=pd.Timestamp("2026-01-02T00:00:00Z"),
        fit_row_count=5,
        atr_bps_sorted=np.asarray([1, 2, 3, 4, 5], dtype=np.float64),
        spread_bps_sorted=np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64),
    )
    frame = pd.DataFrame(
        {
            "D1_dist_from_ema200_atr": [-2.0, 0.0, 2.0],
            "atr_bps": [1.0, 3.0, 5.0],
            "spread_bps": [0.1, 0.3, 0.5],
        }
    )

    _add_regime_categoricals(frame, rank_reference=reference)

    assert frame["atr_bucket"].tolist() == [1, 3, 4]
    assert frame["spread_bucket"].tolist() == [1, 3, 4]


def test_basic_v1_spread_owner_does_not_require_slippage() -> None:
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

    _require_observed_spread_input(df)

    assert BASIC_V1_OBSERVED_SPREAD_FEATURES == (
        "_v1_spread_p",
        "_v1_spread_z",
    )
    assert "_v1_slip_bps" not in df.columns
    assert "_v1_cost_bps_est" not in df.columns


def test_basic_v1_ignores_post_order_slippage_as_a_feature_source() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0, 100.1],
            "bid_close": [100.0, 100.0],
            "ask_close": [100.1, 100.2],
            "slippage_bps": [0.5, 0.75],
        }
    )

    _require_observed_spread_input(df)

    assert "spread_pct" in df.columns
    assert "_v1_slip_bps" not in df.columns
    assert "_v1_cost_bps_est" not in df.columns


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
