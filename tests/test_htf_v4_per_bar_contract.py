"""V4-only per-bar multi-timeframe feature, routing, and clock contract.

V29 Phase A stage 2 (2026-08-11): the per-TF surface carries the 21 event
fields plus the 11 level-registry and 30 trendline-registry fields; every
expected count below is DERIVED from the declared owner tuples.  The V29
registry blocks require an explicit constants payload (no default exists);
tests use the shared synthetic-execution payload from
``tests.htf_v29_registry_test_support`` (proves the code runs; never a
production value).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gx1.features import htf_features as htf
from tests.htf_v29_registry_test_support import (
    synthetic_v29_registry_constants,
)

V29_TEST_REGISTRY_CONSTANTS = synthetic_v29_registry_constants()


def _compute_v4(bars: pd.DataFrame, *, timeframe: str) -> pd.DataFrame:
    return htf.compute_per_bar_features_v4(
        bars,
        timeframe=timeframe,
        v29_registry_constants=V29_TEST_REGISTRY_CONSTANTS,
    )


def _bars(n: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 2000.0 + np.cumsum(rng.normal(0.0, 1.5, size=n))
    high = close + np.abs(rng.normal(0.0, 1.0, size=n))
    low = close - np.abs(rng.normal(0.0, 1.0, size=n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    index = pd.date_range("2021-01-04", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "volume": np.abs(rng.normal(500.0, 50.0, size=n)),
        },
        index=index,
    )


EXPECTED_V4_GROUP_A_BASE_FEATURES = (
    "atr_bps_14",
    "rsi14_centered",
    "mom_5_atr",
    "mom_20_atr",
    "close_open_atr",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "ema20_dist_atr",
    "ema50_dist_atr",
    "ema100_dist_atr",
    "ema200_dist_atr",
    "ema20_slope_atr",
    "ema50_slope_atr",
    "ema200_slope_atr",
    "ema_stack_aligned_v2",
    "regime_class_id",
    "vwap_local_cycle_dist_atr",
    "vwap20_dist_atr",
    "vwap96_dist_atr",
    "vwap_local_cycle_slope_atr",
    "bb_position",
    "bb_width_atr",
    "adx_centered",
    "trend_age_bars_norm",
)


EXPECTED_V29_TREND_EVENT_FEATURES = (
    "ema50_200_spread_atr",
    "ema50_200_bull_state",
    "ema50_200_cross_up",
    "ema50_200_cross_down",
    "ema50_200_cross_age_norm",
    "price_x_ema50_cross_up",
    "price_x_ema50_cross_down",
    "price_x_ema200_cross_up",
    "price_x_ema200_cross_down",
    "price_above_ema50_age_norm",
    "price_above_ema200_age_norm",
)
EXPECTED_V29_MOMENTUM_EVENT_FEATURES = (
    "rsi_cross_up_30",
    "rsi_cross_down_70",
    "rsi_cross_up_50",
    "rsi_cross_down_50",
    "rsi_extreme_age_norm",
    "mom20_sign_flip_up",
    "mom20_sign_flip_down",
    "bear_divergence_event",
    "bull_divergence_event",
    "divergence_age_norm",
)


def test_v4_is_one_exact_derived_field_contract() -> None:
    from gx1.features.level_registry_v1 import LEVEL_REGISTRY_MTF_FEATURE_NAMES
    from gx1.features.trendline_registry_v1 import (
        TRENDLINE_REGISTRY_FEATURE_NAMES_V1,
    )

    expected_width = (
        111
        + len(EXPECTED_V29_TREND_EVENT_FEATURES)
        + len(EXPECTED_V29_MOMENTUM_EVENT_FEATURES)
        + len(LEVEL_REGISTRY_MTF_FEATURE_NAMES)
        + len(TRENDLINE_REGISTRY_FEATURE_NAMES_V1)
    )
    assert htf.MULTI_TF_FEATURE_COUNT_V4 == expected_width
    assert htf.MULTI_TF_V4_GROUP_A_BASE_FEATURES == (
        EXPECTED_V4_GROUP_A_BASE_FEATURES
    )
    assert (
        htf.MULTI_TF_PER_BAR_FEATURES_V4[:25]
        == EXPECTED_V4_GROUP_A_BASE_FEATURES
    )
    assert len(set(htf.MULTI_TF_PER_BAR_FEATURES_V4)) == expected_width
    assert htf.MULTI_TF_V4_TREND_EVENT_FEATURES == (
        EXPECTED_V29_TREND_EVENT_FEATURES
    )
    assert htf.MULTI_TF_V4_MOMENTUM_EVENT_FEATURES == (
        EXPECTED_V29_MOMENTUM_EVENT_FEATURES
    )
    registry_tail = tuple(LEVEL_REGISTRY_MTF_FEATURE_NAMES) + tuple(
        TRENDLINE_REGISTRY_FEATURE_NAMES_V1
    )
    assert htf.MULTI_TF_PER_BAR_FEATURES_V4[-len(registry_tail):] == (
        registry_tail
    )
    event_block = (
        EXPECTED_V29_TREND_EVENT_FEATURES
        + EXPECTED_V29_MOMENTUM_EVENT_FEATURES
    )
    tail_start = -(len(registry_tail) + len(event_block))
    assert (
        htf.MULTI_TF_PER_BAR_FEATURES_V4[tail_start : -len(registry_tail)]
        == event_block
    )


def test_v4_candlestick_names_come_from_the_owner() -> None:
    from gx1.features.entry_candlestick_patterns_v1 import (
        CANDLESTICK_PATTERN_FEATURE_NAMES,
    )

    expected = tuple(
        f"mtf_{name.split('.', 1)[1] if '.' in name else name}"
        for name in CANDLESTICK_PATTERN_FEATURE_NAMES
    )
    assert htf.MULTI_TF_V4_CANDLESTICK_FEATURES == expected


def test_v4_rejects_missing_volume_and_non_m5_cache_source() -> None:
    bars = _bars(400, seed=3)
    with np.testing.assert_raises_regex(RuntimeError, "volume"):
        _compute_v4(bars.drop(columns="volume"), timeframe="M5")

    off_grid = bars.copy()
    off_grid.index = off_grid.index + pd.Timedelta(minutes=1)
    with np.testing.assert_raises_regex(
        RuntimeError,
        "HTF_INPUT_FAIL|HTF_V4_SOURCE_TIMESTAMP_GEOMETRY_INVALID",
    ):
        htf.build_multi_tf_per_bar_features_v4(
            off_grid,
            v29_registry_constants=V29_TEST_REGISTRY_CONSTANTS,
        )


def test_v4_routes_every_field_to_all_eight_specialists() -> None:
    from gx1.features.entry_specialist_feature_groups_v1 import (
        MODEL_NATIVE_TRAINING_SPECIALISTS,
        require_multi_tf_specialist_routing_v4,
    )
    from gx1.features.level_registry_v1 import LEVEL_REGISTRY_MTF_FEATURE_NAMES
    from gx1.features.trendline_registry_v1 import (
        TRENDLINE_REGISTRY_FEATURE_NAMES_V1,
    )

    routing = require_multi_tf_specialist_routing_v4(
        htf.MULTI_TF_PER_BAR_FEATURES_V4
    )
    assert htf.MULTI_TF_FEATURE_COUNT_V4 == len(
        htf.MULTI_TF_PER_BAR_FEATURES_V4
    )
    assert tuple(routing) == MODEL_NATIVE_TRAINING_SPECIALISTS
    # Pre-V29 audited per-specialist widths plus the V29 additions, derived
    # from the declared owner tuples (design doc §5.2.3).
    assert {name: len(indices) for name, indices in routing.items()} == {
        "structure_swing_encoder": 5,
        "smc_liquidity_encoder": 11 + len(LEVEL_REGISTRY_MTF_FEATURE_NAMES),
        "trend_ema_encoder": 10
        + len(htf.MULTI_TF_V4_TREND_EVENT_FEATURES),
        "vol_compression_encoder": 2,
        "momentum_flow_encoder": 4
        + len(htf.MULTI_TF_V4_MOMENTUM_EVENT_FEATURES),
        "session_regime_encoder": 5,
        "chart_geometry_encoder": 10
        + len(TRENDLINE_REGISTRY_FEATURE_NAMES_V1),
        "price_action_candle_encoder": 64,
    }
    flattened = [index for indices in routing.values() for index in indices]
    assert sorted(flattened) == list(range(htf.MULTI_TF_FEATURE_COUNT_V4))
    assert "vwap_session_dist_atr" not in htf.MULTI_TF_PER_BAR_FEATURES_V4
    assert "vwap_session_slope_atr" not in htf.MULTI_TF_PER_BAR_FEATURES_V4
    assert "vwap_local_cycle_dist_atr" in htf.MULTI_TF_PER_BAR_FEATURES_V4


def test_v4_smc_and_geometry_are_causal_and_have_one_warmup_prefix() -> None:
    bars = _bars(4000, seed=23)
    original = _compute_v4(bars, timeframe="M5")
    matrix = original.to_numpy(dtype=np.float64)
    warmup = htf.validate_causal_feature_matrix(
        matrix,
        expected_width=htf.MULTI_TF_FEATURE_COUNT_V4,
        context="HTF_V4_TEST",
    )
    assert warmup > 0
    assert np.isfinite(matrix[warmup:]).all()

    cutoff = 2500
    changed = bars.copy()
    changed.iloc[cutoff:, changed.columns.get_loc("open")] *= 1.1
    changed.iloc[cutoff:, changed.columns.get_loc("high")] *= 1.1
    changed.iloc[cutoff:, changed.columns.get_loc("low")] *= 1.1
    changed.iloc[cutoff:, changed.columns.get_loc("close")] *= 1.1
    future_changed = _compute_v4(changed, timeframe="M5")
    assert np.array_equal(
        original.iloc[:cutoff].to_numpy(),
        future_changed.iloc[:cutoff].to_numpy(),
        equal_nan=True,
    )

    # A valid trend can place the latest confirmed low above an older
    # confirmed high. That remains available structure evidence.
    trending = _compute_v4(_bars(4000, seed=0), timeframe="M5")
    trending_matrix = trending.to_numpy(dtype=np.float64)
    trending_warmup = htf.validate_causal_feature_matrix(
        trending_matrix,
        expected_width=htf.MULTI_TF_FEATURE_COUNT_V4,
        context="HTF_V4_TRENDING_TEST",
    )
    assert np.isfinite(trending_matrix[trending_warmup:]).all()


def test_v4_equal_latest_high_low_pivots_use_confirmed_pivot_envelope(
    monkeypatch,
) -> None:
    """Equal latest pivots are valid XAU structure, not an interior data gap."""
    from gx1.features import smc_v1 as smc

    rows = 12
    high = np.full(rows, 101.0)
    low = np.full(rows, 99.0)
    close = np.full(rows, 100.0)
    high[1] = 105.0
    high[3] = 100.0
    low[2] = 95.0
    low[4] = 100.0
    close[3] = 100.0
    close[4] = 100.0
    frame = pd.DataFrame(
        {"high": high, "low": low, "close": close, "atr": np.ones(rows)}
    )

    def fixed_pivots(_high, _low, _lookback):
        swing_high = np.zeros(rows, dtype=bool)
        swing_low = np.zeros(rows, dtype=bool)
        swing_high[[1, 3]] = True
        swing_low[[2, 4]] = True
        return swing_high, swing_low

    monkeypatch.setattr(smc, "_detect_swing_pivots", fixed_pivots)
    built = smc.compute_smc_mtf_primitives_v1(frame, swing_lookback=1)
    matrix = built.to_numpy(dtype=np.float64)

    # At row 5 both latest confirmed pivots equal 100. The causal envelope of
    # the four already-confirmed pivots remains [95,105], so every subsequent
    # row is finite and carries a mathematically defined 0.5 position.
    assert np.isfinite(matrix[5:]).all()
    assert built.loc[5, "mtf_smc_premium_discount"] == 0.5
    assert built.loc[5, "mtf_smc_range_width_atr"] == 10.0


def test_v4_removes_cross_owner_duplicate_smc_geometry_fields() -> None:
    from gx1.features.smc_v1 import (
        SMC_MTF_FEATURE_NAMES_V1,
        SMC_MTF_GEOMETRY_FEATURE_NAMES_V1,
    )

    assert "mtf_smc_premium_discount" in SMC_MTF_FEATURE_NAMES_V1
    assert "mtf_smc_range_width_atr" in SMC_MTF_FEATURE_NAMES_V1
    assert "mtf_geometry_channel_position" not in (
        SMC_MTF_GEOMETRY_FEATURE_NAMES_V1
    )
    assert "mtf_geometry_channel_width_atr" not in (
        SMC_MTF_GEOMETRY_FEATURE_NAMES_V1
    )
    matrix = _compute_v4(_bars(4000, seed=31), timeframe="M5")
    assert (
        matrix["mtf_smc_choch_up"].sum()
        + matrix["mtf_smc_choch_down"].sum()
    ) > 0.0


def test_resolution_windows_must_form_strict_wall_clock_pyramid() -> None:
    accepted = htf.require_multi_tf_resolution_pyramid(
        {"M5": 16, "M15": 16, "H1": 16, "H4": 8, "D1": 8}
    )
    spans = list(accepted["coverage_seconds"].values())
    assert all(left < right for left, right in zip(spans, spans[1:]))

    with np.testing.assert_raises_regex(
        RuntimeError,
        "MULTI_TF_RESOLUTION_PYRAMID_COVERAGE_INVALID",
    ):
        htf.require_multi_tf_resolution_pyramid(
            {"M5": 500, "M15": 16, "H1": 16, "H4": 8, "D1": 8}
        )


def test_exact_resolution_pyramid_is_sliceable_across_all_split_boundaries() -> None:
    features: dict[str, pd.DataFrame] = {}
    end = pd.Timestamp("2026-01-21T00:00:00Z")
    for timeframe, rule in htf.MULTI_TF_RESAMPLE_RULES.items():
        index = pd.date_range(end=end, periods=400, freq=rule)
        values = np.ones(
            (len(index), htf.MULTI_TF_FEATURE_COUNT_V4),
            dtype=np.float32,
        )
        values[:10] = np.nan
        frame = pd.DataFrame(
            values,
            index=index,
            columns=htf.MULTI_TF_PER_BAR_FEATURES_V4,
        )
        frame.attrs["ts_int64"] = index.asi8.astype(np.int64, copy=True)
        frame.attrs["feats_np"] = values
        frame.attrs["causal_warmup_rows"] = 10
        frame.attrs["htf_feature_contract"] = htf.HTF_V4_MATRIX_CONTRACT
        features[timeframe] = frame

    route_split_times = {
        "entry": {
            "train": pd.date_range(
                "2026-01-20T00:00:00Z", periods=2, freq="5min"
            ),
            "val": pd.date_range(
                "2026-01-20T00:10:00Z", periods=2, freq="5min"
            ),
        },
        "exit": {
            "train": pd.date_range(
                "2026-01-20T00:00:00Z", periods=2, freq="1min"
            ),
            "val": pd.date_range(
                "2026-01-20T00:02:00Z", periods=2, freq="1min"
            ),
        },
    }
    lengths = {"M5": 2, "M15": 2, "H1": 2, "H4": 2, "D1": 2}
    proof = htf.require_multi_tf_decision_window_coverage(
        features,
        per_tf_seq_lens=lengths,
        decision_times_by_route_split=route_split_times,
    )

    assert proof["all_route_split_boundaries_sliceable"] is True
    assert proof["schema_version"] == (
        "entry_exit_multi_tf_decision_window_coverage_v2"
    )
    assert proof["routes"]["entry"][
        "target_availability_shift_seconds"
    ] == 300
    assert proof["routes"]["exit"][
        "target_availability_shift_seconds"
    ] == 60
    assert proof["per_tf"]["M5"]["routes"]["entry"] == {
        "enabled": False,
        "boundaries": {},
    }
    assert proof["per_tf"]["M5"]["routes"]["exit"]["enabled"] is True
    assert "test" not in str(proof).lower()
    assert proof["resolution_pyramid"]["per_tf_seq_lens"] == lengths
    assert set(proof["per_tf"]) == set(htf.MULTI_TF_RESAMPLE_RULES)
    assert len(proof["contract_sha256"]) == 64

    features["D1"].attrs["feats_np"][:399] = np.nan
    features["D1"].attrs["causal_warmup_rows"] = 399
    with np.testing.assert_raises_regex(
        RuntimeError,
        "MULTI_TF_DECISION_COVERAGE_UNAVAILABLE",
    ):
        htf.require_multi_tf_decision_window_coverage(
            features,
            per_tf_seq_lens=lengths,
            decision_times_by_route_split=route_split_times,
        )

    stale = dict(proof)
    stale["schema_version"] = "entry_multi_tf_decision_window_coverage_v1"
    with np.testing.assert_raises_regex(
        RuntimeError,
        "MULTI_TF_DECISION_COVERAGE_METADATA_INVALID",
    ):
        htf.require_multi_tf_decision_window_coverage_metadata(
            stale,
            per_tf_seq_lens=lengths,
        )


def test_shared_v4_cache_has_exact_entry_and_exit_routes_and_clocks() -> None:
    from gx1.contracts.entry_exit_feature_base_v1 import (
        ENTRY_DECISION_BAR_SECONDS,
        ENTRY_MTF_CONTEXT_TIMEFRAMES,
        EXIT_DECISION_BAR_SECONDS,
        EXIT_MTF_CONTEXT_TIMEFRAMES,
    )

    decision = pd.Timestamp("2026-01-20T18:00:00Z")
    features: dict[str, pd.DataFrame] = {}
    for timeframe, rule in htf.MULTI_TF_RESAMPLE_RULES.items():
        index = pd.date_range(end=decision, periods=400, freq=rule)
        values = np.repeat(
            np.arange(len(index), dtype=np.float32).reshape(-1, 1),
            htf.MULTI_TF_FEATURE_COUNT_V4,
            axis=1,
        )
        frame = pd.DataFrame(
            values,
            index=index,
            columns=htf.MULTI_TF_PER_BAR_FEATURES_V4,
        )
        frame.attrs["ts_int64"] = index.asi8.astype(np.int64, copy=True)
        frame.attrs["feats_np"] = values
        frame.attrs["causal_warmup_rows"] = 0
        frame.attrs["htf_feature_contract"] = htf.HTF_V4_MATRIX_CONTRACT
        features[timeframe] = frame
    lengths = {tf: 2 for tf in features}

    entry = htf.get_model_native_multi_tf_route_windows(
        features,
        decision_bar_start=decision,
        per_tf_seq_lens=lengths,
        route_timeframes=ENTRY_MTF_CONTEXT_TIMEFRAMES,
        base_bar_duration=pd.Timedelta(seconds=ENTRY_DECISION_BAR_SECONDS),
    )
    exit_route = htf.get_model_native_multi_tf_route_windows(
        features,
        decision_bar_start=decision,
        per_tf_seq_lens=lengths,
        route_timeframes=EXIT_MTF_CONTEXT_TIMEFRAMES,
        base_bar_duration=pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS),
    )

    assert tuple(entry) == ENTRY_MTF_CONTEXT_TIMEFRAMES
    assert tuple(exit_route) == EXIT_MTF_CONTEXT_TIMEFRAMES
    assert "M5" not in entry
    assert exit_route["M5"][-1, 0] == 398.0

    with np.testing.assert_raises_regex(
        RuntimeError,
        "MODEL_NATIVE_MTF_LOCAL_CLOCK_INVALID",
    ):
        htf.get_model_native_multi_tf_route_windows(
            features,
            decision_bar_start=decision,
            per_tf_seq_lens=lengths,
            route_timeframes=ENTRY_MTF_CONTEXT_TIMEFRAMES,
            base_bar_duration=pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS),
        )
    with np.testing.assert_raises_regex(
        RuntimeError,
        "MODEL_NATIVE_MTF_ROUTE_INVALID",
    ):
        htf.get_model_native_multi_tf_route_windows(
            features,
            decision_bar_start=decision,
            per_tf_seq_lens=lengths,
            route_timeframes=("M5", "M15", "H1", "H4"),
            base_bar_duration=pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS),
        )


def test_v4_cache_surface_excludes_every_open_trailing_resample_bucket(
    monkeypatch,
) -> None:
    source = _bars(419, seed=37)

    def finite_contract(
        frame: pd.DataFrame, *, timeframe: str, **_kwargs
    ) -> pd.DataFrame:
        assert timeframe in htf.MULTI_TF_RESAMPLE_RULES
        values = np.ones(
            (len(frame), htf.MULTI_TF_FEATURE_COUNT_V4),
            dtype=np.float32,
        )
        return pd.DataFrame(
            values,
            index=frame.index,
            columns=htf.MULTI_TF_PER_BAR_FEATURES_V4,
        )

    monkeypatch.setattr(htf, "compute_per_bar_features_v4", finite_contract)
    built = htf.build_multi_tf_per_bar_features_v4(
        source,
        v29_registry_constants=V29_TEST_REGISTRY_CONSTANTS,
    )

    assert source.index[-1] == pd.Timestamp("2021-01-05T10:50:00Z")
    assert built["M5"].index[-1] == pd.Timestamp("2021-01-05T10:50:00Z")
    assert built["M15"].index[-1] == pd.Timestamp("2021-01-05T10:30:00Z")
    assert built["H1"].index[-1] == pd.Timestamp("2021-01-05T09:00:00Z")
    assert built["H4"].index[-1] == pd.Timestamp("2021-01-05T04:00:00Z")
    assert built["D1"].index[-1] == pd.Timestamp("2021-01-04T00:00:00Z")


def test_v4_closed_geometry_floors_friday_h4_and_d1_to_real_labels() -> None:
    source_index = pd.date_range(
        "2026-07-20T00:00:00Z",
        "2026-07-24T20:55:00Z",
        freq="5min",
    )

    expected = htf.build_multi_tf_v4_closed_timestamp_indices(source_index)

    assert expected["M5"][-1] == pd.Timestamp("2026-07-24T20:55:00Z")
    assert expected["M15"][-1] == pd.Timestamp("2026-07-24T20:45:00Z")
    assert expected["H1"][-1] == pd.Timestamp("2026-07-24T20:00:00Z")
    assert expected["H4"][-1] == pd.Timestamp("2026-07-24T16:00:00Z")
    assert expected["D1"][-1] == pd.Timestamp("2026-07-23T00:00:00Z")


# ---------------------------------------------------------------------------
# V29 Phase A per-TF event additions (trend_ema GAP-1/2/3 + momentum G1/G2).
# ---------------------------------------------------------------------------


def test_v29_edge_helpers_fire_exactly_once_per_crossing() -> None:
    series = pd.Series(
        [np.nan, np.nan, -1.0, -0.5, 0.5, 1.0, 0.0, -2.0, 3.0, 3.0]
    )
    up = htf._cross_up_event(series).to_numpy(dtype=np.float64)
    down = htf._cross_down_event(series).to_numpy(dtype=np.float64)
    # One fire per crossing; a touch of exactly zero does not fire the
    # opposite side until the sign actually flips.
    assert np.array_equal(
        up,
        np.array(
            [np.nan, np.nan, np.nan, 0, 1, 0, 0, 0, 1, 0], dtype=np.float64
        ),
        equal_nan=True,
    )
    assert np.array_equal(
        down,
        np.array(
            [np.nan, np.nan, np.nan, 0, 0, 0, 0, 1, 0, 0], dtype=np.float64
        ),
        equal_nan=True,
    )


def test_v29_bars_since_event_counts_on_the_valid_suffix() -> None:
    event = np.array([False, False, False, True, False, False, True, False])
    valid = np.array([False, False, True, True, True, True, True, True])
    ages = htf._bars_since_event(event, valid)
    assert np.array_equal(
        ages,
        np.array([np.nan, np.nan, 0, 0, 1, 2, 0, 1], dtype=np.float64),
        equal_nan=True,
    )
    with np.testing.assert_raises_regex(
        RuntimeError,
        "HTF_V4_EVENT_AGE_VALIDITY_NOT_ONE_PREFIX",
    ):
        htf._bars_since_event(
            event,
            np.array([True, False, True, True, True, True, True, True]),
        )


def test_v29_trend_events_bit_identical_to_local_layer_formula() -> None:
    """The per-TF 50/200 fields must equal the local price-derived layer's
    exact formulas (entry_model_native_feature_layers_v1: ewm min_periods=span,
    spread = ema50 - ema200, cross = (spread>0)&(spread.shift(1)<=0))."""
    bars = _bars(4000, seed=23)
    matrix = _compute_v4(bars, timeframe="M5")
    close = bars["close"].astype(np.float64)
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
    spread = ema50 - ema200

    def check_cross_pair(gap: pd.Series, up_name: str, down_name: str) -> None:
        previous = gap.shift(1)
        valid = (gap.notna() & previous.notna()).to_numpy()
        expected_up = ((gap > 0) & (previous <= 0)).astype(np.float64)
        expected_down = ((gap < 0) & (previous >= 0)).astype(np.float64)
        observed_up = matrix[up_name].to_numpy(dtype=np.float64)
        observed_down = matrix[down_name].to_numpy(dtype=np.float64)
        assert np.array_equal(
            observed_up[valid], expected_up.to_numpy()[valid]
        ), up_name
        assert np.array_equal(
            observed_down[valid], expected_down.to_numpy()[valid]
        ), down_name
        assert not np.isfinite(observed_up[~valid]).any(), up_name
        assert not np.isfinite(observed_down[~valid]).any(), down_name
        assert observed_up[valid].sum() > 0, up_name
        assert observed_down[valid].sum() > 0, down_name

    check_cross_pair(spread, "ema50_200_cross_up", "ema50_200_cross_down")
    check_cross_pair(
        close - ema50, "price_x_ema50_cross_up", "price_x_ema50_cross_down"
    )
    check_cross_pair(
        close - ema200, "price_x_ema200_cross_up", "price_x_ema200_cross_down"
    )

    state_valid = spread.notna().to_numpy()
    expected_state = (spread > 0).astype(np.float64).to_numpy()
    observed_state = matrix["ema50_200_bull_state"].to_numpy(dtype=np.float64)
    assert np.array_equal(observed_state[state_valid], expected_state[state_valid])
    assert not np.isfinite(observed_state[~state_valid]).any()

    atr14 = htf._atr(
        bars["high"].astype(np.float64),
        bars["low"].astype(np.float64),
        close,
        14,
    )
    atr_safe = np.maximum(atr14, np.maximum(close * 1e-4, 1e-3))
    expected_spread_atr = (
        (spread / atr_safe).clip(-30.0, 30.0).to_numpy(dtype=np.float64)
    ).astype(np.float32)
    observed_spread_atr = matrix["ema50_200_spread_atr"].to_numpy(
        dtype=np.float32
    )
    assert np.array_equal(
        observed_spread_atr, expected_spread_atr, equal_nan=True
    )


def test_v29_rsi_threshold_events_use_raw_wilder_series() -> None:
    bars = _bars(4000, seed=7)
    matrix = _compute_v4(bars, timeframe="M5")
    close = bars["close"].astype(np.float64)
    rsi = htf._rsi(close, 14)
    rsi.iloc[:14] = np.nan
    previous = rsi.shift(1)
    valid = (rsi.notna() & previous.notna()).to_numpy()
    assert htf.RSI_WILDER_OVERSOLD == 30.0
    assert htf.RSI_WILDER_OVERBOUGHT == 70.0
    assert htf.RSI_WILDER_MIDLINE == 50.0
    assert htf.RSI_EXTREME_BAND_WIDTH == 20.0
    cases = (
        ("rsi_cross_up_30", (rsi > 30.0) & (previous <= 30.0)),
        ("rsi_cross_down_70", (rsi < 70.0) & (previous >= 70.0)),
        ("rsi_cross_up_50", (rsi > 50.0) & (previous <= 50.0)),
        ("rsi_cross_down_50", (rsi < 50.0) & (previous >= 50.0)),
    )
    for name, expected in cases:
        observed = matrix[name].to_numpy(dtype=np.float64)
        assert np.array_equal(
            observed[valid],
            expected.astype(np.float64).to_numpy()[valid],
        ), name
        assert not np.isfinite(observed[~valid]).any(), name
        assert observed[valid].sum() > 0, name

    # rsi_extreme_age_norm: zero exactly on |rsi-50| >= 20 rows, +1 per bar
    # since the last such row (or since the first valid row), capped at 500,
    # log1p/500 scale.
    n = len(bars)
    rsi_np = rsi.to_numpy(dtype=np.float64)
    rsi_valid = np.isfinite(rsi_np)
    extreme = rsi_valid & (np.abs(rsi_np - 50.0) >= 20.0)
    assert extreme.any()
    expected_age = np.full(n, np.nan, dtype=np.float64)
    anchor = None
    for i in range(n):
        if not rsi_valid[i]:
            continue
        if anchor is None or extreme[i]:
            anchor = i
        expected_age[i] = np.log1p(min(float(i - anchor), 500.0)) / np.log1p(
            500.0
        )
    observed_age = matrix["rsi_extreme_age_norm"].to_numpy(dtype=np.float64)
    assert np.array_equal(
        observed_age.astype(np.float32),
        expected_age.astype(np.float32),
        equal_nan=True,
    )
    assert np.all(observed_age[extreme] == 0.0)


def test_v29_mom20_flips_and_cross_age_semantics() -> None:
    bars = _bars(4000, seed=23)
    matrix = _compute_v4(bars, timeframe="M5")
    close = bars["close"].astype(np.float64)
    atr14 = htf._atr(
        bars["high"].astype(np.float64),
        bars["low"].astype(np.float64),
        close,
        14,
    )
    atr_safe = np.maximum(atr14, np.maximum(close * 1e-4, 1e-3))
    mom20 = ((close - close.shift(20)) / atr_safe).clip(-10.0, 10.0)
    previous = mom20.shift(1)
    valid = (mom20.notna() & previous.notna()).to_numpy()
    expected_up = ((mom20 > 0) & (previous <= 0)).astype(np.float64).to_numpy()
    expected_down = (
        ((mom20 < 0) & (previous >= 0)).astype(np.float64).to_numpy()
    )
    observed_up = matrix["mom20_sign_flip_up"].to_numpy(dtype=np.float64)
    observed_down = matrix["mom20_sign_flip_down"].to_numpy(dtype=np.float64)
    assert np.array_equal(observed_up[valid], expected_up[valid])
    assert np.array_equal(observed_down[valid], expected_down[valid])
    assert observed_up[valid].sum() > 0
    assert observed_down[valid].sum() > 0

    # ema50_200_cross_age_norm: zero exactly at state flips, +1 per bar in an
    # unchanged state, capped at 500, log1p/500 scale; zero at cross fires.
    state = matrix["ema50_200_bull_state"].to_numpy(dtype=np.float64)
    age_norm = matrix["ema50_200_cross_age_norm"].to_numpy(dtype=np.float64)
    finite = np.isfinite(state)
    expected = np.full(len(state), np.nan, dtype=np.float64)
    run = 0.0
    previous_state = None
    for i in np.where(finite)[0]:
        if previous_state is None or state[i] != previous_state:
            run = 0.0
        else:
            run += 1.0
        previous_state = state[i]
        expected[i] = np.log1p(min(run, 500.0)) / np.log1p(500.0)
    assert np.array_equal(
        age_norm.astype(np.float32), expected.astype(np.float32), equal_nan=True
    )
    fires = np.where(
        (matrix["ema50_200_cross_up"].to_numpy(dtype=np.float64) == 1.0)
        | (matrix["ema50_200_cross_down"].to_numpy(dtype=np.float64) == 1.0)
    )[0]
    assert len(fires) > 0
    assert np.all(age_norm[fires] == 0.0)


def test_v29_divergence_matches_confirmed_pivot_definition() -> None:
    """Spec mirror: divergence must use smc_v1's confirmed-pivot machinery
    (one pivot truth, SWING_LOOKBACK=3) and fire only at the confirmation
    bar of a new pivot pair — never earlier (no lookahead)."""
    from gx1.features.smc_v1 import (
        SWING_LOOKBACK,
        _detect_swing_pivots,
        _track_recent_swings,
    )

    bars = _bars(4000, seed=19)
    matrix = _compute_v4(bars, timeframe="M5")
    n = len(bars)
    high = bars["high"].to_numpy(dtype=np.float64)
    low = bars["low"].to_numpy(dtype=np.float64)
    close = bars["close"].astype(np.float64)
    rsi = htf._rsi(close, 14)
    rsi.iloc[:14] = np.nan
    rsi_np = rsi.to_numpy(dtype=np.float64)
    rsi_valid = np.isfinite(rsi_np)
    sh_mask, sl_mask = _detect_swing_pivots(high, low, SWING_LOOKBACK)
    last_sh, prev_sh, last_sl, prev_sl = _track_recent_swings(
        sh_mask, sl_mask, SWING_LOOKBACK
    )
    lh = np.clip(last_sh, 0, n - 1)
    ph = np.clip(prev_sh, 0, n - 1)
    ll = np.clip(last_sl, 0, n - 1)
    pl = np.clip(prev_sl, 0, n - 1)
    new_sh = last_sh != np.roll(last_sh, 1)
    new_sh[0] = False
    new_sl = last_sl != np.roll(last_sl, 1)
    new_sl[0] = False
    bear_defined = (prev_sh >= 0) & rsi_valid[ph]
    bull_defined = (prev_sl >= 0) & rsi_valid[pl]
    expected_bear = np.where(
        bear_defined,
        (
            new_sh
            & bear_defined
            & (high[lh] > high[ph])
            & (rsi_np[lh] < rsi_np[ph])
        ).astype(np.float64),
        np.nan,
    )
    expected_bull = np.where(
        bull_defined,
        (
            new_sl
            & bull_defined
            & (low[ll] < low[pl])
            & (rsi_np[ll] > rsi_np[pl])
        ).astype(np.float64),
        np.nan,
    )
    observed_bear = matrix["bear_divergence_event"].to_numpy(dtype=np.float64)
    observed_bull = matrix["bull_divergence_event"].to_numpy(dtype=np.float64)
    assert np.array_equal(observed_bear, expected_bear, equal_nan=True)
    assert np.array_equal(observed_bull, expected_bull, equal_nan=True)

    bear_fires = np.where(observed_bear == 1.0)[0]
    bull_fires = np.where(observed_bull == 1.0)[0]
    assert len(bear_fires) > 0
    assert len(bull_fires) > 0
    # Confirmation lag: the newly confirmed pivot of every firing pair sits
    # exactly SWING_LOOKBACK bars before the event bar.
    assert np.all(last_sh[bear_fires] == bear_fires - SWING_LOOKBACK)
    assert np.all(last_sl[bull_fires] == bull_fires - SWING_LOOKBACK)
    # Age resets to zero at every divergence event.
    observed_age = matrix["divergence_age_norm"].to_numpy(dtype=np.float64)
    assert np.all(observed_age[bear_fires] == 0.0)
    assert np.all(observed_age[bull_fires] == 0.0)


def _engineered_divergence_bars() -> tuple[pd.DataFrame, int]:
    """Flat warmup, impulsive rally to peak A (high RSI), sharp decline, weak
    grind to a marginally higher peak B (lower RSI): a textbook bearish
    divergence whose B pivot confirms exactly SWING_LOOKBACK bars later."""
    closes = [100.0]
    for i in range(250):
        closes.append(closes[-1] + (0.05 if i % 2 == 0 else -0.05))
    for _ in range(10):
        closes.append(closes[-1] + 2.0)
    peak_a_close = closes[-1]
    for _ in range(8):
        closes.append(closes[-1] - 1.5)
    while closes[-1] <= peak_a_close + 0.5:
        closes.append(closes[-1] + 0.4)
        closes.append(closes[-1] - 0.2)
    peak_b = len(closes) - 2
    for _ in range(6):
        closes.append(closes[-1] - 1.0)
    for i in range(10):
        closes.append(closes[-1] + (0.05 if i % 2 == 0 else -0.05))
    close = np.asarray(closes, dtype=np.float64)
    open_ = np.concatenate([[close[0]], close[:-1]])
    rising = close >= open_
    high = np.where(rising, close + 0.3, open_ + 0.05)
    low = np.where(rising, open_ - 0.05, close - 0.3)
    index = pd.date_range(
        "2021-01-04", periods=len(close), freq="5min", tz="UTC"
    )
    bars = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(len(close), 1.0),
        },
        index=index,
    )
    return bars, peak_b


def test_v29_divergence_fires_at_confirmation_and_never_early() -> None:
    from gx1.features.smc_v1 import SWING_LOOKBACK

    bars, peak_b = _engineered_divergence_bars()
    matrix = _compute_v4(bars, timeframe="M5")
    bear = matrix["bear_divergence_event"].fillna(0.0).to_numpy(
        dtype=np.float64
    )
    assert bear.sum() == 1.0
    assert bear[peak_b + SWING_LOOKBACK] == 1.0
    assert (
        matrix["bull_divergence_event"]
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
        .sum()
        == 0.0
    )
    observed_age = matrix["divergence_age_norm"].to_numpy(dtype=np.float64)
    assert observed_age[peak_b + SWING_LOOKBACK] == 0.0
    # Causality: without the SWING_LOOKBACK confirmation bars after peak B the
    # pivot is unconfirmed and no divergence may exist anywhere.
    truncated = _compute_v4(bars.iloc[: peak_b + SWING_LOOKBACK], timeframe="M5")
    assert (
        truncated["bear_divergence_event"]
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
        .sum()
        == 0.0
    )


def test_v29_event_columns_have_one_honest_warmup_prefix() -> None:
    bars = _bars(4000, seed=23)
    matrix = _compute_v4(bars, timeframe="M5")
    v29_names = (
        htf.MULTI_TF_V4_TREND_EVENT_FEATURES
        + htf.MULTI_TF_V4_MOMENTUM_EVENT_FEATURES
    )
    for name in v29_names:
        column = matrix[name].to_numpy(dtype=np.float64)
        finite = np.isfinite(column)
        assert finite.any(), name
        first = int(np.argmax(finite))
        assert finite[first:].all(), name
        assert not finite[:first].any(), name
    # Exact honest floors: local-layer min_periods (50/200), the file's own
    # 14-row RSI mask, the 20-bar momentum lag; +1 wherever the edge trigger
    # needs the previous bar.  Divergence floors are pivot/data-dependent and
    # are covered by the single-prefix loop above.
    expected_first_finite = {
        "ema50_200_spread_atr": 199,
        "ema50_200_bull_state": 199,
        "ema50_200_cross_up": 200,
        "ema50_200_cross_down": 200,
        "ema50_200_cross_age_norm": 199,
        "price_x_ema50_cross_up": 50,
        "price_x_ema50_cross_down": 50,
        "price_x_ema200_cross_up": 200,
        "price_x_ema200_cross_down": 200,
        "price_above_ema50_age_norm": 49,
        "price_above_ema200_age_norm": 199,
        "rsi_cross_up_30": 15,
        "rsi_cross_down_70": 15,
        "rsi_cross_up_50": 15,
        "rsi_cross_down_50": 15,
        "rsi_extreme_age_norm": 14,
        "mom20_sign_flip_up": 21,
        "mom20_sign_flip_down": 21,
    }
    for name, first_row in expected_first_finite.items():
        column = matrix[name].to_numpy(dtype=np.float64)
        assert int(np.argmax(np.isfinite(column))) == first_row, name
    warmup = htf.validate_causal_feature_matrix(
        matrix.to_numpy(dtype=np.float64),
        expected_width=htf.MULTI_TF_FEATURE_COUNT_V4,
        context="HTF_V4_V29_TEST",
    )
    assert warmup >= 200


def test_v4_closed_geometry_rejects_off_grid_source_timestamp() -> None:
    source_index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-07-24T20:50:00Z"),
            pd.Timestamp("2026-07-24T20:55:00.000001Z"),
        ]
    )

    with np.testing.assert_raises_regex(
        RuntimeError,
        "HTF_V4_SOURCE_TIMESTAMP_GEOMETRY_INVALID",
    ):
        htf.build_multi_tf_v4_closed_timestamp_indices(source_index)

