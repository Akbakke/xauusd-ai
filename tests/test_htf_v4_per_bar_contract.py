"""V4-only per-bar multi-timeframe feature, routing, and clock contract."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_DECISION_BAR_SECONDS,
    ENTRY_MTF_CONTEXT_TIMEFRAMES,
    EXIT_DECISION_BAR_SECONDS,
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.features import htf_features as htf


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


def test_v4_is_one_exact_111_field_contract() -> None:
    assert htf.MULTI_TF_FEATURE_COUNT_V4 == 111
    assert htf.MULTI_TF_V4_GROUP_A_BASE_FEATURES == (
        EXPECTED_V4_GROUP_A_BASE_FEATURES
    )
    assert (
        htf.MULTI_TF_PER_BAR_FEATURES_V4[:25]
        == EXPECTED_V4_GROUP_A_BASE_FEATURES
    )
    assert len(set(htf.MULTI_TF_PER_BAR_FEATURES_V4)) == 111


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
        htf.compute_per_bar_features_v4(
            bars.drop(columns="volume"), timeframe="M5"
        )

    off_grid = bars.copy()
    off_grid.index = off_grid.index + pd.Timedelta(minutes=1)
    with np.testing.assert_raises_regex(
        RuntimeError,
        "HTF_INPUT_FAIL|HTF_V4_SOURCE_TIMESTAMP_GEOMETRY_INVALID",
    ):
        htf.build_multi_tf_per_bar_features_v4(off_grid)


def test_v4_routes_every_field_to_all_eight_specialists() -> None:
    from gx1.features.entry_specialist_feature_groups_v1 import (
        MODEL_NATIVE_TRAINING_SPECIALISTS,
        require_multi_tf_specialist_routing_v4,
    )

    routing = require_multi_tf_specialist_routing_v4(
        htf.MULTI_TF_PER_BAR_FEATURES_V4
    )
    assert htf.MULTI_TF_FEATURE_COUNT_V4 == 111
    assert tuple(routing) == MODEL_NATIVE_TRAINING_SPECIALISTS
    assert {name: len(indices) for name, indices in routing.items()} == {
        "structure_swing_encoder": 5,
        "smc_liquidity_encoder": 11,
        "trend_ema_encoder": 10,
        "vol_compression_encoder": 2,
        "momentum_flow_encoder": 4,
        "session_regime_encoder": 5,
        "chart_geometry_encoder": 10,
        "price_action_candle_encoder": 64,
    }
    flattened = [index for indices in routing.values() for index in indices]
    assert sorted(flattened) == list(range(htf.MULTI_TF_FEATURE_COUNT_V4))
    assert "vwap_session_dist_atr" not in htf.MULTI_TF_PER_BAR_FEATURES_V4
    assert "vwap_session_slope_atr" not in htf.MULTI_TF_PER_BAR_FEATURES_V4
    assert "vwap_local_cycle_dist_atr" in htf.MULTI_TF_PER_BAR_FEATURES_V4


def test_v4_smc_and_geometry_are_causal_and_have_one_warmup_prefix() -> None:
    bars = _bars(4000, seed=23)
    original = htf.compute_per_bar_features_v4(bars, timeframe="M5")
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
    future_changed = htf.compute_per_bar_features_v4(changed, timeframe="M5")
    assert np.array_equal(
        original.iloc[:cutoff].to_numpy(),
        future_changed.iloc[:cutoff].to_numpy(),
        equal_nan=True,
    )

    # A valid trend can place the latest confirmed low above an older
    # confirmed high. That remains available structure evidence.
    trending = htf.compute_per_bar_features_v4(_bars(4000, seed=0), timeframe="M5")
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
    matrix = htf.compute_per_bar_features_v4(_bars(4000, seed=31), timeframe="M5")
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

    def finite_contract(frame: pd.DataFrame, *, timeframe: str) -> pd.DataFrame:
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
    built = htf.build_multi_tf_per_bar_features_v4(source)

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

