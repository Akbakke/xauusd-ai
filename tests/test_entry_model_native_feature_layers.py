from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_BASE_FIELDS
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_CTX_CAT_FIELDS, MODEL_NATIVE_CTX_CONT_FIELDS
from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_STRUCTURE_FEATURE_NAMES,
    FOUNDATION_STRUCTURE_SOURCE_FIELDS,
)
from gx1.features.entry_model_native_feature_layers_v1 import (
    MOMENTUM_EVENT_M5_LAYER_FEATURE_NAMES,
    SMC_LOCAL_EVENT_LAYER_FEATURE_NAMES,
    TRENDLINE_REGISTRY_M5_LAYER_FEATURE_NAMES,
    build_candle_primitive_derived_layer,
    build_chart_layer,
    build_momentum_event_m5_layer,
    build_price_derived_layer,
    build_smc_local_event_layer,
    build_trendline_registry_m5_layer,
)
from gx1.features import htf_features as htf
from gx1.features.entry_candle_primitives_v1 import (
    build_entry_candle_primitive_layer,
)
from gx1.features.technical_indicators_v1 import (
    ema50_200_spread_atr_block,
    wilder_rsi,
)
from tests.htf_v29_registry_test_support import synthetic_v29_registry_constants
from tests.volatility_squeeze_test_support import (
    make_volatility_squeeze_artifact_set,
)


_SQUEEZE_TEST_ARTIFACTS = None


@pytest.fixture(scope="module", autouse=True)
def _bind_squeeze_artifacts(tmp_path_factory):
    global _SQUEEZE_TEST_ARTIFACTS
    _SQUEEZE_TEST_ARTIFACTS = make_volatility_squeeze_artifact_set(
        tmp_path_factory.mktemp("squeeze-artifacts")
    )


def _valid_inputs(tmp_path: Path, *, rows: int = 240):
    names = [f"snap.{name}" for name in MODEL_NATIVE_BASE_FIELDS]
    names.extend(f"ctx_cont.{name}" for name in MODEL_NATIVE_CTX_CONT_FIELDS)
    names.extend(f"ctx_cat.{name}" for name in MODEL_NATIVE_CTX_CAT_FIELDS)
    row = np.arange(rows, dtype=np.float64)[:, None]
    column = np.arange(len(names), dtype=np.float64)[None, :]
    matrix = (0.5 + 0.4 * np.sin(row * 0.071 + column * 0.137)).astype(np.float32)
    for offset, name in enumerate(FOUNDATION_STRUCTURE_SOURCE_FIELDS):
        position = names.index(name)
        matrix[:, position] = 0.0
        matrix[11 + offset :: 37 + offset, position] = 1.0
    # EMA200 plus its two causal spread derivatives need 202 closed source
    # bars before the first emitted sample.  Warmup belongs to the source
    # history, not to zero-filled model rows.
    warmup_rows = 202
    source_rows = warmup_rows + rows
    times = pd.date_range("2026-01-01", periods=source_rows, freq="5min", tz="UTC")
    index = np.arange(source_rows, dtype=np.float64)
    mid = 2500.0 + index * 0.1 + np.sin(index * 0.11)
    open_ = mid - 0.05
    close = mid + 0.05 * np.sin(index * 0.17)
    source = pd.DataFrame(
        {
            "time": times,
            "mid": mid,
            "atr": 2.0 + 0.1 * np.cos(index * 0.09),
            "open": open_,
            "high": np.maximum(open_, close) + 0.2,
            "low": np.minimum(open_, close) - 0.2,
            "close": close,
        }
    )
    source_path = tmp_path / "canonical_source.parquet"
    source.to_parquet(source_path, index=False)
    samples = pd.DataFrame({"time": times[warmup_rows:]})
    return matrix, names, samples, source, source_path


def _sha256_matrix(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float32).tobytes(order="C")).hexdigest()


def _sha256_names(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def test_valid_full_contract_has_stable_names_order_and_bits(tmp_path: Path) -> None:
    matrix, names, samples, _source, source_path = _valid_inputs(tmp_path)
    chart_x, chart_names = build_chart_layer(matrix, names)
    price_x, price_names = build_price_derived_layer(samples, source_path)
    candle_x, candle_names = build_candle_primitive_derived_layer(samples, source_path)

    # The chart layer is a pure dispatcher of its two registered children;
    # the per-column value hashes below are bit-identical to the pre-removal
    # emissions of the same columns.
    assert chart_names == [
        *FOUNDATION_STRUCTURE_FEATURE_NAMES,
    ]

    expected = {
        "chart": (
            chart_x,
            chart_names,
            # Raw uncapped BOS/CHOCH ages with honest pre-event NaN prefixes.
            (240, 3),
            "4aef5b0f178a4c005c6af39281c5cc4ba9772f9e745ab8663e951a12cb8090b4",
            "47bd77bc92702ed39099d2bbebda4f3cd20206010b92f9e9aa8d276a5b91b726",
        ),
        "price": (
            price_x,
            price_names,
            # V30 (2026-08-13): package 1 added chart.local_kama_efficiency_30,
            # package 2 the three GAP-2/3 local age fields, and package 3 the
            # four price-vs-EMA cross events (19 fields); all hashes
            # re-measured on the unchanged source fixture.
            (240, 19),
            "bf143b30872d6513d47a9232ae25beea3f13946a98f8703d9e15250a5097e32a",
            "af300b1db50c88411851bbc82c69e78ece640acfa293f09c3fd9593d74e26d0e",
        ),
            "candle": (
                candle_x,
                candle_names,
                # Raw one-/two-bar geometry plus exact causal relation-state
                # durations; no named/thresholded candlestick patterns.
                (240, 26),
                "7cdc42e2c64f9ee8d21234ef1d2089d534a67edf67a5136cb594bd207618f596",
                "e099255c02a80471066fec98a98b3c071764e6d696f596fe69899bbdec0999b0",
        ),
    }
    for values, feature_names, shape, value_hash, name_hash in expected.values():
        assert values.shape == shape
        assert len(feature_names) == len(set(feature_names)) == shape[1]
        if values is chart_x:
            for column in range(values.shape[1]):
                finite = np.isfinite(values[:, column])
                assert finite.any()
                assert finite[int(np.argmax(finite)) :].all()
        else:
            assert np.isfinite(values).all()
        assert _sha256_matrix(values) == value_hash
        assert _sha256_names(feature_names) == name_hash


def test_chart_layer_rejects_missing_duplicate_and_nonfinite_sources(tmp_path: Path) -> None:
    matrix, names, _samples, _source, _source_path = _valid_inputs(tmp_path)
    missing_index = names.index("snap.smc_bos_up")
    with pytest.raises(RuntimeError, match="CHART_LAYER_SOURCE_FIELDS_MISSING"):
        build_chart_layer(np.delete(matrix, missing_index, axis=1), names[:missing_index] + names[missing_index + 1 :])

    bad = matrix.copy()
    bad[3, missing_index] = np.nan
    with pytest.raises(RuntimeError, match="CHART_LAYER_SOURCE_NONFINITE"):
        build_chart_layer(bad, names)

    with pytest.raises(RuntimeError, match="CHART_LAYER_DUPLICATE_FEATURE_NAMES"):
        build_chart_layer(
            np.concatenate([matrix, matrix[:, :1]], axis=1),
            names + [names[0]],
        )


def test_price_layer_rejects_missing_or_invalid_source_evidence(tmp_path: Path) -> None:
    _matrix, _names, samples, source, source_path = _valid_inputs(tmp_path)

    expected, expected_names = build_price_derived_layer(samples, source_path)
    missing_atr = tmp_path / "missing_atr.parquet"
    source.drop(columns=["atr"]).to_parquet(missing_atr, index=False)
    observed, observed_names = build_price_derived_layer(samples, missing_atr)
    assert observed_names == expected_names
    np.testing.assert_array_equal(observed, expected)

    nonfinite = source.copy()
    nonfinite.loc[10, "atr"] = np.nan
    nonfinite_path = tmp_path / "nonfinite_atr.parquet"
    nonfinite.to_parquet(nonfinite_path, index=False)
    observed, _ = build_price_derived_layer(samples, nonfinite_path)
    np.testing.assert_array_equal(observed, expected)

    missing_high = tmp_path / "missing_high.parquet"
    source.drop(columns=["high"]).to_parquet(missing_high, index=False)
    with pytest.raises(RuntimeError, match="PRICE_DERIVED_SOURCE_OHLC_MISSING"):
        build_price_derived_layer(samples, missing_high)

    invalid_geometry = source.copy()
    invalid_geometry.loc[10, "high"] = invalid_geometry.loc[10, "low"] - 1.0
    invalid_geometry_path = tmp_path / "invalid_geometry.parquet"
    invalid_geometry.to_parquet(invalid_geometry_path, index=False)
    with pytest.raises(RuntimeError, match="WILDER_ATR_SOURCE_GEOMETRY_INVALID"):
        build_price_derived_layer(samples, invalid_geometry_path)

    duplicate = pd.concat([source, source.iloc[[4]]], ignore_index=True)
    duplicate_path = tmp_path / "duplicate_time.parquet"
    duplicate.to_parquet(duplicate_path, index=False)
    with pytest.raises(RuntimeError, match="PRICE_DERIVED_SOURCE_TIME_DUPLICATE"):
        build_price_derived_layer(samples, duplicate_path)

    cold_samples = pd.DataFrame({"time": source["time"].iloc[:200]})
    with pytest.raises(RuntimeError, match="PRICE_DERIVED_LOCAL_EMA_WARMUP_INCOMPLETE"):
        build_price_derived_layer(cold_samples, tmp_path / "canonical_source.parquet")

    bad_samples = pd.DataFrame({"time": ["not-a-timestamp"]})
    with pytest.raises(RuntimeError, match="PRICE_DERIVED_SAMPLE_TIME_INVALID"):
        build_price_derived_layer(bad_samples, tmp_path / "canonical_source.parquet")

    gap_samples = pd.concat(
        [samples, pd.DataFrame({"time": [pd.Timestamp("2030-01-01", tz="UTC")]})],
        ignore_index=True,
    )
    with pytest.raises(RuntimeError, match="PRICE_DERIVED_SOURCE_ROW_GAP"):
        build_price_derived_layer(gap_samples, tmp_path / "canonical_source.parquet")


def test_price_layer_uses_exact_close_and_has_no_mid_fallback(tmp_path: Path) -> None:
    _matrix, _names, samples, source, source_path = _valid_inputs(tmp_path)
    expected, expected_names = build_price_derived_layer(samples, source_path)

    changed_mid = source.copy()
    changed_mid["mid"] = changed_mid["mid"] * 7.0 + 123.0
    changed_mid_path = tmp_path / "changed_mid.parquet"
    changed_mid.to_parquet(changed_mid_path, index=False)
    observed, observed_names = build_price_derived_layer(samples, changed_mid_path)
    assert observed_names == expected_names
    np.testing.assert_array_equal(observed, expected)

    mid_only_path = tmp_path / "mid_only.parquet"
    source.drop(columns=["close"]).to_parquet(mid_only_path, index=False)
    with pytest.raises(RuntimeError, match="PRICE_DERIVED_SOURCE_PRICE_MISSING"):
        build_price_derived_layer(samples, mid_only_path)


def test_candlestick_layer_rejects_bad_ohlc_geometry(tmp_path: Path) -> None:
    _matrix, _names, samples, source, _source_path = _valid_inputs(tmp_path)
    invalid_ohlc = source.copy()
    invalid_ohlc.loc[8, "high"] = invalid_ohlc.loc[8, "low"] - 1.0
    invalid_path = tmp_path / "invalid_ohlc.parquet"
    invalid_ohlc.to_parquet(invalid_path, index=False)
    with pytest.raises(RuntimeError, match="CANDLE_PRIMITIVE_DERIVED_SOURCE_OHLC_GEOMETRY_INVALID"):
        build_candle_primitive_derived_layer(samples, invalid_path)


@pytest.mark.parametrize("freq", ["1min", "5min"])
def test_local_m1_and_m5_candle_routes_match_the_owner_exactly(
    tmp_path: Path,
    freq: str,
) -> None:
    source = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=9, freq=freq, tz="UTC"),
            "open": [11.0, 9.5, 11.0, 11.0, 13.0, 14.0, 12.5, 13.2, 13.0],
            "high": [12.0, 12.5, 12.0, 12.0, 14.0, 15.0, 13.5, 13.5, 13.0],
            "low": [9.0, 9.0, 9.5, 9.5, 11.8, 12.5, 11.0, 12.0, 13.0],
            "close": [10.0, 11.5, 10.5, 10.5, 12.0, 13.0, 13.0, 12.3, 13.0],
        }
    )
    source_path = tmp_path / f"candle_{freq}.parquet"
    source.to_parquet(source_path, index=False)
    sample = source.loc[1:, ["time"]]

    observed, observed_names = build_candle_primitive_derived_layer(
        sample,
        source_path,
    )
    expected, expected_names = build_entry_candle_primitive_layer(source)
    assert observed_names == expected_names
    np.testing.assert_array_equal(observed, expected[1:])


@pytest.mark.parametrize("frequency", ("1min", "5min"))
def test_native_smc_and_momentum_layers_use_full_exact_source_history(
    tmp_path: Path,
    frequency: str,
) -> None:
    from gx1.features.htf_features import (
        compute_v29_momentum_event_block_from_ohlc,
    )
    from gx1.features.smc_v1 import compute_smc_features

    _matrix, _names, samples, source, _source_path = _valid_inputs(tmp_path)
    source["time"] = pd.date_range(
        "2026-01-01", periods=len(source), freq=frequency, tz="UTC"
    )
    samples = pd.DataFrame({"time": source["time"].iloc[-len(samples):]})
    source_path = tmp_path / f"smc_{frequency}.parquet"
    source.to_parquet(source_path, index=False)
    source_index = pd.DatetimeIndex(source["time"])
    indexed = source.set_index(source_index)

    smc_values, smc_names = build_smc_local_event_layer(samples, source_path)
    assert tuple(smc_names) == SMC_LOCAL_EVENT_LAYER_FEATURE_NAMES
    expected_smc = compute_smc_features(
        indexed[["high", "low", "close", "atr"]],
        include_v30_additions=True,
    ).loc[source_index[-len(samples):], list(smc_names)]
    np.testing.assert_array_equal(
        smc_values,
        expected_smc.to_numpy(dtype=np.float32),
    )
    split = len(samples) // 2
    left, left_names = build_smc_local_event_layer(
        samples.iloc[:split], source_path
    )
    right, right_names = build_smc_local_event_layer(
        samples.iloc[split:], source_path
    )
    assert left_names == right_names == smc_names
    np.testing.assert_array_equal(np.vstack((left, right)), smc_values)

    momentum_values, momentum_names = build_momentum_event_m5_layer(
        samples,
        source_path,
    )
    assert tuple(momentum_names) == MOMENTUM_EVENT_M5_LAYER_FEATURE_NAMES
    expected_momentum = compute_v29_momentum_event_block_from_ohlc(
        indexed[["high", "low", "close"]],
        include_v30_primitives=True,
    ).loc[source_index[-len(samples):], list(momentum_names)]
    np.testing.assert_array_equal(
        momentum_values,
        expected_momentum.to_numpy(dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("timeframe", "frequency"),
    (("M1", "1min"), ("M5", "5min")),
)
def test_native_momentum_builder_has_exact_clock_parity_warmup_and_chunks(
    tmp_path: Path,
    timeframe: str,
    frequency: str,
) -> None:
    _matrix, _names, samples, source, _source_path = _valid_inputs(
        tmp_path,
        rows=300,
    )
    source["time"] = pd.date_range(
        "2026-01-01",
        periods=len(source),
        freq=frequency,
        tz="UTC",
    )
    samples = pd.DataFrame({"time": source["time"].iloc[-300:]})
    source_path = tmp_path / f"native_{timeframe.lower()}.parquet"
    source.to_parquet(source_path, index=False)

    observed, names = build_momentum_event_m5_layer(samples, source_path)
    raw, raw_names = build_momentum_event_m5_layer(
        None,
        source_path,
        raw_frame=True,
    )
    assert tuple(names) == tuple(raw_names) == MOMENTUM_EVENT_M5_LAYER_FEATURE_NAMES
    primitive_names = tuple(htf.LOCAL_MOMENTUM_V30_PRIMITIVE_FEATURES)
    primitive_positions = [names.index(name) for name in primitive_names]

    source_index = pd.DatetimeIndex(source["time"])
    ohlc = source.set_index(source_index)[["high", "low", "close"]]
    expected_local = htf.compute_v29_momentum_event_block_from_ohlc(
        ohlc,
        include_v30_primitives=True,
    )
    np.testing.assert_array_equal(
        observed[:, primitive_positions],
        expected_local.loc[samples["time"], list(primitive_names)].to_numpy(
            dtype=np.float32
        ),
    )

    if timeframe == "M5":
        per_tf_source = source.set_index(source_index)[
            ["open", "high", "low", "close"]
        ].copy()
        per_tf_source["volume"] = np.full(len(source), 100, dtype=np.int64)
        per_tf = htf.compute_per_bar_features_v4(
            per_tf_source,
            timeframe="M5",
            v29_registry_constants=synthetic_v29_registry_constants(),
            volatility_squeeze_artifacts=_SQUEEZE_TEST_ARTIFACTS,
        )
        np.testing.assert_array_equal(
            observed[:, primitive_positions],
            per_tf.loc[samples["time"], list(primitive_names)].to_numpy(
                dtype=np.float32
            ),
        )

    expected_first_finite = {
        "rsi14_centered": 14,
        "rsi14_delta_5": 19,
        "mom_5_atr": 13,
        "mom_20_atr": 20,
    }
    for name, expected_first in expected_first_finite.items():
        finite = np.isfinite(raw[name].to_numpy(dtype=np.float64))
        assert int(np.argmax(finite)) == expected_first
        assert not finite[:expected_first].any()
        assert finite[expected_first:].all()

    rsi = wilder_rsi(ohlc["close"], 14)
    np.testing.assert_array_equal(
        raw["rsi14_delta_5"].to_numpy(dtype=np.float64),
        (rsi - rsi.shift(5)).to_numpy(dtype=np.float64),
    )

    split = len(samples) // 2
    left, left_names = build_momentum_event_m5_layer(
        samples.iloc[:split],
        source_path,
    )
    right, right_names = build_momentum_event_m5_layer(
        samples.iloc[split:],
        source_path,
    )
    assert left_names == right_names == names
    np.testing.assert_array_equal(np.vstack([left, right]), observed)

    changed = source.copy()
    changed.loc[changed.index[-1], "close"] += 5.0
    changed.loc[changed.index[-1], "high"] = max(
        changed.loc[changed.index[-1], "high"],
        changed.loc[changed.index[-1], "close"] + 0.2,
    )
    changed_path = tmp_path / f"native_{timeframe.lower()}_changed.parquet"
    changed.to_parquet(changed_path, index=False)
    changed_values, changed_names = build_momentum_event_m5_layer(
        samples,
        changed_path,
    )
    assert changed_names == names
    np.testing.assert_array_equal(
        changed_values[:-1, primitive_positions],
        observed[:-1, primitive_positions],
    )
    assert np.all(
        changed_values[-1, primitive_positions]
        != observed[-1, primitive_positions]
    )


@pytest.mark.parametrize(
    ("timeframe", "frequency"),
    (("M1", "1min"), ("M5", "5min")),
)
def test_native_price_builder_and_per_tf_share_exact_ema_spread_owner(
    tmp_path: Path,
    timeframe: str,
    frequency: str,
) -> None:
    _matrix, _names, _samples, source, _source_path = _valid_inputs(
        tmp_path,
        rows=300,
    )
    source["time"] = pd.date_range(
        "2026-01-01",
        periods=len(source),
        freq=frequency,
        tz="UTC",
    )
    samples = pd.DataFrame({"time": source["time"].iloc[-300:]})
    source_path = tmp_path / f"ema_{timeframe.lower()}.parquet"
    source.to_parquet(source_path, index=False)

    local, local_names = build_price_derived_layer(samples, source_path)
    local_position = local_names.index("chart.local_ema50_200_spread_atr")
    source_index = pd.DatetimeIndex(source["time"])
    indexed = source.set_index(source_index)
    shared = ema50_200_spread_atr_block(
        indexed["high"].astype(np.float64),
        indexed["low"].astype(np.float64),
        indexed["close"].astype(np.float64),
    )
    expected_storage = shared.loc[
        samples["time"],
        "spread_atr",
    ].to_numpy(dtype=np.float32)
    np.testing.assert_array_equal(local[:, local_position], expected_storage)

    if timeframe == "M5":
        per_tf_source = indexed[["open", "high", "low", "close"]].copy()
        per_tf_source["volume"] = np.full(len(source), 100, dtype=np.int64)
        per_tf = htf.compute_per_bar_features_v4(
            per_tf_source,
            timeframe="M5",
            v29_registry_constants=synthetic_v29_registry_constants(),
            volatility_squeeze_artifacts=_SQUEEZE_TEST_ARTIFACTS,
        )
        np.testing.assert_array_equal(
            local[:, local_position],
            per_tf.loc[
                samples["time"],
                "ema50_200_spread_atr",
            ].to_numpy(dtype=np.float32),
        )


def test_native_trendline_builder_preserves_explicit_clock_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gx1.features.entry_model_native_feature_layers_v1 as layers

    _matrix, _names, _samples, source, source_path = _valid_inputs(tmp_path)
    observed: list[str] = []

    def _capture_timeframe(registry_source, **kwargs):
        observed.append(kwargs["timeframe"])
        return (
            pd.DataFrame(
                0.0,
                index=registry_source.index,
                columns=layers.TRENDLINE_REGISTRY_FEATURE_NAMES_V1,
            ),
            object(),
        )

    monkeypatch.setattr(
        layers,
        "compute_trendline_registry_features_v1",
        _capture_timeframe,
    )
    for timeframe in ("M1", "M5"):
        frame, names = build_trendline_registry_m5_layer(
            None,
            source_path,
            timeframe=timeframe,
            band_atr=0.25,
            seq_len=480 if timeframe == "M1" else 96,
            identity_expiry_bars=480 if timeframe == "M1" else 96,
            raw_frame=True,
        )
        assert tuple(names) == TRENDLINE_REGISTRY_M5_LAYER_FEATURE_NAMES
        assert frame.index.equals(pd.DatetimeIndex(source["time"]))

    assert observed == ["M1", "M5"]
