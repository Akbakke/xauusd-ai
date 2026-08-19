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
    LOCAL_EMA_SLOPE_LOOKBACK_BARS,
    MOMENTUM_EVENT_M5_LAYER_FEATURE_NAMES,
    PRICE_DERIVED_CAUSAL_WARMUP_ROWS,
    PRICE_DERIVED_FEATURE_NAMES,
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
from gx1.features.model_native_market_context_v1 import (
    derive_model_native_atr_spread_bps,
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
    # Warmup belongs to the source history, not to zero-filled model rows, and
    # the floor is read from the producing owner instead of restated here
    # (2026-08-19: the local EMA slope repair moved it 202 -> 204, because
    # ``ema200[t] - ema200[t-5]`` is first finite at the classic EMA200 seed
    # row 199 plus the 5-bar lookback).
    warmup_rows = PRICE_DERIVED_CAUSAL_WARMUP_ROWS
    source_rows = warmup_rows + rows
    times = pd.date_range("2026-01-01", periods=source_rows, freq="5min", tz="UTC")
    # The generator is anchored to the source index at which the emitted
    # window began under the previous 202-row floor, so raising the floor adds
    # history at the FRONT and leaves the emitted 240 rows' OHLC bit-identical.
    # Every non-price layer's pinned hash below is therefore evidence that this
    # wave did not perturb it, rather than a number that had to be refreshed.
    index = np.arange(source_rows, dtype=np.float64) - float(warmup_rows - 202)
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
            # four price-vs-EMA cross events.
            # 2026-08-19 fidelity repair, first pass: six columns changed unit
            # and/or formula in place (four price-relative `_bps` -> `_atr`, and
            # the two `ema*_slope_bps` columns, which the classic EMA recursion
            # made an exact multiple of the price gap, -> the genuine 5-bar EMA
            # change over the same positive Wilder ATR) at an unchanged width.
            # 2026-08-19 second pass: chart.local_ema50_200_spread_bps RETIRED,
            # so the width narrows by one.  MEASURED attribution — re-inserting
            # that one column at its former position 0, recomputed from the
            # shared block owner as `spread / close.abs() * 1e4`, reproduces the
            # pre-removal value hash
            # d00c709a58fbf60ce8284427456d1466953d14a690c635499282c4692ecca587
            # exactly, so every surviving column is bit-identical and this is a
            # narrower surface rather than a changed one.  The width itself is
            # never restated: it is read from the owner tuple below.
            (240, len(PRICE_DERIVED_FEATURE_NAMES)),
            "c2e41f45317ddb571fac9c722984da81c8472d1751596d9f6dcb8286698f2dfe",
            "e2dd5d7119a90e593815f46d205d8bbfb0315ec5df9dfc587503492bfb9e6752",
        ),
        "candle": (
            candle_x,
            candle_names,
            # Raw one-/two-bar geometry plus exact causal relation-state
            # durations; no named/thresholded candlestick patterns.
            # 2026-08-15: candle.raw_zero_range_flag retired (constant
            # post-warmup on H4/D1, hence a hard liveness RED with no
            # scaleable exemption).
            # 2026-08-18 (V30 wave 2): candle.raw_close_location,
            # candle.raw_range_change_local_geometry and the two
            # candle.raw_*_rejection_depth_local_geometry columns retired,
            # each an exact function of columns that stay in this owner.
            # Both hashes re-derived on the unchanged source fixture AFTER
            # proving every surviving column is bit-identical to the
            # pre-removal emission of the same column, so this is a narrower
            # surface, not a changed one.
            # 2026-08-19: this owner is UNTOUCHED by the local-EMA repair; the
            # value hash moved only because that repair raised the layer's
            # declared warmup floor 202 -> 204, so the fixture's source carries
            # two more leading bars.  MEASURED attribution: rebuilding this
            # layer on the 202-row fixture reproduces the previous hash
            # 95882cf125b3152876ab3f2b7b6788af3fc1789b8bd618c895781799756d0955
            # exactly, and exactly one column differs between the two —
            # candle.raw_observed_body_direction_duration_bars, uniformly +2
            # bars, i.e. the two extra prefix rows counted by a run-length
            # field.  Every other column is bit-identical and the name hash is
            # unchanged.
            (240, 21),
            "52288c6502211a316f5c67661a6136e6fd1cc26c1e20c225d648508714756058",
            "9a869c450465859c43e7eab1bfca8a6bd7f9a3fc05e636df36a29d6c29ff26a7",
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


# ---------------------------------------------------------------------------
# 2026-08-19 fidelity repair of the local EMA layer.
#
# Two defects were repaired in gx1/features/entry_model_native_feature_layers_v1.py:
#   D1 (formula) `chart.local_ema50_slope_bps` / `chart.local_ema200_slope_bps`
#       were `ema.diff()/close*1e4`, which the classic EMA recursion makes an
#       exact positive multiple of the price-vs-EMA gap already in the layer.
#   D2 (unit)    the four price-relative fields divided by `close`, so their
#       dispersion tracked realised volatility instead of the trend state.
# Both are now ATR-normalized against the ONE shared positive Wilder-14
# denominator, and the slopes are genuine k-bar EMA changes.
# ---------------------------------------------------------------------------


_REPAIRED_ATR_FIELDS_TO_PER_TF_OWNER = {
    "chart.local_price_vs_ema50_atr": "ema50_dist_atr",
    "chart.local_price_vs_ema200_atr": "ema200_dist_atr",
    "chart.local_ema50_slope_atr": "ema50_slope_atr",
    "chart.local_ema200_slope_atr": "ema200_slope_atr",
}


def test_repaired_local_atr_fields_are_bit_identical_to_the_per_tf_owner(
    tmp_path: Path,
) -> None:
    """The repaired columns ARE the per-TF owner's formula, not a copy of it.

    ``htf_features`` has emitted ``ema{50,200}_dist_atr`` and
    ``ema{50,200}_slope_atr`` on every per-TF clock since V4; the local M5/M1
    clock carried neither a genuine EMA slope nor an ATR-normalized price
    distance.  Asserting bit-equality against that owner on the same native M5
    frame binds the lookback (5 closed bars), the denominator (the shared
    block's ``atr14_positive``) and the numerator at once: no literal restated
    in the local owner can drift from the convention it claims to adopt.
    """

    _matrix, _names, _samples, source, _source_path = _valid_inputs(
        tmp_path,
        rows=300,
    )
    source["time"] = pd.date_range(
        "2026-01-01",
        periods=len(source),
        freq="5min",
        tz="UTC",
    )
    samples = pd.DataFrame({"time": source["time"].iloc[-300:]})
    source_path = tmp_path / "ema_slope_m5.parquet"
    source.to_parquet(source_path, index=False)

    local, local_names = build_price_derived_layer(samples, source_path)
    source_index = pd.DatetimeIndex(source["time"])
    indexed = source.set_index(source_index)
    per_tf_source = indexed[["open", "high", "low", "close"]].copy()
    per_tf_source["volume"] = np.full(len(source), 100, dtype=np.int64)
    per_tf = htf.compute_per_bar_features_v4(
        per_tf_source,
        timeframe="M5",
        v29_registry_constants=synthetic_v29_registry_constants(),
        volatility_squeeze_artifacts=_SQUEEZE_TEST_ARTIFACTS,
    )
    for local_name, per_tf_name in _REPAIRED_ATR_FIELDS_TO_PER_TF_OWNER.items():
        np.testing.assert_array_equal(
            local[:, local_names.index(local_name)],
            per_tf.loc[samples["time"], per_tf_name].to_numpy(dtype=np.float32),
            err_msg=f"{local_name} != per-TF {per_tf_name}",
        )


def test_retired_bps_slope_was_an_exact_multiple_of_the_price_gap(
    tmp_path: Path,
) -> None:
    """Reproduce the retired formula and prove why it could not be a slope.

    This is the defect, executed rather than described.  For the classic
    recursion ``ema[t] = ema[t-1] + a*(c[t] - ema[t-1])`` with ``a = 2/(s+1)``:

        ema.diff()[t] = a*(c[t] - ema[t-1])
        c[t] - ema[t] = (1-a)*(c[t] - ema[t-1])
        => ema.diff()[t] === (a/(1-a)) * (c[t] - ema[t]) === (2/(s-1))*(gap)

    so the retired ``ema.diff()/close*1e4`` was ``(2/49)`` resp. ``(2/199)``
    times ``price_vs_ema{50,200}_bps``: a positive scalar multiple, which the
    positively homogeneous ``asinh((x-median)/IQR)`` input normalizer maps to a
    bit-identical column.  ``slope > 0`` was therefore exactly ``close > ema``,
    and a rising average with price pulled back below it was representationally
    impossible.  The identity is algebraic and holds on any price path, so this
    test proves the retirement was mandatory, not a preference.
    """

    _matrix, _names, samples, source, source_path = _valid_inputs(tmp_path)
    source_index = pd.DatetimeIndex(source["time"])
    indexed = source.set_index(source_index)
    block = ema50_200_spread_atr_block(
        indexed["high"].astype(np.float64),
        indexed["low"].astype(np.float64),
        indexed["close"].astype(np.float64),
    )
    close = indexed["close"].astype(np.float64)
    for span in (50, 200):
        ema = block[f"ema{span}"]
        retired_slope_bps = (ema.diff() / close.abs() * 1e4).loc[samples["time"]]
        gap_bps = ((close - ema) / close.abs() * 1e4).loc[samples["time"]]
        ratio = 2.0 / (span - 1.0)
        np.testing.assert_allclose(
            retired_slope_bps.to_numpy(dtype=np.float64),
            ratio * gap_bps.to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        )

    # The repaired field is NOT that quantity.  The discriminator is chosen to
    # be data-independent: under the retired formula the ratio
    # slope/gap is the CONSTANT 2/(span-1) on every row of every price path, so
    # a ratio that varies at all is proof the field is no longer a rescaling of
    # the gap.  (The stronger observable — a rising average with price pulled
    # back below it — is a property of real tapes and is deliberately NOT
    # asserted on this synthetic monotone fixture; rule 2c.)
    local, local_names = build_price_derived_layer(samples, source_path)
    slope = local[:, local_names.index("chart.local_ema50_slope_atr")].astype(
        np.float64
    )
    gap = local[:, local_names.index("chart.local_price_vs_ema50_atr")].astype(
        np.float64
    )
    assert np.all(gap != 0.0)
    ratio = slope / gap
    assert not np.allclose(ratio, 2.0 / 49.0)
    assert not np.allclose(ratio, float(ratio[0]))


def test_repaired_spread_derivatives_are_raw_difference_over_current_atr(
    tmp_path: Path,
) -> None:
    """delta/accel adopt the repository's k-bar change convention exactly.

    ``mom_5_atr``, ``ema20_slope_atr``, ``_v1_tema20_change_3_atr`` and
    ``_v1_kama30_change_5_atr`` all divide a RAW difference by the CURRENT
    closed bar's positive ATR.  Differencing an already ATR-normalized series
    instead would fold the ATR's own bar-to-bar change into a field named for
    the spread, so the two are asserted to be different quantities here.
    """

    _matrix, _names, samples, source, source_path = _valid_inputs(tmp_path)
    source_index = pd.DatetimeIndex(source["time"])
    indexed = source.set_index(source_index)
    block = ema50_200_spread_atr_block(
        indexed["high"].astype(np.float64),
        indexed["low"].astype(np.float64),
        indexed["close"].astype(np.float64),
    )
    spread = block["spread"]
    atr14_positive = block["atr14_positive"]
    expected_delta = (spread.diff() / atr14_positive).loc[samples["time"]]
    expected_accel = (spread.diff().diff() / atr14_positive).loc[samples["time"]]

    local, local_names = build_price_derived_layer(samples, source_path)
    np.testing.assert_array_equal(
        local[:, local_names.index("chart.local_ema50_200_spread_delta_atr")],
        expected_delta.to_numpy(dtype=np.float32),
    )
    np.testing.assert_array_equal(
        local[:, local_names.index("chart.local_ema50_200_spread_accel_atr")],
        expected_accel.to_numpy(dtype=np.float32),
    )
    differenced_normalized = (
        block["spread_atr"].diff().loc[samples["time"]].to_numpy(dtype=np.float64)
    )
    assert not np.allclose(
        expected_delta.to_numpy(dtype=np.float64),
        differenced_normalized,
    )


def test_layer_has_no_price_relative_field_and_bps_stays_exactly_recoverable(
    tmp_path: Path,
) -> None:
    """No column divides by a price level, and the retired bps reading survives.

    2026-08-19, second pass of the same fidelity wave.  The first pass kept
    ``local_ema50_200_spread_bps`` on ONE ground: it was the exact unit
    conversion anchor, because ``ctx_cont.atr_bps`` then divided its true range
    by the bar MIDPOINT while this layer divides by ``close``.  That premise is
    gone -- ``model_native_market_context_v1`` now emits
    ``wilder_atr(high, low, close, 14) / close * 1e4`` from the same one Wilder
    owner that ``ema50_200_spread_atr_block`` uses here -- so

        spread_atr * atr_bps
            == (spread / atr14) * (atr14 / close * 1e4)
            == spread / close * 1e4

    is the retired field's exact former definition, from two inputs the model
    already reads.  Rule 4 is discharged by that algebra and by
    ``spread_atr`` itself, which is the SAME numerator and keeps the full signed
    magnitude of the EMA50-200 spread.

    Both owners are EXECUTED here rather than restated (rule 13): a consumer
    that re-derived ``atr14/close*1e4`` inline would keep passing after the ctx
    owner changed its denominator, which is exactly the failure this test
    exists to catch.  The tolerance is the ``rtol=2e-6`` this file already uses
    where one factor is float32 storage; it is roughly four orders of magnitude
    tighter than the 1.27e-02 max relative split the retired midpoint
    denominator produced, so a denominator regression fails loudly.

    The REASON for the retirement -- the field's IQR width grew 1.32x between
    the last and first third of the real tape while ``spread_atr`` sat at 1.00
    -- is a property of a real price history and is deliberately not asserted
    on this synthetic fixture (rule 2c).  What is asserted here is the
    invariant: nothing in this layer may be price-relative again.
    """

    price_relative = tuple(
        name for name in PRICE_DERIVED_FEATURE_NAMES if name.endswith("_bps")
    )
    assert price_relative == ()

    _matrix, _names, samples, source, source_path = _valid_inputs(tmp_path)
    local, local_names = build_price_derived_layer(samples, source_path)
    spread_atr = local[
        :, local_names.index("chart.local_ema50_200_spread_atr")
    ].astype(np.float64)
    assert np.all(np.abs(spread_atr) > 0.0)

    source_index = pd.DatetimeIndex(source["time"])
    indexed = source.set_index(source_index)
    # The ctx owner returns the observed quoted spread alongside the ATR and
    # therefore requires quote columns.  They are declared equal to ``close``
    # so no spread magnitude is invented (rule 2b); only ``atr_bps`` is read.
    ctx_frame = indexed.copy()
    ctx_frame["bid_close"] = ctx_frame["close"]
    ctx_frame["ask_close"] = ctx_frame["close"]
    atr_bps = derive_model_native_atr_spread_bps(ctx_frame)["atr_bps"]
    recovered = spread_atr * atr_bps.loc[samples["time"]].to_numpy(
        dtype=np.float64
    )

    block = ema50_200_spread_atr_block(
        indexed["high"].astype(np.float64),
        indexed["low"].astype(np.float64),
        indexed["close"].astype(np.float64),
    )
    close = indexed["close"].astype(np.float64)
    retired_spread_bps = (
        (block["spread"] / close.abs() * 1e4)
        .loc[samples["time"]]
        .to_numpy(dtype=np.float64)
    )
    np.testing.assert_allclose(
        recovered,
        retired_spread_bps,
        rtol=2e-6,
        atol=0.0,
    )

    # Non-vacuity: the comparison must be sensitive to the denominator, not
    # merely to the field name.  The retired midpoint convention -- the exact
    # reason the anchor could not be dropped before -- is rejected here.
    midpoint = ((indexed["high"] + indexed["low"]) / 2.0).astype(np.float64)
    midpoint_atr_bps = (
        (block["atr14"] / midpoint * 1e4).loc[samples["time"]].to_numpy(
            dtype=np.float64
        )
    )
    assert not np.allclose(
        spread_atr * midpoint_atr_bps,
        retired_spread_bps,
        rtol=2e-6,
        atol=0.0,
    )


def test_price_layer_warmup_floor_is_exactly_the_ema200_slope_first_finite_row(
    tmp_path: Path,
) -> None:
    """One row below the declared floor fails; the floor itself passes.

    The floor is derived, not chosen: classic EMA200's first valid row (199)
    plus the shared 5-bar EMA-slope lookback.  Asserting both sides of the
    boundary keeps a future lookback or EMA-seed change from silently emitting
    a NaN prefix as model evidence.
    """

    assert PRICE_DERIVED_CAUSAL_WARMUP_ROWS == 199 + LOCAL_EMA_SLOPE_LOOKBACK_BARS

    _matrix, _names, _samples, source, source_path = _valid_inputs(tmp_path)
    times = pd.DatetimeIndex(source["time"])
    with pytest.raises(RuntimeError, match="PRICE_DERIVED_LOCAL_EMA_WARMUP_INCOMPLETE"):
        build_price_derived_layer(
            pd.DataFrame({"time": times[PRICE_DERIVED_CAUSAL_WARMUP_ROWS - 1 :]}),
            source_path,
        )
    values, names = build_price_derived_layer(
        pd.DataFrame({"time": times[PRICE_DERIVED_CAUSAL_WARMUP_ROWS:]}),
        source_path,
    )
    assert tuple(names) == PRICE_DERIVED_FEATURE_NAMES
    assert np.isfinite(values).all()


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
            raw_frame=True,
        )
        assert tuple(names) == TRENDLINE_REGISTRY_M5_LAYER_FEATURE_NAMES
        assert frame.index.equals(pd.DatetimeIndex(source["time"]))

    assert observed == ["M1", "M5"]
