"""The V3 per-bar multi-timeframe contract.

The 513 signal fields are M5-only because their builders sit on 199 upstream
source fields, 194 of them derived, with dependencies between the families - so
producing them per timeframe means rebuilding the entire context pipeline. Two
owners have no such dependency: the candlestick family needs exactly
["open", "high", "low", "close", "time"] and swing structure is a pure function
of (high, low, close). V3 adds those to V2's 25 so the higher timeframes carry
real price geometry instead of a generic lens.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

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


def test_v3_extends_v2_without_redefining_it() -> None:
    assert htf.MULTI_TF_FEATURE_COUNT_V2 == 25
    assert htf.MULTI_TF_FEATURE_COUNT_V3 == 90
    # V2 is a bit-identical prefix, so every artifact built on V2 stays valid.
    assert htf.MULTI_TF_PER_BAR_FEATURES_V3[:25] == htf.MULTI_TF_PER_BAR_FEATURES_V2
    assert len(set(htf.MULTI_TF_PER_BAR_FEATURES_V3)) == 90
    assert htf.MULTI_TF_FEATURE_NAMES_SHA256_V3 != htf.MULTI_TF_FEATURE_NAMES_SHA256_V2
    assert htf.HTF_V3_MATRIX_CONTRACT != htf.HTF_V2_MATRIX_CONTRACT


def test_v3_candlestick_names_come_from_the_owner() -> None:
    """A rename in the candlestick owner must not silently desync this contract."""
    from gx1.features.entry_candlestick_patterns_v1 import (
        CANDLESTICK_PATTERN_FEATURE_NAMES,
    )

    assert len(htf.MULTI_TF_PER_BAR_CANDLESTICK_V3) == len(
        CANDLESTICK_PATTERN_FEATURE_NAMES
    )
    for declared, owned in zip(
        htf.MULTI_TF_PER_BAR_CANDLESTICK_V3,
        CANDLESTICK_PATTERN_FEATURE_NAMES,
        strict=True,
    ):
        assert declared.endswith(owned.split(".", 1)[1])


def test_v3_holds_its_contract_at_every_resolution() -> None:
    """Exact width, exact order, and a single causal warmup prefix per timeframe."""
    m5 = _bars(20000, seed=7)
    for timeframe, rule in htf.MULTI_TF_RESAMPLE_RULES.items():
        resampled = (
            m5.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        if len(resampled) < 300:
            continue
        built = htf.compute_per_bar_features_v3(resampled)
        assert tuple(built.columns) == htf.MULTI_TF_PER_BAR_FEATURES_V3, timeframe
        matrix = built.to_numpy(dtype=np.float32)
        warmup = htf.validate_causal_feature_matrix(
            matrix,
            expected_width=htf.MULTI_TF_FEATURE_COUNT_V3,
            context=f"HTF_V3_{timeframe}",
        )
        assert np.isfinite(matrix[warmup:]).all(), timeframe


def test_v3_first_25_columns_equal_v2_exactly() -> None:
    """Adding families may not perturb a single existing value."""
    bars = _bars(6000, seed=11)
    v2 = htf.compute_per_bar_features_v2(bars).to_numpy(dtype=np.float64)
    v3 = htf.compute_per_bar_features_v3(bars).to_numpy(dtype=np.float64)

    assert v3.shape[1] == 90
    both_finite = np.isfinite(v2) & np.isfinite(v3[:, :25])
    assert np.array_equal(v2[both_finite], v3[:, :25][both_finite])
    assert np.array_equal(np.isfinite(v2), np.isfinite(v3[:, :25]))


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
    original = htf.compute_per_bar_features_v4(bars)
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
    future_changed = htf.compute_per_bar_features_v4(changed)
    assert np.array_equal(
        original.iloc[:cutoff].to_numpy(),
        future_changed.iloc[:cutoff].to_numpy(),
        equal_nan=True,
    )

    # A valid trend can place the latest confirmed low above an older
    # confirmed high. That remains available structure evidence.
    trending = htf.compute_per_bar_features_v4(_bars(4000, seed=0))
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
    matrix = htf.compute_per_bar_features_v4(_bars(4000, seed=31))
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

    split_times = {
        "train": pd.date_range(
            "2026-01-20T00:00:00Z", periods=2, freq="5min"
        ),
        "val": pd.date_range(
            "2026-01-20T00:10:00Z", periods=2, freq="5min"
        ),
        "test": pd.date_range(
            "2026-01-20T00:20:00Z", periods=2, freq="5min"
        ),
    }
    lengths = {"M5": 2, "M15": 2, "H1": 2, "H4": 2, "D1": 2}
    proof = htf.require_multi_tf_decision_window_coverage(
        features,
        per_tf_seq_lens=lengths,
        decision_times_by_split=split_times,
    )

    assert proof["all_split_boundaries_sliceable"] is True
    assert proof["resolution_pyramid"]["per_tf_seq_lens"] == lengths
    assert set(proof["per_tf"]) == set(htf.MULTI_TF_RESAMPLE_RULES)
    assert len(proof["contract_sha256"]) == 64

    features["D1"].attrs["causal_warmup_rows"] = 399
    with np.testing.assert_raises_regex(
        RuntimeError,
        "MULTI_TF_DECISION_COVERAGE_UNAVAILABLE.*D1",
    ):
        htf.require_multi_tf_decision_window_coverage(
            features,
            per_tf_seq_lens=lengths,
            decision_times_by_split=split_times,
        )


def test_v4_cache_surface_excludes_every_open_trailing_resample_bucket(
    monkeypatch,
) -> None:
    source = _bars(419, seed=37)

    def finite_contract(frame: pd.DataFrame) -> pd.DataFrame:
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


def test_dataset_rejects_undeclared_and_split_brain_contracts() -> None:
    """The Dataset reads the cache's declaration; it may not assume one.

    Pinning the width to MULTI_TF_FEATURE_COUNT_V2 is what held the higher
    timeframes at 25 generic features. Now the tables say what they are: an
    undeclared contract, an unknown one, or five timeframes that disagree all
    fail closed rather than silently taking V2.
    """
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    source = (
        pathlib.Path(trainer.__file__).read_text(encoding="utf-8")
    )
    for marker in (
        "MULTI_TF_CONTRACT_SPLIT_BRAIN",
        "MULTI_TF_CONTRACT_UNKNOWN",
        "MULTI_TF_CONTRACT_WIDTH_MISMATCH",
        "MULTI_TF_CONTRACT_ORDER_MISMATCH",
    ):
        assert marker in source, marker
    # and the width is no longer pinned to a constant
    assert "self._multi_tf_feature_count = int(MULTI_TF_FEATURE_COUNT_V2)" not in source


def test_bundle_and_normalization_read_the_declared_contract() -> None:
    """A lineage that records V2's names beside a V3 cache is train != serve."""
    import pathlib as _p

    repo = _p.Path(__file__).resolve().parents[1]

    normalization = (
        repo / "gx1/models/entry_v10/entry_v10_input_normalization.py"
    ).read_text()
    assert "def resolve_mtf_per_bar_contract(" in normalization
    assert "ENTRY_INPUT_NORMALIZATION_MTF_CONTRACT_UNKNOWN" in normalization
    # the lineage hash comes from the manifest, not from an imported constant
    assert '_mtf_manifest_declared["feature_names"]' in normalization

    bundle = (repo / "gx1/models/entry_v10/entry_v10_bundle.py").read_text()
    assert "ENTRY_BUNDLE_MODEL_NATIVE_MTF_EIGHT_FAMILY_REQUIRED" in bundle
    assert "HTF_V4_MATRIX_CONTRACT" in bundle


def test_trainer_sweeps_orphaned_memmap_scratch() -> None:
    """Killed runs leaked 69 GB each; four were found holding 295 GB.

    TemporaryDirectory cleans up through a finalizer that a killed process never
    runs. The scratch name carries the creating PID, so an orphan is provable
    rather than guessed, and a live PID is left alone so a concurrent run is
    safe.
    """
    import os as _os
    import pathlib as _pathlib
    import tempfile

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    with tempfile.TemporaryDirectory() as raw_root:
        root = _pathlib.Path(raw_root)
        dead = root / "v10_seq513_dataset__DIR_H24B_train_999999999_abc"
        dead.mkdir()
        (dead / "seq.float32.mmap").write_bytes(b"x" * 32)
        alive = root / f"v10_seq513_dataset__DIR_H24B_train_{_os.getpid()}_xyz"
        alive.mkdir()
        (alive / "seq.float32.mmap").write_bytes(b"x" * 32)

        trainer._sweep_orphaned_memmap_scratch(root)

        assert not dead.exists(), "orphaned scratch survived the sweep"
        assert alive.exists(), "the sweep removed a live run's scratch"
