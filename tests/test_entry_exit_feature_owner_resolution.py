from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_EXIT_FEATURE_BASE_SCHEMA_VERSION,
    entry_exit_feature_owner_resolution_contract,
    entry_exit_shared_feature_base_contract,
    require_entry_exit_feature_surface_identity,
    require_entry_exit_shared_feature_base_contract,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.features.entry_candlestick_patterns_v1 import (
    build_entry_candlestick_pattern_layer,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)
from gx1.scripts import build_entry_exit_m1_enriched_frame_v1 as enriched_producer
from gx1.scripts import materialize_entry_exit_m1_feature_base_v1 as feature_producer
from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    _align_native_m5_feature_surface,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _candle_frame(*, freq: str, falling: bool) -> pd.DataFrame:
    rows = 12
    time = pd.date_range("2026-01-01", periods=rows, freq=freq, tz="UTC")
    step = np.linspace(0.0, 2.2, rows)
    close = 2_000.0 - step if falling else 2_000.0 + step
    close = close + 0.18 * np.sin(np.arange(rows, dtype=np.float64) * 1.7)
    open_ = np.roll(close, 1)
    open_[0] = close[0] - (0.15 if not falling else -0.15)
    return pd.DataFrame(
        {
            "time": time,
            "open": open_,
            "high": np.maximum(open_, close) + 0.25,
            "low": np.minimum(open_, close) - 0.25,
            "close": close,
        }
    )


def test_all_eight_owners_are_bound_to_local_m5_and_m1_with_closed_mtf() -> None:
    contract = entry_exit_feature_owner_resolution_contract()

    assert tuple(contract["owners"]) == MODEL_NATIVE_TRAINING_SPECIALISTS
    assert len(contract["owners"]) == 8
    assert contract["pre_owner_combined_m1_m5_source_package_allowed"] is False
    assert contract["cross_resolution_value_copy_allowed"] is False
    assert contract["computed_feature_resampling_allowed"] is False
    assert contract["mtf_construction"] == (
        "closed_ohlcv_before_feature_computation"
    )
    assert contract["entry_route"] == {
        "local_timeframe": "M5",
        "mtf_context_timeframes": ["M15", "H1", "H4", "D1"],
    }
    assert contract["exit_route"] == {
        "local_timeframe": "M1",
        "mtf_context_timeframes": ["M5", "M15", "H1", "H4", "D1"],
    }
    assert sum(contract["mandatory_feature_count_by_owner"].values()) == 378
    for owner in MODEL_NATIVE_TRAINING_SPECIALISTS:
        assert contract["mandatory_feature_count_by_owner"][owner] > 0
        assert contract["owners"][owner] == {
            "M5": {
                "consumer": "ENTRY",
                "source": "closed_native_m5_rows",
                "bar_seconds": 300,
            },
            "M1": {
                "consumer": "EXIT",
                "source": "closed_native_m1_rows",
                "bar_seconds": 60,
            },
        }


def test_shared_contract_rejects_one_owner_missing_one_resolution() -> None:
    contract = entry_exit_shared_feature_base_contract()
    assert contract["schema_version"] == ENTRY_EXIT_FEATURE_BASE_SCHEMA_VERSION

    stale = copy.deepcopy(contract)
    del stale["feature_owner_resolution_contract"]["owners"][
        "price_action_candle_encoder"
    ]["M1"]
    with pytest.raises(
        RuntimeError,
        match="ENTRY_EXIT_SHARED_FEATURE_BASE_CONTRACT_MISMATCH",
    ):
        require_entry_exit_shared_feature_base_contract(
            stale,
            context="PYTEST",
        )


def test_exit_surface_must_match_entry_field_manifest_and_rank_state() -> None:
    fields = ["snap.a", "chart.b"]
    signal_manifest = "/immutable/signal.json"
    signal_sha = "a" * 64
    rank_sha = "b" * 64
    surface = {
        "anchor_timeframe": "M1",
        "feature_field_order": fields,
        "feature_field_order_sha256": _canonical_sha256(fields),
        "seq_structure_manifest": signal_manifest,
        "seq_structure_manifest_sha256": signal_sha,
        "rank_reference_sha256": rank_sha,
        "shared_feature_base_contract": entry_exit_shared_feature_base_contract(),
    }
    assert require_entry_exit_feature_surface_identity(
        surface,
        expected_timeframe="M1",
        expected_ordered_fields=fields,
        expected_signal_manifest_path=signal_manifest,
        expected_signal_manifest_sha256=signal_sha,
        expected_rank_reference_sha256=rank_sha,
        context="PYTEST",
    ) == surface

    for field in (
        "feature_field_order",
        "seq_structure_manifest_sha256",
        "rank_reference_sha256",
    ):
        stale = copy.deepcopy(surface)
        stale[field] = [] if field == "feature_field_order" else "c" * 64
        with pytest.raises(
            RuntimeError,
            match="ENTRY_EXIT_RESOLUTION_SURFACE_IDENTITY_MISMATCH",
        ):
            require_entry_exit_feature_surface_identity(
                stale,
                expected_timeframe="M1",
                expected_ordered_fields=fields,
                expected_signal_manifest_path=signal_manifest,
                expected_signal_manifest_sha256=signal_sha,
                expected_rank_reference_sha256=rank_sha,
                context="PYTEST",
            )


def test_m1_and_m5_wrappers_call_one_shared_owner_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enriched_calls: list[dict[str, object]] = []
    feature_calls: list[dict[str, object]] = []

    def fake_enriched(**kwargs: object) -> dict[str, object]:
        enriched_calls.append(dict(kwargs))
        return dict(kwargs)

    def fake_feature(**kwargs: object) -> dict[str, object]:
        feature_calls.append(dict(kwargs))
        return dict(kwargs)

    monkeypatch.setattr(enriched_producer, "_build_enriched_frame", fake_enriched)
    monkeypatch.setattr(feature_producer, "_materialize_feature_base", fake_feature)

    enriched_producer.build_m1_enriched_frame(marker="exit")
    enriched_producer.build_m5_enriched_frame(marker="entry")
    feature_producer.materialize_m1_feature_base(marker="exit")
    feature_producer.materialize_m5_feature_base(marker="entry")

    assert enriched_calls == [
        {"timeframe": "M1", "marker": "exit"},
        {"timeframe": "M5", "marker": "entry"},
    ]
    assert feature_calls == [
        {"timeframe": "M1", "marker": "exit"},
        {"timeframe": "M5", "marker": "entry"},
    ]


def test_candlestick_owner_computes_native_m1_and_m5_values_causally() -> None:
    m1 = _candle_frame(freq="min", falling=False)
    m5 = _candle_frame(freq="5min", falling=True)

    m1_values, m1_names = build_entry_candlestick_pattern_layer(m1)
    m5_values, m5_names = build_entry_candlestick_pattern_layer(m5)

    assert m1_names == m5_names
    assert m1_values.shape == m5_values.shape == (12, len(m1_names))
    assert not np.array_equal(m1_values, m5_values)

    m1_prefix, prefix_names = build_entry_candlestick_pattern_layer(m1.iloc[:8])
    m5_prefix, _ = build_entry_candlestick_pattern_layer(m5.iloc[:8])
    assert prefix_names == m1_names
    np.testing.assert_array_equal(m1_values[:8], m1_prefix)
    np.testing.assert_array_equal(m5_values[:8], m5_prefix)


def _m5_surface_arrays(rows: int) -> dict[str, np.ndarray]:
    return {
        "signal": np.zeros((rows, MODEL_NATIVE_SIGNAL_DIM), dtype=np.float32),
        "ctx_cont": np.zeros(
            (rows, MODEL_NATIVE_CTX_CONT_DIM), dtype=np.float32
        ),
        "ctx_cat": np.zeros((rows, MODEL_NATIVE_CTX_CAT_DIM), dtype=np.int64),
    }


def test_entry_uses_one_zero_copy_m5_surface_window_for_each_split() -> None:
    surface_times = pd.date_range(
        "2026-01-01", periods=8, freq="5min", tz="UTC"
    )
    arrays = _m5_surface_arrays(len(surface_times))

    aligned = _align_native_m5_feature_surface(
        target_times=surface_times[2:7],
        surface_times=surface_times,
        surface_arrays=arrays,
    )

    assert aligned["signal"].shape == (5, MODEL_NATIVE_SIGNAL_DIM)
    assert aligned["ctx_cont"].shape == (5, MODEL_NATIVE_CTX_CONT_DIM)
    assert aligned["ctx_cat"].shape == (5, MODEL_NATIVE_CTX_CAT_DIM)
    for name in arrays:
        assert np.shares_memory(aligned[name], arrays[name])


@pytest.mark.parametrize(
    ("target_positions", "message"),
    [
        ([0, 2], "WINDOW_NONCONTIGUOUS"),
        ([0, 8], "TIME_MISSING"),
    ],
)
def test_entry_m5_surface_alignment_fails_closed(
    target_positions: list[int], message: str
) -> None:
    surface_times = pd.date_range(
        "2026-01-01", periods=8, freq="5min", tz="UTC"
    )
    target_times = pd.DatetimeIndex(
        [surface_times[0], surface_times[0] + pd.Timedelta(minutes=5 * target_positions[1])]
    )

    with pytest.raises(RuntimeError, match=message):
        _align_native_m5_feature_surface(
            target_times=target_times,
            surface_times=surface_times,
            surface_arrays=_m5_surface_arrays(len(surface_times)),
        )
