from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_DECISION_BAR_SECONDS,
    ENTRY_FEATURE_SEQUENCE_BARS,
    EXIT_DECISION_BAR_SECONDS,
    EXIT_FEATURE_SEQUENCE_BARS,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CAT_MIN_MAX,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.features.entry_candlestick_patterns_v1 import (
    CANDLESTICK_PATTERN_FEATURE_NAMES,
)
from gx1.features.entry_model_native_feature_layers_v1 import (
    PRICE_DERIVED_CAUSAL_WARMUP_ROWS,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    PRICE_DERIVED_FEATURE_NAMES,
    build_candlestick_derived_layer,
    build_price_derived_layer,
)
from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_EVENT_AGE_CARRY_KEYS,
    foundation_event_age_carry_scope,
)
from gx1.features.entry_support_resistance_memory_v1 import (
    SUPPORT_RESISTANCE_MEMORY_STATE_KEYS,
    SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS,
    build_entry_support_resistance_memory_layer,
)
from gx1.features import entry_model_native_feature_layers_v1 as feature_layers
from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    _build_inline_seq_structure_extension,
)
from gx1.scripts import materialize_entry_exit_m1_feature_base_v1 as producer
from gx1.scripts.materialize_entry_exit_m1_feature_base_v1 import (
    _BOUNDED_CAUSAL_OVERLAP_ROWS,
    _build_bounded_extension_chunk,
)


def _synthetic_enriched_frame(rows: int) -> pd.DataFrame:
    index = np.arange(rows, dtype=np.float32)
    close = 2_000.0 + 0.08 * index + 0.35 * np.sin(index * 0.31)
    open_ = close - 0.12 * np.cos(index * 0.17)
    columns: dict[str, object] = {
        "time": pd.date_range(
            "2026-01-01",
            periods=rows,
            freq="min",
            tz="UTC",
        ),
        "open": open_,
        "high": np.maximum(open_, close) + 0.3,
        "low": np.minimum(open_, close) - 0.3,
        "close": close,
        "atr": np.full(rows, 1.25, dtype=np.float32),
    }
    numeric_fields = [
        *MODEL_NATIVE_BASE_FIELDS,
        *MODEL_NATIVE_CTX_CONT_FIELDS,
    ]
    for offset, name in enumerate(dict.fromkeys(numeric_fields)):
        if name in columns:
            continue
        columns[name] = (
            0.2 * np.sin(index * np.float32(0.013 + offset * 0.0001))
            + 0.1 * np.cos(index * np.float32(0.021 + offset * 0.00007))
        ).astype(np.float32)
    for name in (
        "H1_range_compression_ratio",
        "M15_range_compression_ratio",
    ):
        columns[name] = (
            1.0 + 0.05 * np.sin(index * np.float32(0.017))
        ).astype(np.float32)
    columns["atr_bps"] = np.full(rows, 10.0, dtype=np.float32)
    columns["spread_bps"] = np.full(rows, 1.0, dtype=np.float32)
    for name in MODEL_NATIVE_CTX_CAT_FIELDS:
        columns[name] = np.zeros(rows, dtype=np.int64)
    return pd.DataFrame(columns)


def _synthetic_enriched_sample_with_price_warmup(
    rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_frame = _synthetic_enriched_frame(rows + 202)
    sample_frame = source_frame.iloc[202:].reset_index(drop=True).copy()
    return sample_frame, source_frame


def _build_bounded_in_batches(
    frame: pd.DataFrame,
    source: Path,
    requested: list[str],
    *,
    batch_rows: int,
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    sample_times = frame[["time"]].copy()
    price, price_names = build_price_derived_layer(sample_times, source)
    candle, candle_names = build_candlestick_derived_layer(
        sample_times,
        source,
    )
    support_state = None
    foundation_event_age_state = None
    observed: list[np.ndarray] = []
    observed_meta = None
    observed_names = None
    for start in range(0, len(frame), batch_rows):
        stop = min(start + batch_rows, len(frame))
        prefix = max(0, start - _BOUNDED_CAUSAL_OVERLAP_ROWS)
        (
            chunk,
            names,
            meta,
            support_state,
            foundation_event_age_state,
        ) = _build_bounded_extension_chunk(
            frame.iloc[prefix:stop].reset_index(drop=True),
            source_parquet=source,
            requested_features=requested,
            ctx_cont_names=list(MODEL_NATIVE_CTX_CONT_FIELDS),
            ctx_cat_names=list(MODEL_NATIVE_CTX_CAT_FIELDS),
            base_signal_fields=list(MODEL_NATIVE_BASE_FIELDS),
            price_layer=price[prefix:stop],
            price_names=price_names,
            candle_layer=candle[prefix:stop],
            candle_names=candle_names,
            emit_offset=start - prefix,
            support_memory_state=support_state,
            foundation_event_age_state=foundation_event_age_state,
            source_contract_label="causal_enriched_m1_frame_v1",
        )
        observed.append(chunk)
        if observed_names is None:
            observed_names = names
            observed_meta = meta
        else:
            assert names == observed_names
            assert meta == observed_meta

    assert observed_names is not None
    assert observed_meta is not None
    return np.concatenate(observed), observed_names, observed_meta


def test_support_resistance_memory_state_is_exact_across_batches() -> None:
    rows = 97
    index = np.arange(rows, dtype=np.float32)
    matrix = np.column_stack(
        [
            0.25 * np.sin(index * np.float32(0.03 + column * 0.001))
            for column in range(len(SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS))
        ]
    ).astype(np.float32)
    names = list(SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS)

    expected, expected_names = build_entry_support_resistance_memory_layer(
        matrix,
        names,
    )
    state = None
    pieces: list[np.ndarray] = []
    for start, stop in ((0, 19), (19, 53), (53, 71), (71, rows)):
        piece, piece_names, state = (
            build_entry_support_resistance_memory_layer(
                matrix[start:stop],
                names,
                memory_state=state,
                return_memory_state=True,
            )
        )
        assert piece_names == expected_names
        pieces.append(piece)

    np.testing.assert_array_equal(np.concatenate(pieces), expected)


def test_bounded_owner_orchestration_matches_full_history_exactly(
    tmp_path: Path,
) -> None:
    frame, source_frame = _synthetic_enriched_sample_with_price_warmup(89)
    source = tmp_path / "synthetic_enriched_m1.parquet"
    source_frame.to_parquet(source, index=False)
    requested = [
        *MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
        "ctx_cont.d1_trend_age_mature_flag_v3",
        "chart.geometry_channel_center_bias",
        "candle.pattern_close_location",
    ]

    expected, expected_names, expected_meta = (
        _build_inline_seq_structure_extension(
            frame,
            requested_features=requested,
            ctx_cont_names=list(MODEL_NATIVE_CTX_CONT_FIELDS),
            ctx_cat_names=list(MODEL_NATIVE_CTX_CAT_FIELDS),
            source_parquet=source,
            source_contract_label="causal_enriched_m1_frame_v1",
            base_signal_fields=list(MODEL_NATIVE_BASE_FIELDS),
        )
    )
    observed, observed_names, observed_meta = _build_bounded_in_batches(
        frame,
        source,
        requested,
        batch_rows=23,
    )

    assert observed_names == expected_names
    assert observed_meta == expected_meta
    np.testing.assert_array_equal(observed, expected)


def test_foundation_event_ages_match_full_history_across_batch_boundary(
    tmp_path: Path,
) -> None:
    rows = 251
    boundary = 120
    events = (
        (
            "smc_bos_up",
            "chart.foundation_bos_up_age_bars",
            9,
        ),
        (
            "smc_bos_down",
            "chart.foundation_bos_down_age_bars",
            52,
        ),
        (
            "smc_choch",
            "chart.foundation_choch_age_bars",
            95,
        ),
    )
    frame, source_frame = _synthetic_enriched_sample_with_price_warmup(rows)
    for source_field, _, distance in events:
        frame[source_field] = np.zeros(rows, dtype=np.float32)
        frame.at[boundary - distance, source_field] = np.float32(1.0)
    source = tmp_path / "event_age_enriched_m1.parquet"
    source_frame.to_parquet(source, index=False)
    age_fields = [event[1] for event in events]
    requested = list(
        dict.fromkeys(
            [
                *MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
                *age_fields,
            ]
        )
    )

    expected, expected_names, expected_meta = (
        _build_inline_seq_structure_extension(
            frame,
            requested_features=requested,
            ctx_cont_names=list(MODEL_NATIVE_CTX_CONT_FIELDS),
            ctx_cat_names=list(MODEL_NATIVE_CTX_CAT_FIELDS),
            source_parquet=source,
            source_contract_label="causal_enriched_m1_frame_v1",
            base_signal_fields=list(MODEL_NATIVE_BASE_FIELDS),
        )
    )
    observed, observed_names, observed_meta = _build_bounded_in_batches(
        frame,
        source,
        requested,
        batch_rows=boundary,
    )

    assert observed_names == expected_names
    assert observed_meta == expected_meta
    for _, field, distance in events:
        column = expected_names.index(field)
        assert expected[boundary, column] == np.float32(distance)
    np.testing.assert_array_equal(observed, expected)


def test_foundation_event_age_carry_state_fails_closed() -> None:
    valid = {name: 0 for name in FOUNDATION_EVENT_AGE_CARRY_KEYS}
    malformed_states = (
        {},
        {**valid, "unexpected": 0},
        {**valid, FOUNDATION_EVENT_AGE_CARRY_KEYS[0]: np.nan},
        {**valid, FOUNDATION_EVENT_AGE_CARRY_KEYS[1]: 1.5},
    )
    for malformed in malformed_states:
        with pytest.raises(
            RuntimeError,
            match="FOUNDATION_EVENT_AGE_CARRY_STATE_INVALID",
        ):
            with foundation_event_age_carry_scope(
                malformed,
                tail_replay_rows=_BOUNDED_CAUSAL_OVERLAP_ROWS,
            ):
                pass


def _write_tiny_surface(
    path: Path,
    *,
    signal_width: int = MODEL_NATIVE_SIGNAL_DIM,
) -> pd.DatetimeIndex:
    times = pd.date_range(
        "2026-02-01",
        periods=3,
        freq="min",
        tz="UTC",
    )
    signal = np.arange(
        len(times) * signal_width,
        dtype=np.float32,
    ).reshape(len(times), signal_width)
    ctx_cont = np.zeros(
        (len(times), len(MODEL_NATIVE_CTX_CONT_FIELDS)),
        dtype=np.float32,
    )
    cat_row = np.asarray(
        [lower for lower, _upper in MODEL_NATIVE_CTX_CAT_MIN_MAX.values()],
        dtype=np.int64,
    )
    ctx_cat = np.tile(cat_row, (len(times), 1))
    table = pa.Table.from_arrays(
        [
            pa.array(times),
            producer._fixed_size_list_column(
                signal,
                signal_width,
                dtype=pa.float32(),
            ),
            producer._fixed_size_list_column(
                ctx_cont,
                len(MODEL_NATIVE_CTX_CONT_FIELDS),
                dtype=pa.float32(),
            ),
            producer._fixed_size_list_column(
                ctx_cat,
                len(MODEL_NATIVE_CTX_CAT_FIELDS),
                dtype=pa.int64(),
            ),
        ],
        names=["time", "signal", "ctx_cont", "ctx_cat"],
    )
    pq.write_table(table, path, row_group_size=2)
    return pd.DatetimeIndex(times).as_unit("ns")


def test_staged_surface_validation_preserves_exact_513_142_5_schema(
    tmp_path: Path,
) -> None:
    surface = tmp_path / "surface.parquet"
    times = _write_tiny_surface(surface)

    producer._validate_staged_feature_surface(
        surface,
        expected_times=times,
        batch_rows=2,
    )

    schema = pq.read_schema(surface)
    assert schema.field("signal").type.list_size == 513
    assert schema.field("ctx_cont").type.list_size == 142
    assert schema.field("ctx_cat").type.list_size == 5

    wrong_width = tmp_path / "wrong_width.parquet"
    _write_tiny_surface(wrong_width, signal_width=512)
    with pytest.raises(
        RuntimeError,
        match="STAGED_SIGNAL_SCHEMA_INVALID",
    ):
        producer._validate_staged_feature_surface(
            wrong_width,
            expected_times=times,
            batch_rows=2,
        )


@pytest.mark.parametrize(
    (
        "timeframe",
        "frequency",
        "bar_seconds",
        "sequence_bars",
        "alignment_required",
    ),
    (
        (
            "M1",
            "min",
            EXIT_DECISION_BAR_SECONDS,
            EXIT_FEATURE_SEQUENCE_BARS,
            True,
        ),
        (
            "M5",
            "5min",
            ENTRY_DECISION_BAR_SECONDS,
            ENTRY_FEATURE_SEQUENCE_BARS,
            False,
        ),
    ),
)
def test_m1_exit_and_m5_entry_use_the_same_bounded_surface_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeframe: str,
    frequency: str,
    bar_seconds: int,
    sequence_bars: int,
    alignment_required: bool,
) -> None:
    # The producer requires PRICE_DERIVED_CAUSAL_WARMUP_ROWS of source history
    # before the first surface row, so a six-row fixture encodes a frame from
    # which no valid surface can be produced at all.
    rows = PRICE_DERIVED_CAUSAL_WARMUP_ROWS + 8
    times = pd.date_range(
        "2026-03-01",
        periods=rows,
        freq=frequency,
        tz="UTC",
    )
    close = 2_000.0 + np.arange(rows, dtype=np.float64)
    columns: dict[str, object] = {
        "time": times,
        "open": close - 0.1,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "atr": np.ones(rows, dtype=np.float32),
    }
    for name in (*MODEL_NATIVE_BASE_FIELDS, *MODEL_NATIVE_CTX_CONT_FIELDS):
        columns.setdefault(name, np.zeros(rows, dtype=np.float32))
    for index, name in enumerate(MODEL_NATIVE_CTX_CAT_FIELDS):
        lower, _upper = tuple(MODEL_NATIVE_CTX_CAT_MIN_MAX.values())[index]
        columns[name] = np.full(rows, lower, dtype=np.int64)
    source = tmp_path / f"{timeframe.lower()}_source.parquet"
    pq.write_table(
        pa.Table.from_pandas(pd.DataFrame(columns), preserve_index=False),
        source,
        row_group_size=2,
    )
    alignment = None
    # No surface may begin inside the price layer's causal warmup, with or
    # without an alignment file.
    expected_times = pd.DatetimeIndex(times).as_unit("ns")[
        PRICE_DERIVED_CAUSAL_WARMUP_ROWS:
    ]
    if alignment_required:
        alignment = tmp_path / "m1_alignment.parquet"
        pd.DataFrame({"time": expected_times}).to_parquet(
            alignment,
            index=False,
        )

    def fake_price_layer(
        sample: pd.DataFrame,
        _source: Path,
    ) -> tuple[np.ndarray, list[str]]:
        return (
            np.zeros(
                (len(sample), len(PRICE_DERIVED_FEATURE_NAMES)),
                dtype=np.float32,
            ),
            list(PRICE_DERIVED_FEATURE_NAMES),
        )

    def fake_candle_layer(
        sample: pd.DataFrame,
        _source: Path,
    ) -> tuple[np.ndarray, list[str]]:
        return (
            np.zeros(
                (len(sample), len(CANDLESTICK_PATTERN_FEATURE_NAMES)),
                dtype=np.float32,
            ),
            list(CANDLESTICK_PATTERN_FEATURE_NAMES),
        )

    selected = [
        f"selected_{index}"
        for index in range(MODEL_NATIVE_SIGNAL_DIM - len(MODEL_NATIVE_BASE_FIELDS))
    ]

    def fake_extension(
        frame: pd.DataFrame,
        **kwargs: object,
    ) -> tuple[
        np.ndarray,
        list[str],
        dict[str, object],
        dict[str, np.float32],
        dict[str, int],
    ]:
        emit_offset = int(kwargs["emit_offset"])
        emitted_rows = len(frame) - emit_offset
        return (
            np.zeros((emitted_rows, len(selected)), dtype=np.float32),
            selected,
            {"mode": "synthetic_exact_owner_test", "features": selected},
            {
                name: np.float32(0.0)
                for name in SUPPORT_RESISTANCE_MEMORY_STATE_KEYS
            },
            {name: 0 for name in FOUNDATION_EVENT_AGE_CARRY_KEYS},
        )

    monkeypatch.setattr(
        feature_layers,
        "build_price_derived_layer",
        fake_price_layer,
    )
    monkeypatch.setattr(
        feature_layers,
        "build_candlestick_derived_layer",
        fake_candle_layer,
    )
    monkeypatch.setattr(
        producer,
        "_build_bounded_extension_chunk",
        fake_extension,
    )
    output = tmp_path / f"{timeframe.lower()}_surface.parquet"
    observed = producer._materialize_bounded_feature_surface(
        source=source,
        alignment=alignment,
        output=output,
        contract={
            "selected_fields": selected,
            "fields": [*MODEL_NATIVE_BASE_FIELDS, *selected],
        },
        timeframe=timeframe,
        bar_seconds=bar_seconds,
        sequence_bars=sequence_bars,
    )

    assert observed["rows"] == len(expected_times)
    assert output.is_file()
    producer._validate_staged_feature_surface(
        output,
        expected_times=expected_times,
        batch_rows=2,
    )


def test_both_native_routes_have_one_bounded_crash_consistent_path() -> None:
    materializer = inspect.getsource(producer._materialize_feature_base)
    bounded_writer = inspect.getsource(
        producer._materialize_bounded_feature_surface
    )
    module_source = inspect.getsource(producer)

    assert materializer.count("_materialize_bounded_feature_surface(") == 1
    assert "frame = pd.read_parquet(source)" not in materializer
    assert "load_m1_feature_surface" not in module_source
    assert ".replace(" not in module_source
    assert bounded_writer.index("_validate_staged_feature_surface(") < (
        bounded_writer.index("_admit_new_file(")
    )


@pytest.mark.parametrize(
    ("timeframe", "alignment"),
    (("M1", None), ("M5", Path("alignment.parquet"))),
)
def test_native_route_mismatch_fails_before_input_open(
    tmp_path: Path,
    timeframe: str,
    alignment: Path | None,
) -> None:
    with pytest.raises(
        RuntimeError,
        match="ENTRY_EXIT_FEATURE_BASE_NATIVE_ROUTE_INVALID",
    ):
        producer._materialize_feature_base(
            source_parquet=tmp_path / "source.parquet",
            alignment_parquet=alignment,
            seq_structure_manifest=tmp_path / "signal.json",
            output_parquet=tmp_path / "surface.parquet",
            dataset_run_id="run",
            pair_generation_id="a" * 64,
            timeframe=timeframe,
        )


def test_atomic_admission_never_overwrites_existing_target(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    destination = tmp_path / "destination"
    stage.write_bytes(b"new")
    destination.write_bytes(b"old")

    with pytest.raises(
        RuntimeError,
        match="ADMISSION_TARGET_EXISTS",
    ):
        producer._admit_new_file(stage, destination)

    assert destination.read_bytes() == b"old"


def test_manifest_is_validated_and_admitted_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "surface.parquet"
    output.write_bytes(b"validated parquet bytes")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    expected = {
        "schema_version": "test_feature_surface_v1",
        "decision": "PASS",
        "output_parquet": str(output),
    }
    monkeypatch.setattr(
        producer,
        "build_entry_exit_feature_surface_manifest",
        lambda **_kwargs: expected,
    )
    original_admit = producer._admit_new_file
    admission_observations: list[tuple[Path, bool, bool]] = []

    def record_admission(stage: Path, destination: Path) -> None:
        admission_observations.append(
            (destination, output.is_file(), destination.exists())
        )
        original_admit(stage, destination)

    monkeypatch.setattr(producer, "_admit_new_file", record_admission)
    observed = producer._publish_feature_base_manifest(
        timeframe="M5",
        dataset_run_id="run",
        pair_generation_id="a" * 64,
        source=tmp_path / "source.parquet",
        source_binding={},
        alignment=None,
        seq_structure_manifest=tmp_path / "signal.json",
        output=output,
        rows=3,
        contract={},
        extension_meta={},
    )

    assert observed == expected
    assert admission_observations == [(manifest_path, True, False)]
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == expected


def test_stale_or_failed_manifest_cannot_form_an_admissible_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "surface.parquet"
    output.write_bytes(b"validated parquet bytes")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_bytes(b"stale")
    builder_called = False

    def build_manifest(**_kwargs: object) -> dict[str, object]:
        nonlocal builder_called
        builder_called = True
        return {"decision": "PASS"}

    monkeypatch.setattr(
        producer,
        "build_entry_exit_feature_surface_manifest",
        build_manifest,
    )
    publish_args = {
        "timeframe": "M5",
        "dataset_run_id": "run",
        "pair_generation_id": "a" * 64,
        "source": tmp_path / "source.parquet",
        "source_binding": {},
        "alignment": None,
        "seq_structure_manifest": tmp_path / "signal.json",
        "output": output,
        "rows": 3,
        "contract": {},
        "extension_meta": {},
    }
    with pytest.raises(
        RuntimeError,
        match="MANIFEST_ADMISSION_STATE_INVALID",
    ):
        producer._publish_feature_base_manifest(**publish_args)
    assert not builder_called
    assert manifest_path.read_bytes() == b"stale"

    manifest_path.unlink()

    def fail_admission(_stage: Path, _destination: Path) -> None:
        raise RuntimeError("injected admission failure")

    monkeypatch.setattr(
        producer,
        "_admit_new_file",
        fail_admission,
    )
    with pytest.raises(RuntimeError, match="injected admission failure"):
        producer._publish_feature_base_manifest(**publish_args)
    assert output.read_bytes() == b"validated parquet bytes"
    assert not manifest_path.exists()
