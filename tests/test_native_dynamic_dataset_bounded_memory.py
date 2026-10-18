"""Small, synthetic regression proofs for the native dataset memory boundaries.

These verify exact cells/schema and exhaustive validation, never native input
admission, physical peak RSS, learning quality, or a complete producer run.
"""
from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.scripts import build_entry_v10_ctx_training_dataset_v3 as builder


@pytest.mark.parametrize("prefix_rows", [0, 2])
def test_context_trim_preserves_cells_attrs_and_original_without_wide_deepcopy(
    monkeypatch, prefix_rows
):
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-05-01", periods=6, freq="5min", tz="UTC"),
            "required_a": np.arange(6, dtype=np.float32),
            "required_b": np.arange(6, dtype=np.float32) + np.float32(0.25),
            "genuine_payload": np.arange(6, dtype=np.float32) * np.float32(-1.5),
            "source_mask": [False, True, False, True, True, False],
        },
        index=pd.Index([11, 13, 17, 19, 23, 29], name="source_row"),
    )
    if prefix_rows:
        frame.loc[frame.index[:prefix_rows], "required_a"] = np.nan
        frame.loc[frame.index[:1], "required_b"] = np.nan
    frame.attrs = {
        "group_a_checkpoint_complete_sha256": "a" * 64,
        "causal_source": {"ordered_fields": ["required_a", "required_b"]},
    }
    original = frame.copy(deep=True)
    attrs_before = deepcopy(frame.attrs)
    expected = frame.iloc[prefix_rows:].copy(deep=True)
    expected.index = pd.RangeIndex(len(expected))
    original_copy = pd.DataFrame.copy

    def forbid_wide_deepcopy(self, deep=True):
        if tuple(self.columns) == tuple(frame.columns) and deep is not False:
            raise AssertionError("whole feature frame deep-copied")
        return original_copy(self, deep=deep)

    with monkeypatch.context() as scoped:
        scoped.setattr(pd.DataFrame, "copy", forbid_wide_deepcopy)
        result = builder._trim_dataset_context_warmup(
            frame, ["required_a", "required_b"]
        )

    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(frame, original)
    assert frame.attrs == attrs_before
    for name, value in attrs_before.items():
        assert result.attrs[name] == value
    assert result.attrs.get("causal_context_warmup_rows_trimmed", 0) == prefix_rows
    assert isinstance(result.index, pd.RangeIndex)
    for name in ("required_a", "required_b", "genuine_payload", "source_mask"):
        assert np.shares_memory(result[name].to_numpy(), frame[name].to_numpy())


def test_context_trim_rejects_interior_nonfinite_instead_of_removing_genuine_rows():
    frame = pd.DataFrame({"required": [np.nan, 1.0, 2.0, np.nan], "payload": [3, 4, 5, 6]})
    with pytest.raises(RuntimeError, match="not a contiguous warmup prefix"):
        builder._trim_dataset_context_warmup(frame, ["required"])


@pytest.mark.parametrize("invalid", [None, np.nan, np.inf, -np.inf])
def test_matrix_finite_check_is_bounded_and_exhaustive_through_final_partial_batch(
    monkeypatch, invalid
):
    batch_rows = builder._MODEL_NATIVE_STREAMING_BATCH_SIZE
    rows = 2 * batch_rows + 1
    values = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
    if invalid is not None:
        values[-1, -1] = invalid
    original_isfinite = np.isfinite
    observed_rows = []

    def bounded_isfinite(part):
        assert part.ndim == 2
        assert part.shape[0] <= batch_rows
        assert np.shares_memory(part, values)
        observed_rows.append(part.shape[0])
        return original_isfinite(part)

    with monkeypatch.context() as scoped:
        scoped.setattr(builder.np, "isfinite", bounded_isfinite)
        if invalid is None:
            builder._require_finite_matrix_bounded(
                values, error="EXACT_SURFACE_NONFINITE", batch_rows=batch_rows
            )
        else:
            with pytest.raises(RuntimeError, match="EXACT_SURFACE_NONFINITE"):
                builder._require_finite_matrix_bounded(
                    values, error="EXACT_SURFACE_NONFINITE", batch_rows=batch_rows
                )
    assert sum(observed_rows) == rows
    assert observed_rows[-1] == 1


class _NoPythonList(np.ndarray):
    def tolist(self):
        raise AssertionError("native nested feature took ndarray.tolist path")


def _feature_batch(*, forbid_python_lists=False):
    sequences = [
        np.array([[0.0, -0.0, 0.1], [1.25, -2.5, 3.75]], dtype=np.float32),
        np.array([[10.0, -11.0, 12.5], [13.25, 14.5, -15.75]], dtype=np.float32),
    ]
    values = {
        "time": pd.date_range("2024-05-01", periods=2, freq="5min", tz="UTC"),
        "seq": sequences,
        "snap": [value[-1].copy() for value in sequences],
        "ctx_cont": [np.array([0.1, -0.0], dtype=np.float32),
                     np.array([-2.25, 3.5], dtype=np.float32)],
        "ctx_cat": [np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)],
        "forecast_observed": [False, True],
        "target": np.array([0.0, -1.5], dtype=np.float32),
        "optional_scalar": [None, 2.5],
        "all_null_scalar": [None, None],
        "diagnostic": ["unobserved", None],
    }
    if forbid_python_lists:
        for name in ("seq", "snap", "ctx_cont", "ctx_cat"):
            values[name] = [value.view(_NoPythonList) for value in values[name]]
    return pd.DataFrame(values)


def _legacy_small_batch_table(frame):
    """Independent old serialization behavior, safe only for this tiny fixture."""
    legacy = frame.copy(deep=True)
    for name in ("seq", "snap", "ctx_cont", "ctx_cat"):
        legacy[name] = [np.asarray(value).tolist() for value in legacy[name]]
    return pa.Table.from_pandas(legacy, preserve_index=False)


def test_native_arrow_batch_preserves_legacy_schema_cells_nulls_and_parquet(tmp_path):
    frame = _feature_batch()
    original = frame.copy(deep=True)
    expected = _legacy_small_batch_table(frame)
    actual = builder._dataset_arrow_batch(frame)
    assert actual.schema.equals(expected.schema, check_metadata=True)
    assert actual.equals(expected, check_metadata=True)
    assert actual.schema.field("seq").type == pa.list_(pa.list_(pa.float64()))
    assert actual.schema.field("ctx_cat").type == pa.list_(pa.int64())
    assert actual.column("all_null_scalar").null_count == len(frame)
    assert actual.column("optional_scalar").null_count == 1
    assert actual.column("diagnostic").null_count == 1
    pd.testing.assert_frame_equal(frame, original)
    path = tmp_path / "native_batch.parquet"
    pq.write_table(actual, path)
    legacy_path = tmp_path / "legacy_small_batch.parquet"
    pq.write_table(expected, legacy_path)
    restored = pq.read_table(path)
    legacy_restored = pq.read_table(legacy_path)
    assert restored.schema.equals(legacy_restored.schema, check_metadata=True)
    assert restored.equals(legacy_restored, check_metadata=True)
    # Preserve the signed zero as well as values that began as float32.
    seq = restored.column("seq").to_pylist()
    assert np.signbit(seq[0][0][1])
    assert seq[0][0][2] == float(np.float32(0.1))


def test_native_arrow_batch_avoids_python_nested_float_lists_and_reuses_schema():
    expected = _legacy_small_batch_table(_feature_batch())
    guarded = _feature_batch(forbid_python_lists=True)
    actual = builder._dataset_arrow_batch(guarded, schema=expected.schema)
    assert actual.schema.equals(expected.schema, check_metadata=True)
    assert actual.equals(expected, check_metadata=True)


@pytest.mark.parametrize("bad", [None, np.empty((0, 3), dtype=np.float32)])
def test_native_arrow_batch_rejects_unobserved_or_empty_nested_feature_cells(bad):
    frame = _feature_batch()
    frame.at[frame.index[0], "seq"] = bad
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        builder._dataset_arrow_batch(frame)


# The source loader must retain every column and pandas' real on-disk semantics.
# Tiny row groups deliberately exercise more than one Arrow chunk per column.
def _write_scalar_source_fixture(tmp_path, *, glitch=False, mixed=False):
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-05-01", periods=8, freq="5min", tz="UTC"),
            "bid_close": np.arange(8, dtype=np.float64) + 2000.0,
            "ask_close": np.arange(8, dtype=np.float64) + 2000.5,
            "genuine_float32": np.arange(8, dtype=np.float32) * np.float32(0.1),
            "genuine_float64": np.arange(8, dtype=np.float64) / 7.0,
            "nullable_integer": pd.array([0, 1, None, 3, 4, None, 6, 7], dtype="Int64"),
            "nullable_boolean": pd.array([True, None, False, True, None, False, True, False], dtype="boolean"),
            "categorical": pd.Categorical(
                ["quiet", "active", None, "quiet", "active", "quiet", None, "active"],
                categories=["quiet", "active"], ordered=True,
            ),
        },
        index=pd.Index([11, 13, 17, 19, 23, 29, 31, 37], name="source_row"),
    )
    if glitch:
        # Outside the selected window: the complete-source price guard remains.
        frame.loc[frame.index[1], ["bid_close", "ask_close"]] *= 10.0
    if mixed:
        frame["is_model_bar"] = True
    frame.attrs = {"source_role": "native_m5", "ordered_roles": {"value": "observed"}}
    table = pa.Table.from_pandas(frame, preserve_index=True)
    metadata = dict(table.schema.metadata or {})
    metadata[b"PANDAS_ATTRS"] = json.dumps(frame.attrs).encode("utf-8")
    path = tmp_path / "source.parquet"
    pq.write_table(table.replace_schema_metadata(metadata), path, row_group_size=2)
    return path, frame


def test_bounded_source_loader_preserves_installed_pandas_parquet_semantics(
    tmp_path, monkeypatch
):
    path, frame = _write_scalar_source_fixture(tmp_path)
    start, end = frame["time"].iloc[2], frame["time"].iloc[6]
    expected = pd.read_parquet(path, filters=[("time", ">=", start), ("time", "<=", end)])
    assert pq.ParquetFile(path).metadata.num_row_groups == 4
    original_read_table = pq.read_table
    conversions = []

    class ObservedArrowTable:
        def __init__(self, table):
            self.table = table

        def __getattr__(self, name):
            return getattr(self.table, name)

        def to_pandas(self, **kwargs):
            conversions.append(dict(kwargs))
            assert kwargs.get("split_blocks") is True
            assert kwargs.get("self_destruct") is True
            assert kwargs.get("use_threads") is False
            # Cache nothing from the Arrow object after self_destruct conversion.
            assert max(column.num_chunks for column in self.table.columns) > 1
            return self.table.to_pandas(**kwargs)

    def observed_read_table(*args, **kwargs):
        table = original_read_table(*args, **kwargs)
        # The historical price audit is intentionally a separate narrow read.
        if kwargs.get("filters") is not None:
            assert kwargs.get("use_threads") is False
            assert kwargs.get("use_pandas_metadata") is True
            assert kwargs.get("columns") is None
            return ObservedArrowTable(table)
        return table

    with monkeypatch.context() as scoped:
        scoped.setattr(pq, "read_table", observed_read_table)
        actual = builder._load_model_native_source_window(path, start=start, end=end)
    assert len(conversions) == 1
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.attrs == expected.attrs == frame.attrs
    assert actual.columns.equals(frame.columns)
    assert actual.index.equals(frame.index[2:7])
    assert actual["nullable_integer"].dtype == pd.Int64Dtype()
    assert actual["nullable_boolean"].dtype == pd.BooleanDtype()
    assert actual["categorical"].dtype == frame["categorical"].dtype


def test_source_loader_rejects_full_history_price_glitch_outside_window(tmp_path):
    path, frame = _write_scalar_source_fixture(tmp_path, glitch=True)
    with pytest.raises(RuntimeError, match="PRICE_SCALE_GLITCH"):
        builder._load_model_native_source_window(
            path, start=frame["time"].iloc[4], end=frame["time"].iloc[6]
        )


def test_source_loader_preserves_mixed_granularity_rejection(tmp_path):
    path, frame = _write_scalar_source_fixture(tmp_path, mixed=True)
    with pytest.raises(RuntimeError, match="SOURCE_PARQUET_MIXED_GRANULARITY_FORBIDDEN"):
        builder._load_model_native_source_window(
            path, start=frame["time"].iloc[2], end=frame["time"].iloc[6]
        )


def _legacy_normalize_time_utc(frame, time_col):
    """Exact small-data reference for the pre-042 normalization semantics."""
    out = frame.copy()
    out["time"] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    if out["time"].isna().any():
        raise RuntimeError("TIME_PARSE_FAIL: time column could not be parsed to tz-aware UTC")
    duplicate_count = int(out["time"].duplicated().sum())
    if duplicate_count:
        raise RuntimeError(f"TIME_DUPLICATE_ROWS: count={duplicate_count}")
    out = out.sort_values("time", kind="mergesort").copy()
    if len(out) == 0:
        raise RuntimeError("EMPTY_AFTER_TIME_NORMALIZATION")
    return out


def _source_preparation_frame():
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2024-05-01", periods=9, freq="5min", tz="UTC"),
            "bid_close": np.arange(9, dtype=np.float64) + 2000.0,
            "ask_close": np.arange(9, dtype=np.float64) + 2000.5,
            **{f"genuine_{i}": np.arange(9, dtype=np.float32) + np.float32(i / 8)
               for i in range(32)},
        },
        index=pd.Index(np.arange(9) * 3 + 10, name="source_row"),
    )
    frame.attrs = {"role": "all_native_inputs", "clock": {"closed": True}}
    return frame


@pytest.mark.parametrize("start_position,end_position", [(0, 8), (2, 6), (4, 4)])
def test_initial_native_preparation_preserves_inclusive_rows_without_wide_copy(
    monkeypatch, start_position, end_position
):
    frame = _source_preparation_frame()
    original = frame.copy(deep=True)
    attrs_before = deepcopy(frame.attrs)
    start, end = frame["time"].iloc[start_position], frame["time"].iloc[end_position]
    expected = _legacy_normalize_time_utc(frame.reset_index(drop=False), "time")
    expected = expected[(expected["time"] >= start) & (expected["time"] <= end)]
    payload_columns = {name for name in frame if name.startswith("genuine_")}
    original_copy = pd.DataFrame.copy

    def forbid_wide_deepcopy(self, deep=True):
        if payload_columns.issubset(self.columns) and deep is not False:
            raise AssertionError("native initial preparation copied the complete feature width")
        return original_copy(self, deep=deep)

    with monkeypatch.context() as scoped:
        scoped.setattr(pd.DataFrame, "copy", forbid_wide_deepcopy)
        actual = builder._prepare_model_native_source_window(frame, start=start, end=end)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(frame, original)
    assert frame.attrs == actual.attrs == attrs_before
    assert actual["time"].iloc[0] == start
    assert actual["time"].iloc[-1] == end
    for name in sorted(payload_columns):
        assert np.shares_memory(actual[name].to_numpy(), frame[name].to_numpy())


def test_initial_native_preparation_unsorted_rows_match_legacy_stable_normalization():
    frame = _source_preparation_frame().iloc[[5, 0, 8, 3, 1, 6, 2, 7, 4]].copy()
    original = frame.copy(deep=True)
    start, end = frame["time"].min() + pd.Timedelta(minutes=5), frame["time"].max() - pd.Timedelta(minutes=5)
    expected = _legacy_normalize_time_utc(frame.reset_index(drop=False), "time")
    expected = expected[(expected["time"] >= start) & (expected["time"] <= end)]
    actual = builder._prepare_model_native_source_window(frame, start=start, end=end)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(frame, original)
    assert actual.attrs == expected.attrs == original.attrs


@pytest.mark.parametrize("copy_data", [True, False])
@pytest.mark.parametrize("bad_clock", ["malformed", "duplicate", "empty"])
def test_time_normalization_errors_match_legacy_for_both_copy_modes(copy_data, bad_clock):
    frame = pd.DataFrame({"time": ["2024-05-01T00:05:00Z", "2024-05-01T00:00:00Z"], "value": [1.0, 2.0]})
    if bad_clock == "malformed":
        frame.loc[0, "time"] = "not-a-time"
    elif bad_clock == "duplicate":
        frame.loc[0, "time"] = frame.loc[1, "time"]
    else:
        frame = frame.iloc[0:0]
    original = frame.copy(deep=True)
    with pytest.raises(RuntimeError) as legacy_error:
        _legacy_normalize_time_utc(frame, "time")
    with pytest.raises(RuntimeError) as actual_error:
        builder._normalize_time_utc(frame, "time", copy_data=copy_data)
    assert str(actual_error.value) == str(legacy_error.value)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("copy_data", [True, False])
def test_time_normalization_preserves_stable_sort_values_and_input(copy_data):
    frame = pd.DataFrame(
        {"stamp": ["2024-05-01T00:10:00Z", "2024-05-01T00:00:00Z", "2024-05-01T00:05:00Z"],
         "genuine": np.array([3.5, 1.5, 2.5], dtype=np.float32)},
        index=pd.Index([31, 17, 23], name="native_row"),
    )
    frame.attrs = {"source": "native"}
    original = frame.copy(deep=True)
    expected = _legacy_normalize_time_utc(frame, "stamp")
    actual = builder._normalize_time_utc(frame, "stamp", copy_data=copy_data)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(frame, original)
    assert actual.attrs == expected.attrs == frame.attrs


def test_time_normalization_default_retains_deep_data_isolation():
    frame = _source_preparation_frame()
    original_value = frame["genuine_0"].iloc[0]
    actual = builder._normalize_time_utc(frame, "time")
    assert not np.shares_memory(actual["genuine_0"].to_numpy(), frame["genuine_0"].to_numpy())
    actual.loc[actual.index[0], "genuine_0"] = np.float32(-123.0)
    assert frame["genuine_0"].iloc[0] == original_value



def test_projected_source_preserves_owner_state_prices_types_attrs_and_disk_schema(
    tmp_path, monkeypatch
):
    required = builder._model_native_source_working_columns(builder.MODEL_NATIVE_BASE_FIELDS)
    _, source_owned = builder._model_native_artifact_owner_fields(builder.MODEL_NATIVE_BASE_FIELDS)
    assert len(required) == len(set(required))
    assert set(source_owned).issubset(required)
    assert set(builder.MODEL_NATIVE_CTX_CONT_SOURCE_PREFIX_FIELDS).issubset(required)
    assert set(builder.MODEL_NATIVE_CTX_CAT_FIELDS).issubset(required)
    assert {"time", "atr"}.issubset(required)
    assert set(required).isdisjoint(builder._gc_context.CONTEXT_FIELDS)
    path, frame = _write_scalar_source_fixture(tmp_path)
    raw_gc = builder._gc_context.CONTEXT_FIELDS[0]
    frame[raw_gc] = np.arange(len(frame), dtype=np.float32) + np.float32(30)
    for j, name in enumerate(required):
        if name not in frame:
            frame[name] = np.arange(len(frame), dtype=np.float32) + np.float32(j)
    # Required categorical state also retains pandas' nullable physical dtype.
    frame[builder.MODEL_NATIVE_CTX_CAT_FIELDS[0]] = pd.array(
        [0, 1, None, 0, 1, 0, None, 1], dtype="Int64"
    )
    table = pa.Table.from_pandas(frame, preserve_index=True)
    metadata = dict(table.schema.metadata or {})
    metadata[b"PANDAS_ATTRS"] = json.dumps(frame.attrs).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(metadata), path, row_group_size=2)
    before_schema = pq.ParquetFile(path).schema_arrow
    start, end = frame["time"].iloc[2], frame["time"].iloc[6]
    physical_columns = [name for name in frame if name in set(required) | {"bid_close", "ask_close"}]
    expected = pd.read_parquet(
        path, columns=physical_columns,
        filters=[("time", ">=", start), ("time", "<=", end)],
    )
    original_read = pq.read_table
    reads = []

    def observed_read(*args, **kwargs):
        if kwargs.get("filters") is not None:
            reads.append(kwargs["columns"])
            assert kwargs["columns"] == physical_columns
            assert raw_gc not in kwargs["columns"]
            assert kwargs["use_pandas_metadata"] is True
        return original_read(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(pq, "read_table", observed_read)
        actual = builder._load_model_native_source_window(
            path, start=start, end=end, columns=required
        )
    assert reads == [physical_columns]
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.attrs == expected.attrs == frame.attrs
    assert actual.index.equals(frame.index[2:7])
    assert actual[builder.MODEL_NATIVE_CTX_CAT_FIELDS[0]].dtype == pd.Int64Dtype()
    assert set(required).issubset(actual.columns)
    assert raw_gc not in actual
    assert before_schema.equals(pq.ParquetFile(path).schema_arrow, check_metadata=True)
    assert raw_gc in pq.ParquetFile(path).schema_arrow.names


@pytest.mark.parametrize("failure", ["price_glitch", "mixed_granularity"])
def test_projected_source_keeps_complete_history_price_and_schema_guards(tmp_path, failure):
    path, frame = _write_scalar_source_fixture(
        tmp_path, glitch=failure == "price_glitch", mixed=failure == "mixed_granularity"
    )
    error = "PRICE_SCALE_GLITCH" if failure == "price_glitch" else "SOURCE_PARQUET_MIXED_GRANULARITY_FORBIDDEN"
    with pytest.raises(RuntimeError, match=error):
        builder._load_model_native_source_window(
            path, start=frame["time"].iloc[4], end=frame["time"].iloc[6],
            columns=builder._model_native_source_working_columns(builder.MODEL_NATIVE_BASE_FIELDS),
        )


def test_projecting_working_source_keeps_all_bound_m5_model_channels():
    times = pd.date_range("2024-05-01", periods=5, freq="5min", tz="UTC")
    arrays = {
        "signal": np.arange(5 * builder.MODEL_NATIVE_SIGNAL_DIM, dtype=np.float32).reshape(5, -1),
        "ctx_cont": np.arange(5 * builder.MODEL_NATIVE_CTX_CONT_DIM, dtype=np.float32).reshape(5, -1),
        "ctx_cat": np.arange(5 * builder.MODEL_NATIVE_CTX_CAT_DIM, dtype=np.int64).reshape(5, -1),
    }
    aligned = builder._align_native_m5_feature_surface(
        target_times=times[1:4], surface_times=times, surface_arrays=arrays,
    )
    for name, values in arrays.items():
        assert np.shares_memory(aligned[name], values)
        np.testing.assert_array_equal(aligned[name], values[1:4])
    gc_positions = [builder.MODEL_NATIVE_CTX_CONT_FIELDS.index(name)
                    for name in builder._gc_context.CONTEXT_FIELDS]
    np.testing.assert_array_equal(
        aligned["ctx_cont"][:, gc_positions], arrays["ctx_cont"][1:4, gc_positions]
    )
    assert aligned["signal"].shape[1] == len(builder.MODEL_NATIVE_BASE_FIELDS) + builder.MODEL_NATIVE_SELECTED_FEATURE_COUNT
    assert aligned["ctx_cont"].shape[1] == len(builder.MODEL_NATIVE_CTX_CONT_FIELDS)
    assert aligned["ctx_cat"].shape[1] == len(builder.MODEL_NATIVE_CTX_CAT_FIELDS)


@pytest.fixture
def tiny_gc_surface(monkeypatch):
    """Two declared fields, real source evaluator, no producer/model/TEST data."""
    owner = builder._gc_context
    fields = [
        {"name": "declared_flow", "availability_field": "declared_flow_observed",
         "table": "declared", "column": "raw_flow", "validity_column": "raw_valid",
         "source_clock_column": "source_ns"},
        {"name": "declared_structural", "availability_field": "declared_structural_observed",
         "table": "declared", "column": "raw_structural", "role": "structural_observation_mask"},
    ]
    gc_names = tuple(name for item in fields for name in (item["name"], item["availability_field"]))
    names = ("non_gc_context", *gc_names)
    times = pd.date_range("2024-05-01", periods=2 * 512 + 1, freq="5min", tz="UTC")
    clock = times.as_unit("ns").asi8[[1, 3, 4]]
    table = pa.table({
        "available_ns": pa.array(clock, type=pa.int64()),
        "source_ns": pa.array(clock, type=pa.int64()),
        "raw_flow": pa.array([1.25, np.nan, -2.5], type=pa.float64()),
        "raw_valid": pa.array([1, 0, 1], type=pa.int64()),
        "raw_structural": pa.array([0, 1, 0], type=pa.int64()),
    })
    sources = {"declared": (clock, table)}
    surface = np.zeros((len(times), len(names)), dtype=np.float32)
    surface[:, 0] = np.float32(7.5)
    surface[1:3, 1] = np.float32(1.25)
    surface[4:, 1] = np.float32(-2.5)
    surface[1:3, 2] = 1
    surface[4:, 2] = 1
    surface[3, 3] = 1
    surface[1:, 4] = 1
    calls = {"sources": 0, "fields": []}
    original_field_values = owner._field_values

    def counted_sources():
        calls["sources"] += 1
        return sources

    def counted_field_values(item, *args):
        calls["fields"].append(item["name"])
        return original_field_values(item, *args)

    def forbid_dense_context(*args, **kwargs):
        raise AssertionError("parity rebuilt the dense GC context")

    monkeypatch.setattr(owner, "REGISTRY", {"fields": fields})
    monkeypatch.setattr(owner, "REGISTRY_SHA256", "a" * 64)
    monkeypatch.setattr(owner, "VALUE_FIELDS", tuple(item["name"] for item in fields))
    monkeypatch.setattr(owner, "CONTEXT_FIELDS", gc_names)
    monkeypatch.setattr(owner, "_sources", counted_sources)
    monkeypatch.setattr(owner, "_field_values", counted_field_values)
    monkeypatch.setattr(owner, "build_context", forbid_dense_context)
    monkeypatch.setattr(builder, "MODEL_NATIVE_CTX_CONT_FIELDS", names)
    return times, names, surface, calls, sources


def test_gc_parity_uses_real_owner_once_per_field_and_every_final_partial_cell(
    tiny_gc_surface, monkeypatch
):
    times, names, surface, calls, _ = tiny_gc_surface
    original_equal = np.array_equal
    comparisons = []

    def bounded_equal(actual, expected):
        # pandas also compares the full int64 decision clock while hashing it.
        # Bound only the actual float32 model-column parity comparisons.
        if (isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray)
                and actual.dtype == expected.dtype == np.dtype(np.float32)
                and np.shares_memory(actual, surface)):
            assert actual.ndim == expected.ndim == 1
            assert 0 < len(actual) <= 512
            comparisons.append(len(actual))
        return original_equal(actual, expected)

    with monkeypatch.context() as scoped:
        scoped.setattr(builder.np, "array_equal", bounded_equal)
        report = builder._require_gc_context_surface_parity(
            decisions=times, ctx_cont_names=names, ctx_cont_mat=surface,
        )
    assert calls == {"sources": 1, "fields": ["declared_flow", "declared_structural"]}
    assert comparisons == [512, 512, 1] * 4
    assert report["rows"] == len(times)
    assert report["value_fields"] == 2
    assert report["context_fields"] == 4
    assert report["compared_cells"] == len(times) * 4
    assert report["verified_all_rows"] is True
    assert report["mismatch_cells"] == 0
    assert report["registry_sha256"] == "a" * 64


@pytest.mark.parametrize("failure", ["order", "missing_field", "shape", "dtype", "value", "mask_missing", "mask_domain"])
def test_gc_parity_rejects_schema_or_last_partial_value_and_mask_mismatch(tiny_gc_surface, failure):
    times, names, surface, _, _ = tiny_gc_surface
    if failure == "order":
        names = (names[0], names[2], names[1], *names[3:])
    elif failure == "missing_field":
        names, surface = names[:-1], surface[:, :-1]
    elif failure == "shape":
        surface = surface[:-1]
    elif failure == "dtype":
        surface = surface.astype(np.float64)
    elif failure == "value":
        surface[-1, 1] += np.float32(1)
    elif failure == "mask_missing":
        surface[-1, 2] = 0
    elif failure == "mask_domain":
        surface[-1, 2] = 2
    error = ("NATIVE_GC_CONTEXT_SURFACE_FIELD_ORDER_INVALID" if failure in ("order", "missing_field")
             else "NATIVE_GC_CONTEXT_SURFACE_SCHEMA_INVALID" if failure in ("shape", "dtype")
             else "NATIVE_GC_CONTEXT_PARITY:declared_flow")
    with pytest.raises(RuntimeError, match=error):
        builder._require_gc_context_surface_parity(
            decisions=times, ctx_cont_names=names, ctx_cont_mat=surface,
        )


@pytest.mark.parametrize("failure,error", [
    ("future_clock", "NATIVE_GC_FUTURE_SOURCE_CLOCK"),
    ("invalid_validity", "NATIVE_GC_SOURCE_VALIDITY_DOMAIN"),
    ("float32_overflow", "NATIVE_GC_NONFINITE"),
    ("invalid_structural", "NATIVE_GC_STRUCTURAL_VALUE_DOMAIN"),
])
def test_gc_parity_preserves_source_owner_clock_validity_and_storage_rejections(
    tiny_gc_surface, failure, error
):
    times, names, surface, _, sources = tiny_gc_surface
    clock, table = sources["declared"]
    column, values = {
        "future_clock": ("source_ns", [int(clock[0]), int(clock[1]), int(clock[2]) + 1]),
        "invalid_validity": ("raw_valid", [1, 0, 2]),
        "float32_overflow": ("raw_flow", [1.25, np.nan, 1e50]),
        "invalid_structural": ("raw_structural", [0, 1, 2]),
    }[failure]
    index = table.schema.get_field_index(column)
    table = table.set_column(index, column, pa.array(values, type=table.schema.field(column).type))
    sources["declared"] = (clock, table)
    with np.errstate(over="ignore"), pytest.raises(RuntimeError, match=error):
        builder._require_gc_context_surface_parity(
            decisions=times, ctx_cont_names=names, ctx_cont_mat=surface,
        )
