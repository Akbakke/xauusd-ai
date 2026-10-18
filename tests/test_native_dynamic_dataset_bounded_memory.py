"""Small, synthetic regression proofs for the native dataset memory boundaries.

These verify exact cells/schema and exhaustive validation, never native input
admission, physical peak RSS, learning quality, or a complete producer run.
"""
from __future__ import annotations

from copy import deepcopy

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
