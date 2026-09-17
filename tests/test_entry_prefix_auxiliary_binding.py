"""Native dataset prefix labels: exact rows, unchanged inputs, fail closed."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.contracts.entry_exit_production_architecture_v1 import PRODUCTION_MTF_PER_TF_WINDOW_BARS
from tests.test_entry_v10_ctx_dataset_memmap import _write_advanced_parquet
from tests.entry_v10_trainer_dataset_support import install_multi_tf_stub


def _binding(path):
    return {"path": str(path), "sha256": trainer._sha256_file(path)}


def _json(path, value):
    path.write_text(json.dumps(value))
    return _binding(path)


@pytest.fixture
def case(tmp_path, monkeypatch):
    def make(role="TRAIN"):
        parent = tmp_path / "parent_train.parquet"
        times = ([f"2026-01-01T00:{i*5:02d}:00Z" for i in range(3)] if role == "TRAIN"
                 else [f"2026-03-02T00:{i*5:02d}:00Z" for i in range(3)])
        _write_advanced_parquet(parent, times=times)
        ds = trainer.EntryV10CtxDataset(
            parent, seq_len=trainer.MODEL_NATIVE_SEQ_LEN,
            m5_prebuilt_path=install_multi_tf_stub(tmp_path, monkeypatch),
            per_tf_seq_lens=dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS),
            multi_tf_closed_bar=True,
        )
        rows_path = tmp_path / "rows.npy"
        np.save(rows_path, np.array([0, 2], dtype=np.int64))
        row_binding = _binding(rows_path)
        start, cutoff, end = "2025-06-01T00:00Z", "2026-03-01T00:00Z", "2026-06-01T00:00Z"
        design_binding = _json(tmp_path / "design.json", {"calendar": {
            "train_entry_start_inclusive": start, "train_control_cutoff": cutoff,
            "development_control_entry_end_exclusive": end,
            "bindings": {"CONTROL256_PARENT_ROWS": row_binding},
        }})
        common = {"fit_scope": "TRAIN_ONLY", "val_test_rows_used_for_fit": 0,
                  "train_start_utc": start, "train_end_utc": cutoff}
        direction = _json(tmp_path / "direction.json", {**common, "policy_sha256": "a"*64})
        size = _json(tmp_path / "size.json", {
            **common, "policy_sha256": "b"*64,
            "entry_causal_m1_target_policy_sha256": "a"*64})
        preparation = {
            "frozen_design": design_binding, "train_start_inclusive": start,
            "support_end_inclusive": cutoff, "direction_policy": direction,
            "direction_policy_sha256": "a"*64, "position_size_policy": size,
            "position_size_policy_sha256": "b"*64,
            "bindings": {"TRAIN_ELIGIBLE_PARENT_ROWS": row_binding},
        }
        prep_binding = _json(tmp_path / "preparation.json", preparation)
        columns = trainer._MODEL_NATIVE_POLICY_DEPENDENT_TARGET_COLS
        labels = pd.DataFrame({"parent_entry_row_index": np.array([0, 2], dtype=np.int64),
                               "time": pd.to_datetime([times[0], times[2]], utc=True)})
        for i, name in enumerate(columns):
            labels[name] = np.array([0.0, 1.0], dtype=np.float32)
        for name in ["y_long_expected_mae_bps", "y_short_expected_mae_bps"]:
            labels[name] = np.array([2.25, 8.5], dtype=np.float32)
        for name in ("m1_outcome_end_time_ns", "line_outcome_end_time_ns"):
            labels[name] = (pd.DatetimeIndex(labels.time).asi8 + 60*60*10**9)
        labels_path = tmp_path / "labels.parquet"
        labels.to_parquet(labels_path, index=False)
        result = {
            "schema_version": "gx1_prefix_policy_dependent_labels_v1",
            "active_target_columns": list(columns),
            "parent_entry_parquet": _binding(parent),
            "parent_entry_manifest": _binding(parent.with_suffix('.manifest.json')),
            "prefix_preparation": prep_binding, "direction_policy": direction,
            "position_size_policy": size, "train_support_end_inclusive": cutoff,
            "control_support_end_inclusive": end,
            "labels": {role: {**_binding(labels_path), "rows": 2, "row_binding": row_binding}},
        }
        result_path = tmp_path / "result.json"
        _json(result_path, result)
        return ds, labels, result, result_path, design_binding["sha256"], role
    return make


def _bind(c, **overrides):
    ds, _, _, path, design_sha, role = c
    arguments = dict(result_path=path, expected_result_sha256=trainer._sha256_file(path),
                     expected_design_sha256=design_sha, role=role)
    arguments.update(overrides)
    ds.bind_policy_dependent_auxiliary_targets(**arguments)


@pytest.mark.parametrize("role", ["TRAIN", "CONTROL256"])
def test_native_getitem_uses_bound_labels_and_preserves_all_inputs_and_raw_targets(case, role):
    c = case(role); ds, labels, result, _, _, _ = c
    before = ds.df.copy(deep=True)
    original = {i: ds[i] for i in range(3)}
    source_hash = trainer._sha256_file(ds.parquet_path)
    # Caller controls sampler order; binding must not resequence it.
    ds.indices = np.array([2, 0, 1])
    _bind(c)
    assert ds.indices.tolist() == [2, 0, 1]
    for i, parent_row in enumerate([2, 0]):
        output = ds[i]
        for name, previous in original[parent_row].items():
            if name in trainer._MODEL_NATIVE_POLICY_DEPENDENT_TARGET_COLS:
                expected = labels.loc[labels.parent_entry_row_index == parent_row, name].iloc[0]
                assert output[name].item() == expected
            else:
                assert torch.equal(output[name], previous), name
    unchanged = [name for name in before if name not in trainer._MODEL_NATIVE_POLICY_DEPENDENT_TARGET_COLS]
    pd.testing.assert_frame_equal(ds.df[unchanged], before[unchanged], check_exact=True)
    assert trainer._sha256_file(ds.parquet_path) == source_hash
    assert ds._policy_dependent_auxiliary_binding["role"] == role
    assert ds._policy_dependent_auxiliary_binding["inactive_diagnostics_refreshed"] is False
    with pytest.raises(RuntimeError, match="UNBOUND_ROW"):
        ds[2]
    with pytest.raises(RuntimeError, match="MODE_INVALID"):
        _bind(c)


@pytest.mark.parametrize("fault", [
    "wrong_result_hash", "wrong_design_hash", "parent_changed", "parent_path",
    "rows_reversed", "rows_missing", "time_shift", "future_m1", "future_line",
    "negative_mae", "fractional_mask", "nonfinite", "mask_sentinel", "double_precision",
    "row_binding", "control_fit", "policy_lineage", "policy_binding", "role",
])
def test_invalid_binding_is_atomic_and_leaves_legacy_dataset_usable(case, fault):
    c = case(); ds, labels, result, path, _, role = c
    before = ds.df.copy(deep=True); before_output = ds[0]
    kwargs = {}
    if fault == "wrong_result_hash": kwargs["expected_result_sha256"] = "f"*64
    elif fault == "wrong_design_hash": kwargs["expected_design_sha256"] = "f"*64
    elif fault == "parent_changed": ds.parquet_path.write_bytes(ds.parquet_path.read_bytes()+b"changed")
    elif fault == "parent_path": result["parent_entry_parquet"]["path"] = str(path)
    elif fault == "role": kwargs["role"] = "VAL"
    elif fault == "row_binding": result["labels"][role]["row_binding"]["sha256"] = "f"*64
    elif fault == "policy_binding": result["direction_policy"]["sha256"] = "f"*64
    elif fault in ("control_fit", "policy_lineage"):
        key = "direction_policy" if fault == "control_fit" else "position_size_policy"
        policy_path = Path(result[key]["path"])
        policy = json.loads(policy_path.read_text())
        if fault == "control_fit": policy["train_end_utc"] = "2026-06-01T00:00Z"
        else: policy["entry_causal_m1_target_policy_sha256"] = "f"*64
        result[key] = _json(policy_path, policy)
        prep_path = Path(result["prefix_preparation"]["path"])
        prep = json.loads(prep_path.read_text()); prep[key] = result[key]
        result["prefix_preparation"] = _json(prep_path, prep)
    else:
        if fault == "rows_reversed": labels = labels.iloc[::-1]
        elif fault == "rows_missing": labels = labels.iloc[:1]
        elif fault == "time_shift": labels.loc[0, "time"] += pd.Timedelta(minutes=5)
        elif fault in ("future_m1", "future_line"):
            name = "m1_outcome_end_time_ns" if fault == "future_m1" else "line_outcome_end_time_ns"
            labels.loc[0, name] = pd.Timestamp("2026-03-01T00:00:01Z").value
        elif fault == "negative_mae": labels.loc[0, "y_long_expected_mae_bps"] = -1
        elif fault == "fractional_mask": labels.loc[0, "y_line_support_touch_mask"] = 0.5
        elif fault == "nonfinite": labels.loc[0, "y_short_expected_mae_bps"] = np.nan
        elif fault == "mask_sentinel": labels.loc[0, "y_position_size_target"] = 0.5
        elif fault == "double_precision": labels["y_long_expected_mae_bps"] = labels.y_long_expected_mae_bps.astype(np.float64)
        labels_path = Path(result["labels"][role]["path"])
        labels.to_parquet(labels_path, index=False)
        result["labels"][role].update(_binding(labels_path))
    _json(path, result)
    with pytest.raises(RuntimeError):
        _bind(c, **kwargs)
    pd.testing.assert_frame_equal(ds.df, before, check_exact=True)
    assert getattr(ds, "_policy_dependent_auxiliary_binding", None) is None
    for name, value in ds[0].items(): assert torch.equal(value, before_output[name])
