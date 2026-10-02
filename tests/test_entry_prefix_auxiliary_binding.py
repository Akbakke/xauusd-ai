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


# Physical v38 mode reuses original targets; these synthetic fixtures exercise
# source routing and fail-closed behavior, not target quality or native admission.
@pytest.fixture(scope="module")
def physical_policies(tmp_path_factory):
    from tests.test_entry_causal_m1_target_policy import _fit, _m1, _sha
    from gx1.contracts.entry_causal_m1_position_size_target_policy_v1 import (
        fit_causal_m1_position_size_target_policy,
    )
    m1 = _m1()
    direction = _fit()
    size = fit_causal_m1_position_size_target_policy(
        closed_m5=pd.DataFrame({"time": m1.time.iloc[::5].reset_index(drop=True)}),
        closed_m1=m1, entry_causal_m1_target_policy=direction,
        source_parquet_sha256=_sha("m5"), tape_provenance_sha256=_sha("tape"),
        m1_source_sha256=_sha("m1"),
        ecdf_artifact_path=tmp_path_factory.mktemp("physical_aux_policy") / "ecdf.npy",
    )
    return direction, size


@pytest.fixture
def physical_case(tmp_path, physical_policies):
    import hashlib
    from gx1.contracts.entry_causal_m1_position_size_target_policy_v1 import (
        causal_m1_position_size_target_policy_contract,
    )
    from gx1.contracts.entry_model_native_aux_targets_v3 import (
        model_native_aux_target_contract_metadata,
    )
    direction, size = physical_policies
    start = pd.Timestamp(direction["train_start_utc"])
    declared_end = pd.Timestamp(direction["train_end_utc"]) + pd.Timedelta(minutes=5)
    cutoff = declared_end + pd.Timedelta(seconds=1)
    end = pd.Timestamp("2024-01-04T00:00Z")
    fixed = model_native_aux_target_contract_metadata()
    clock_hash = lambda ns: hashlib.sha256(np.asarray(ns, dtype="<i8").tobytes()).hexdigest()

    def make(*, full=False):
        sources, frames, proofs = {}, {}, {}
        for split, count, first in (
            ("train", 3, start), ("val", 257, pd.Timestamp("2024-01-02T00:00Z"))
        ):
            path = tmp_path / (split + ".parquet")
            times = pd.date_range(first, periods=count, freq="5min")
            frame = pd.DataFrame({"time": times})
            for name in trainer._MODEL_NATIVE_ACTIVE_TARGET_COLS:
                frame[name] = np.zeros(count, dtype=np.float32)
            frame["y_long_expected_mae_bps"] = np.arange(count, dtype=np.float32)
            frame["y_dip_mfe_long_K12"] = -np.arange(count, dtype=np.float32)
            if full:
                import pyarrow as pa
                import pyarrow.parquet as pq
                _write_advanced_parquet(path)
                table = pq.read_table(path).take(pa.array(np.arange(count) % 3))
                for name in frame:
                    table = table.set_column(table.schema.get_field_index(name), name,
                                             pa.array(frame[name]))
                pq.write_table(table, path)
                manifest = json.loads(path.with_suffix(".manifest.json").read_text())
            else:
                frame.to_parquet(path, index=False)
                manifest = {"extra": {}}
            windows = {
                "train": {"start": start.isoformat(), "end": declared_end.isoformat()},
                "val": {"start": cutoff.isoformat(), "end": (end-pd.Timedelta(seconds=1)).isoformat()},
            }
            manifest["splits"] = windows
            manifest["extra"].update({
                **causal_m1_position_size_target_policy_contract(size),
                "rows": count, "diagnostic_outcome_policy_sha256": direction["policy_sha256"],
                "aux_head_target_contract": {
                    **fixed, "incomplete_tail_rows_total": fixed["max_future_horizon_bars"],
                    "candidate_rows_before_completeness": count,
                    "incomplete_candidate_rows_excluded": 0, "complete_rows_emitted": count,
                },
            })
            sources[split] = {
                "parquet": _binding(path),
                "manifest": _json(path.with_suffix(".manifest.json"), manifest),
                "physical_rows": count, "declared_window": windows[split],
                "clock_sha256": clock_hash(times.asi8),
            }
            frames[split] = frame
            horizon = direction["selected_direction_horizon_bars"]
            support = {}
            for h in set(fixed["future_horizon_bars_by_column"].values()) | {horizon}:
                closes = times + pd.Timedelta(minutes=5*(h+1))
                support[str(h)] = {
                    "rows": count, "all_before_or_at_split_boundary": True,
                    "first_support_close_utc": closes[0].isoformat(),
                    "last_support_close_utc": closes[-1].isoformat(),
                    "support_clock_sha256": clock_hash(closes.asi8),
                }
            proofs[split] = {
                "rows": count, "entry_clock_sha256": sources[split]["clock_sha256"],
                "exact_m1_fill_and_contiguous_outcome_rows": count,
                "fixed_target_columns": len(fixed["columns"]),
                "observed_m5_support_by_horizon": support,
                "policy_horizon_m5_bars": horizon, "policy_horizon_m1_bars": horizon*5,
                "direction_policy_sha256": direction["policy_sha256"],
                "position_size_policy_sha256": size["policy_sha256"],
                "supervision_boundary_exclusive_utc": (cutoff if split=="train" else end).isoformat(),
                "last_exact_m1_outcome_utc": (times[-1]+pd.Timedelta(minutes=5*(horizon+1))).isoformat(),
            }
        row_bindings = {}
        for name, rows in (("TRAIN_CALENDAR_PARENT_ROWS", np.arange(3, dtype=np.int64)),
                           ("CONTROL256_PARENT_ROWS", np.arange(256, dtype=np.int64))):
            p = tmp_path / (name+".npy"); np.save(p, rows); row_bindings[name] = _binding(p)
        proofs["val"]["control256"] = {
            "rows": 256, "physical_coordinate_source": "val", "all_support_checks_passed": True,
            "entry_clock_sha256": clock_hash(pd.DatetimeIndex(frames["val"].time.iloc[:256]).asi8),
        }
        design = {
            "schema_version": "gx1_frozen_chronological_learning_design_v1",
            "selection": {"control_entries": 256}, "budget": {"later_control_entries": 256},
            "calendar": {
                "physical_source_splits": {"train": "train", "control": "val"},
                "physical_coordinate_namespaces_are_separate": True, "source_bindings": sources,
                "train_entry_start_inclusive": start.isoformat(), "train_control_cutoff": cutoff.isoformat(),
                "development_control_entry_end_exclusive": end.isoformat(),
                "bindings": row_bindings, "control256": {"rows": 256},
            },
        }
        inputs = {"design": _json(tmp_path/"physical_design.json", design),
                  "control256": row_bindings["CONTROL256_PARENT_ROWS"],
                  "signal": _json(tmp_path/"signal.json", {"feature_ranking": {
                      "entry_direction_target_policy": direction,
                      "entry_direction_target_policy_sha256": direction["policy_sha256"],
                  }})}
        for split in sources:
            for kind in ("parquet", "manifest"):
                inputs[split+"_"+kind] = sources[split][kind]
        plan = {"schema_version": "gx1_native_v38_auxiliary_reuse_precheck_plan_v1",
                "input_bindings": inputs}
        result = {
            "schema_version": "gx1_native_v38_auxiliary_reuse_precheck_v1",
            "decision": "PASS_FROZEN_TRAIN_POLICIES_AND_COMPLETE_PRETEST_TARGET_CLOCK_SUPPORT",
            "identical_frozen_policies_across_physical_splits": True,
            "producer_code_unchanged_for_five_target_owners": True,
            "test_accessed": False, "forbidden_access_attempts": [],
            "fit_start_utc": direction["train_start_utc"], "fit_end_utc": direction["train_end_utc"],
            "plan": _json(tmp_path/"physical_plan.json", plan), "splits": proofs,
        }
        result_binding = _json(tmp_path/"physical_result.json", result)
        return dict(frames=frames, sources=sources, design=design, inputs=inputs, plan=plan,
                    result=result, result_binding=result_binding)
    return make


def _physical_bind(c, role="TRAIN", **overrides):
    split = "train" if role == "TRAIN" else "val"
    args = dict(parquet_path=Path(c["sources"][split]["parquet"]["path"]),
                result_binding=c["result_binding"], role=role,
                expected_design_sha256=c["inputs"]["design"]["sha256"])
    args.update(overrides)
    return trainer._physical_auxiliary_target_binding(c["frames"][split], **args)


@pytest.mark.parametrize("role", ["TRAIN", "CONTROL256"])
def test_physical_dataset_getitem_reuses_exact_original_targets(physical_case, tmp_path, monkeypatch, role):
    c = physical_case(full=True)
    split = "train" if role == "TRAIN" else "val"
    ds = trainer.EntryV10CtxDataset(
        Path(c["sources"][split]["parquet"]["path"]), seq_len=trainer.MODEL_NATIVE_SEQ_LEN,
        m5_prebuilt_path=install_multi_tf_stub(tmp_path, monkeypatch),
        per_tf_seq_lens=dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS), multi_tf_closed_bar=True,
    )
    ds.indices = np.array([2, 0, 1] if role == "TRAIN" else [255, 0, 256])
    before = ds.df
    outputs = [ds[i] for i in range(3)]
    arguments = dict(result_path=Path(c["result_binding"]["path"]),
        expected_result_sha256=c["result_binding"]["sha256"],
        expected_design_sha256=c["inputs"]["design"]["sha256"], role=role)
    with pytest.raises(RuntimeError, match="HASH_MISMATCH"):
        ds.bind_policy_dependent_auxiliary_targets(**{**arguments, "expected_result_sha256": "f"*64})
    assert ds.df is before and getattr(ds, "_policy_dependent_auxiliary_binding", None) is None
    ds.bind_policy_dependent_auxiliary_targets(**arguments)
    assert ds.df is before
    assert ds.indices.tolist() == ([2, 0, 1] if role=="TRAIN" else [255, 0, 256])
    for i in range(3 if role=="TRAIN" else 2):
        for key, value in ds[i].items():
            assert torch.equal(value, outputs[i][key]), key
    binding = ds._policy_dependent_auxiliary_binding
    assert binding["source_split"] == split and binding["targets_rewritten"] is False
    assert binding["active_target_columns"] == list(trainer._MODEL_NATIVE_ACTIVE_TARGET_COLS)
    assert trainer._sha256_file(ds.parquet_path) == c["sources"][split]["parquet"]["sha256"]
    if role == "CONTROL256":
        with pytest.raises(RuntimeError, match="UNBOUND_ROW"): ds[2]
    with pytest.raises(RuntimeError, match="MODE_INVALID"):
        ds.bind_policy_dependent_auxiliary_targets(**arguments)


def test_overlapping_integer_ids_are_distinct_physical_coordinates(physical_case):
    c = physical_case()
    train_mask, train = _physical_bind(c)
    val_mask, val = _physical_bind(c, "CONTROL256")
    assert np.flatnonzero(train_mask).tolist() == [0, 1, 2]
    assert np.flatnonzero(val_mask)[:3].tolist() == [0, 1, 2]
    assert train["parent_entry_parquet"] != val["parent_entry_parquet"]
    assert train["policies"] == val["policies"]


@pytest.mark.parametrize("fault", [
    "design_hash", "wrong_parent", "same_parent", "test_pointer", "physical_flag",
    "clock_order", "clock_value", "naive_clock", "population", "double_precision",
    "nonfinite", "fractional_mask", "negative_mae", "signed_mfe_upper", "mask_sentinel",
    "incomplete_support", "future_support", "missing_horizon", "wrong_policy_hash",
    "fit_cutoff", "control_source", "control_clock", "control_size", "rows_reverse",
    "train_subset", "manifest_hash", "result_failure", "policy_lineage",
])
def test_physical_target_binding_rejects_mismatch_without_mutation(physical_case, monkeypatch, fault):
    c = physical_case()
    role = "CONTROL256" if fault.startswith("control_") else "TRAIN"
    split = "val" if role == "CONTROL256" else "train"
    args = {}
    frame = c["frames"][split]
    if fault == "design_hash": args["expected_design_sha256"] = "f"*64
    elif fault == "wrong_parent": args["parquet_path"] = Path(c["sources"]["val"]["parquet"]["path"])
    elif fault == "same_parent":
        c["sources"]["val"]["parquet"] = c["sources"]["train"]["parquet"]
        c["inputs"]["val_parquet"] = c["sources"]["train"]["parquet"]
    elif fault == "test_pointer":
        c["inputs"]["train_parquet"] = {"path": "/DO_NOT_ACCESS_test.parquet", "sha256": "f"*64}
    elif fault == "physical_flag": c["design"]["calendar"]["physical_coordinate_namespaces_are_separate"] = False
    elif fault == "clock_order": c["frames"][split] = frame.iloc[::-1]
    elif fault == "clock_value": frame.loc[0, "time"] += pd.Timedelta(seconds=1)
    elif fault == "naive_clock": frame["time"] = frame.time.dt.tz_localize(None)
    elif fault == "population": c["frames"][split] = frame.iloc[:-1]
    elif fault == "double_precision": frame["y_long_expected_mae_bps"] = frame.y_long_expected_mae_bps.astype(np.float64)
    elif fault == "nonfinite": frame.loc[0, "y_long_expected_mae_bps"] = np.nan
    elif fault == "fractional_mask": frame.loc[0, "y_position_size_mask"] = 0.5
    elif fault == "negative_mae": frame.loc[0, "y_long_expected_mae_bps"] = -1
    elif fault == "signed_mfe_upper": frame.loc[0, "y_dip_mfe_long_K12"] = 1001
    elif fault == "mask_sentinel": frame.loc[0, "y_position_size_target"] = 0.5
    elif fault == "incomplete_support": c["result"]["splits"]["train"]["exact_m1_fill_and_contiguous_outcome_rows"] -= 1
    elif fault == "future_support": c["result"]["splits"]["train"]["observed_m5_support_by_horizon"]["96"]["last_support_close_utc"] = "2027-01-01T00:00Z"
    elif fault == "missing_horizon": del c["result"]["splits"]["train"]["observed_m5_support_by_horizon"]["96"]
    elif fault == "wrong_policy_hash": c["result"]["splits"]["train"]["position_size_policy_sha256"] = "f"*64
    elif fault == "fit_cutoff": c["result"]["fit_end_utc"] = c["design"]["calendar"]["train_control_cutoff"]
    elif fault == "control_source": c["result"]["splits"]["val"]["control256"]["physical_coordinate_source"] = "train"
    elif fault == "control_clock": c["result"]["splits"]["val"]["control256"]["entry_clock_sha256"] = "f"*64
    elif fault == "control_size": c["result"]["splits"]["val"]["control256"]["rows"] = 255
    elif fault in ("rows_reverse", "train_subset"):
        b = c["design"]["calendar"]["bindings"]["TRAIN_CALENDAR_PARENT_ROWS"]
        np.save(b["path"], np.array([2, 1, 0] if fault=="rows_reverse" else [0, 2], dtype=np.int64))
        b.update(_binding(Path(b["path"])))
    elif fault == "manifest_hash":
        c["sources"]["train"]["manifest"]["sha256"] = "f"*64
    elif fault == "result_failure": c["result"]["decision"] = "FAIL"
    elif fault == "policy_lineage":
        p = Path(c["sources"]["train"]["manifest"]["path"])
        manifest = json.loads(p.read_text())
        manifest["extra"]["entry_causal_m1_position_size_target_policy"]["entry_causal_m1_target_policy_sha256"] = "f"*64
        c["sources"]["train"]["manifest"].update(_json(p, manifest))
    c["inputs"]["design"] = _json(Path(c["inputs"]["design"]["path"]), c["design"])
    c["result"]["plan"] = _json(Path(c["result"]["plan"]["path"]), c["plan"])
    c["result_binding"] = _json(Path(c["result_binding"]["path"]), c["result"])
    touched = []
    original = trainer._sequence_source_exact_regular_file
    def guarded(path, **kw):
        touched.append(str(path))
        assert "DO_NOT_ACCESS" not in str(path), "forbidden pointer followed"
        return original(path, **kw)
    monkeypatch.setattr(trainer, "_sequence_source_exact_regular_file", guarded)
    before = c["frames"][split].copy(deep=True)
    with pytest.raises(RuntimeError):
        _physical_bind(c, role, **args)
    pd.testing.assert_frame_equal(c["frames"][split], before, check_exact=True)
    if fault in ("wrong_parent", "same_parent", "test_pointer", "physical_flag"):
        assert not any(p.endswith(".parquet") or p.endswith(".manifest.json") for p in touched)
