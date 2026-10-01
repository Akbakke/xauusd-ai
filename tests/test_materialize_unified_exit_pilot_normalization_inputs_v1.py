from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.unified_exit_market_closure_authority_v1 import (
    build_market_closure_authority,
    seal_exact_market_schedule,
)
from gx1.scripts.materialize_unified_exit_pilot_normalization_inputs_v1 import (
    EXPECTED_SUMMARY_REGISTRY_SCHEMA,
    _canonical_sha256,
    _merge_intervals,
    _sha256_file,
    build_child_sequence_reconstruction_audit,
    build_normalization_inputs,
    build_train_normalization_population_witness,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _fixed(values: np.ndarray, width: int, arrow_type: pa.DataType) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(
        pa.array(np.asarray(values).reshape(-1), type=arrow_type), width
    )


def _write_surface(
    path: Path,
    times: pd.DatetimeIndex,
    signal: np.ndarray,
) -> None:
    rows = len(times)
    ctx_cont = np.arange(rows * MODEL_NATIVE_CTX_CONT_DIM, dtype=np.float32).reshape(
        rows, MODEL_NATIVE_CTX_CONT_DIM
    )
    ctx_cat = np.zeros((rows, MODEL_NATIVE_CTX_CAT_DIM), dtype=np.int64)
    table = pa.table(
        {
            "time": pa.array(times),
            "signal": _fixed(signal, MODEL_NATIVE_SIGNAL_DIM, pa.float32()),
            "ctx_cont": _fixed(ctx_cont, MODEL_NATIVE_CTX_CONT_DIM, pa.float32()),
            "ctx_cat": _fixed(ctx_cat, MODEL_NATIVE_CTX_CAT_DIM, pa.int64()),
        }
    )
    pq.write_table(table, path)


def _sequence_fixture(tmp_path: Path) -> dict[str, Any]:
    times = pd.date_range("2025-05-31T16:00:00Z", periods=120, freq="5min")
    signal = np.arange(len(times) * MODEL_NATIVE_SIGNAL_DIM, dtype=np.float32).reshape(
        len(times), MODEL_NATIVE_SIGNAL_DIM
    )
    surface = tmp_path / "m5_surface.parquet"
    _write_surface(surface, times, signal)
    surface_manifest = tmp_path / "m5_surface.manifest.json"
    _write_json(
        surface_manifest,
        {
            "output_parquet": str(surface),
            "output_parquet_sha256": _sha256_file(surface),
            "rows": len(times),
            "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
            "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
            "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
            "schema_version": "synthetic_m5_surface_v1",
        },
    )

    positions = np.asarray([95, 96], dtype=np.int64)
    sequence = np.stack(
        [signal[position - 95 : position + 1] for position in positions]
    )
    snapshot = signal[positions]
    child_path = tmp_path / "train.parquet"
    pq.write_table(
        pa.table(
            {
                "time": pa.array(times[positions]),
                "seq": pa.FixedSizeListArray.from_arrays(
                    _fixed(
                        sequence.reshape(-1, MODEL_NATIVE_SIGNAL_DIM),
                        MODEL_NATIVE_SIGNAL_DIM,
                        pa.float32(),
                    ),
                    MODEL_NATIVE_SEQ_LEN,
                ),
                "snap": _fixed(snapshot, MODEL_NATIVE_SIGNAL_DIM, pa.float32()),
            }
        ),
        child_path,
    )
    child_manifest = tmp_path / "train.manifest.json"
    _write_json(child_manifest, {"child": True})
    binding = {
        "dataset_run_id": "PARENT",
        "inline_split_recomputation": False,
        "manifest_path": str(surface_manifest),
        "manifest_sha256": _sha256_file(surface_manifest),
        "pair_generation_id": "pair",
        "path": str(surface),
        "rows": len(times),
        "schema_version": "synthetic_m5_surface_v1",
        "sha256": _sha256_file(surface),
        "signal_manifest_sha256": "1" * 64,
        "time_alignment": "exact_entry_m5_source_timeline",
    }
    parent = {
        "extra": {
            "signal_bridge": {
                "seq_structure_extension_v1": {"feature_surface": binding}
            }
        }
    }
    child = {
        "child_dataset_run_id": "CHILD",
        "parent_dataset_run_id": "PARENT",
        "witness_sha256": "2" * 64,
        "splits": {
            "train": {
                "parquet_path": str(child_path),
                "parquet_sha256": _sha256_file(child_path),
                "manifest_path": str(child_manifest),
                "manifest_file_sha256": _sha256_file(child_manifest),
                "rows": 2,
                "clock_sha256": hashlib.sha256(
                    np.asarray(times[positions].asi8, dtype="<i8").tobytes()
                ).hexdigest(),
            }
        },
    }
    parent_audit_path = tmp_path / "parent_sequence_audit.json"
    _write_json(parent_audit_path, {"parent": "verified"})
    return {
        "times": times,
        "signal": signal,
        "surface": surface,
        "surface_manifest": surface_manifest,
        "child_path": child_path,
        "child_manifest": child_manifest,
        "parent": parent,
        "parent_audit_path": parent_audit_path,
        "child": child,
    }


def test_child_sequence_audit_is_exhaustive_and_rejects_changed_value(
    tmp_path: Path,
) -> None:
    fixture = _sequence_fixture(tmp_path)
    audit = build_child_sequence_reconstruction_audit(
        child_admission=fixture["child"],
        child_admission_file_sha256="3" * 64,
        parent_manifest=fixture["parent"],
        parent_sequence_audit={"parent": "verified"},
        parent_sequence_audit_path=fixture["parent_audit_path"],
    )
    assert audit["decision"] == "PASS"
    assert audit["child_train_rows"] == 2
    assert audit["sequence_shape"] == [
        2,
        MODEL_NATIVE_SEQ_LEN,
        MODEL_NATIVE_SIGNAL_DIM,
    ]
    assert audit["val_rows_scanned"] == 0
    assert audit["test_rows_scanned"] == 0

    table = pq.read_table(fixture["child_path"])
    snap = table.column("snap").combine_chunks().values.to_numpy().copy()
    snap[0] += np.float32(1.0)
    changed = table.set_column(
        table.schema.get_field_index("snap"),
        "snap",
        pa.chunked_array(
            [_fixed(snap.reshape(2, -1), MODEL_NATIVE_SIGNAL_DIM, pa.float32())]
        ),
    )
    pq.write_table(changed, fixture["child_path"])
    with pytest.raises(RuntimeError, match="SEQUENCE_VALUE_MISMATCH"):
        build_child_sequence_reconstruction_audit(
            child_admission=fixture["child"],
            child_admission_file_sha256="3" * 64,
            parent_manifest=fixture["parent"],
            parent_sequence_audit={"parent": "verified"},
            parent_sequence_audit_path=fixture["parent_audit_path"],
        )


def _population_fixture(
    tmp_path: Path,
) -> dict[str, Any]:
    fixture = _sequence_fixture(tmp_path)
    m1_times = pd.date_range(fixture["times"][0], periods=600, freq="1min")
    parent_m1_source = tmp_path / "parent_m1.parquet"
    pd.DataFrame({"time": m1_times}).to_parquet(parent_m1_source, index=False)
    parent_m1_manifest = tmp_path / "parent_m1.manifest.json"
    _write_json(parent_m1_manifest, {"source": "synthetic_parent"})
    m1_source = tmp_path / "m1.parquet"
    pd.DataFrame({"time": m1_times}).to_parquet(m1_source, index=False)
    m1_manifest = tmp_path / "m1.manifest.json"
    _write_json(
        m1_manifest,
        {
            "schema_version": "gx1_unified_exit_pilot_m1_child_view_v1",
            "split": "train",
            "decision": "PASS",
            "output_parquet": str(m1_source),
            "output_parquet_sha256": _sha256_file(m1_source),
            "parent_m1_path": str(parent_m1_source),
            "parent_m1_sha256": _sha256_file(parent_m1_source),
            "parent_m1_manifest_path": str(parent_m1_manifest),
            "parent_m1_manifest_sha256": _sha256_file(parent_m1_manifest),
            "right_censor_time_utc_exclusive": "2025-06-02T00:00:00Z",
            "test_accessed": False,
        },
    )

    m1_signal = np.arange(
        len(m1_times) * MODEL_NATIVE_SIGNAL_DIM, dtype=np.float32
    ).reshape(len(m1_times), MODEL_NATIVE_SIGNAL_DIM)
    m1_feature = tmp_path / "m1_feature.parquet"
    _write_surface(m1_feature, m1_times, m1_signal)
    m1_feature_manifest = tmp_path / "m1_feature.manifest.json"
    feature_manifest = {
        "output_parquet": str(m1_feature),
        "output_parquet_sha256": _sha256_file(m1_feature),
        "alignment_parquet": str(parent_m1_source),
        "alignment_sha256": _sha256_file(parent_m1_source),
        "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "rows": len(m1_times),
        "feature_field_order_sha256": "4" * 64,
    }
    _write_json(m1_feature_manifest, feature_manifest)

    schedule_path = tmp_path / "schedule.json"
    schedule = seal_exact_market_schedule(
        {
            "schema_version": "gx1_xau_exact_market_closure_schedule_v1",
            "decision": "PASS",
            "instrument": "XAU_USD",
            "timeframe": "M1",
            "coverage_start_utc": m1_times[0].isoformat(),
            "coverage_end_utc_exclusive": (
                m1_times[-1] + pd.Timedelta(minutes=1)
            ).isoformat(),
            "interval_semantics": "left_closed_right_open_utc",
            "source_method": ("externally_sourced_exact_xau_utc_closure_intervals_v1"),
            "source_reference_sha256": "5" * 64,
            "intervals": [],
            "test_data_used": False,
        }
    )
    _write_json(schedule_path, schedule)
    closure = build_market_closure_authority(
        m1_times=m1_times,
        m1_source_path=m1_source,
        m1_source_sha256=_sha256_file(m1_source),
        m1_source_manifest_path=m1_manifest,
        m1_source_manifest_sha256=_sha256_file(m1_manifest),
        exact_schedule=schedule,
        exact_schedule_path=schedule_path,
        exact_schedule_file_sha256=_sha256_file(schedule_path),
    )
    closure_path = tmp_path / "closure.json"
    _write_json(closure_path, closure)

    fixture["child"]["m1_source_binding"] = {
        "parquet_path": str(parent_m1_source),
        "parquet_sha256": _sha256_file(parent_m1_source),
        "manifest_path": str(parent_m1_manifest),
        "manifest_sha256": _sha256_file(parent_m1_manifest),
    }
    mtf_manifest = tmp_path / "mtf.json"
    _write_json(mtf_manifest, {"cache": "bound"})
    mtf = {
        "cache_dir": str(tmp_path / "cache"),
        "manifest_path": str(mtf_manifest),
        "manifest_sha256": _sha256_file(mtf_manifest),
        "cache_identity_sha256": "6" * 64,
    }
    sequence_audit = {"contract_sha256": "7" * 64}
    kwargs = dict(
        child_admission=fixture["child"],
        child_admission_file_sha256="3" * 64,
        child_sequence_audit=sequence_audit,
        m1_source_path=m1_source,
        m1_source_manifest_path=m1_manifest,
        m1_feature_base_path=m1_feature,
        m1_feature_base_manifest_path=m1_feature_manifest,
        market_closure_authority_path=closure_path,
        parent_manifest=fixture["parent"],
        mtf_cache_binding=mtf,
        mtf_cache_manifest_path=mtf_manifest,
        train_end="2025-06-02T00:00:00Z",
    )
    return fixture, kwargs, closure


def test_population_witness_scans_unique_rows_without_sampler_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture, kwargs, closure = _population_fixture(tmp_path)
    from gx1.scripts import materialize_unified_exit_pilot_normalization_inputs_v1 as owner
    def no_full_signal(*args, **kwargs):
        raise AssertionError("population needs only the validated M5 clock")
    monkeypatch.setattr(owner, "_load_surface_signal", no_full_signal)
    closure_path = kwargs["market_closure_authority_path"]
    witness = build_train_normalization_population_witness(**kwargs)
    assert witness["decision"] == "PASS"
    assert witness["train_entry_decision_rows"] == 2
    assert witness["entry_m5_local_unique_rows"] == 97
    assert witness["exit_m1_current_unique_rows"] == 120
    encoded = json.dumps(witness, sort_keys=True)
    for forbidden in ("chunk_count", "sample_index", "reward", "target_q"):
        assert forbidden not in encoded
    assert witness["val_fit_rows"] == 0
    assert witness["test_fit_rows"] == 0

    bad = dict(closure)
    bad["m1_source_sha256"] = "8" * 64
    bad["artifact_sha256"] = _canonical_sha256(
        {key: value for key, value in bad.items() if key != "artifact_sha256"}
    )
    _write_json(closure_path, bad)
    with pytest.raises(RuntimeError, match="MARKET_CLOSURE_AUTHORITY_INVALID"):
        build_train_normalization_population_witness(**kwargs)


@pytest.mark.parametrize("mode", ["validate", "publish", "concurrent_directory"])
def test_view_binds_registry_and_publishes_without_replacing_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    output = tmp_path / "pilot" / "NORMALIZATION_INPUTS"
    child_manifest = tmp_path / "child.manifest.json"
    parent_manifest_path = tmp_path / "parent.manifest.json"
    _write_json(
        child_manifest,
        {
            "source_manifest": {
                "path": str(parent_manifest_path),
                "sha256": "9" * 64,
            }
        },
    )
    _write_json(parent_manifest_path, {"parent": True})
    admission = {
        "child_dataset_run_id": "CHILD",
        "witness_sha256": "a" * 64,
        "splits": {
            "train": {
                "manifest_path": str(child_manifest),
                "rows": 2,
            }
        },
    }
    parent = {
        "feature_contract": {"signal": "bound"},
        "extra": {
            "signal_bridge": {
                "seq_structure_extension_v1": {
                    "feature_surface": {
                        "dataset_run_id": "PARENT",
                        "inline_split_recomputation": False,
                        "manifest_path": "/surface.json",
                        "manifest_sha256": "b" * 64,
                        "pair_generation_id": "pair",
                        "path": "/surface.parquet",
                        "rows": 100,
                        "schema_version": "surface",
                        "sha256": "c" * 64,
                        "signal_manifest_sha256": "d" * 64,
                        "time_alignment": "exact_entry_m5_source_timeline",
                    }
                }
            }
        },
    }
    mtf = {"manifest_path": "/mtf", "cache_identity_sha256": "e" * 64}
    monkeypatch.setattr(
        "gx1.scripts.materialize_unified_exit_pilot_normalization_inputs_v1."
        "_require_child_admission",
        lambda *_args, **_kwargs: (admission, "f" * 64),
    )
    monkeypatch.setattr(
        "gx1.scripts.materialize_unified_exit_pilot_normalization_inputs_v1."
        "_parent_sources",
        lambda *_args, **_kwargs: (parent, {"audit": True}, mtf),
    )
    monkeypatch.setattr(
        "gx1.scripts.materialize_unified_exit_pilot_normalization_inputs_v1."
        "build_child_sequence_reconstruction_audit",
        lambda **_kwargs: {"contract_sha256": "1" * 64},
    )
    monkeypatch.setattr(
        "gx1.scripts.materialize_unified_exit_pilot_normalization_inputs_v1."
        "build_train_normalization_population_witness",
        lambda **_kwargs: {"contract_sha256": "2" * 64},
    )

    arguments = dict(
        pilot_root=tmp_path / "pilot",
        output_dir=output,
        child_admission_path=tmp_path / "admission.json",
        parent_sequence_audit_path=tmp_path / "parent_audit.json",
        m1_source_path=tmp_path / "m1.parquet",
        m1_source_manifest_path=tmp_path / "m1.json",
        m1_feature_base_path=tmp_path / "m1_feature.parquet",
        m1_feature_base_manifest_path=tmp_path / "m1_feature.json",
        mtf_cache_manifest_path=tmp_path / "mtf.json",
        market_closure_authority_path=tmp_path / "closure.json",
        publish=mode != "validate",
        expected_train_rows=2,
        expected_val_rows=2,
    )
    from gx1.contracts.immutable_event_authority_v1 import ImmutableEventAuthorityError
    from gx1.scripts import materialize_unified_exit_pilot_normalization_inputs_v1 as owner

    if mode == "concurrent_directory":
        publish = owner._publish_file_noreplace
        winner_inode = []

        def publish_after_race(source: Path, destination: Path) -> None:
            destination.mkdir()
            winner_inode.append(destination.stat().st_ino)
            publish(source, destination)

        monkeypatch.setattr(owner, "_publish_file_noreplace", publish_after_race)
        with pytest.raises(ImmutableEventAuthorityError, match="already exists"):
            build_normalization_inputs(**arguments)
        assert output.stat().st_ino == winner_inode[0]
        assert list(output.iterdir()) == []
        assert list(output.parent.iterdir()) == [output]
        return
    report = build_normalization_inputs(**arguments)
    if mode == "publish":
        assert report["published"] is True
        assert sorted(path.name for path in output.iterdir()) == [
            "CHILD_NORMALIZATION_VIEW.json",
            "CHILD_TRAIN_SEQUENCE_RECONSTRUCTION_AUDIT.json",
            "TRAIN_NORMALIZATION_POPULATION_WITNESS.json",
        ]
        view = json.loads((output / "CHILD_NORMALIZATION_VIEW.json").read_text())
        assert view["normalization_fit_status"] == "READY_FOR_TRAIN_ONLY_FIT"
        assert view["final_normalization_published"] is False
        return

    view = report["normalization_view"]
    assert report["published"] is False
    from gx1.contracts.unified_exit_lifetime_summary_v1 import (
        lifetime_summary_registry,
    )

    registry = lifetime_summary_registry()
    assert view["decision"] == "PASS"
    assert view["blockers"] == []
    assert view["lifetime_summary_registry"] == {
        "schema_version": EXPECTED_SUMMARY_REGISTRY_SCHEMA,
        "status": "BOUND",
        "registry_sha256": registry["registry_sha256"],
        "field_order": registry["field_order"],
        "field_order_sha256": registry["field_order_sha256"],
        "dimension": registry["dimension"],
    }
    assert view["normalization_fit_status"] == "READY_FOR_TRAIN_ONLY_FIT"
    assert view["final_normalization_published"] is False
    assert view["first_state_witness_published"] is False
    assert not output.exists()


def test_interval_merge_is_deterministic() -> None:
    assert _merge_intervals([(5, 9), (0, 2), (2, 4), (12, 13)]) == [
        (0, 4),
        (5, 9),
        (12, 13),
    ]


def test_frozen_physical_population_replaces_only_implicit_legacy_row_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.scripts.materialize_unified_exit_pilot_normalization_inputs_v1 import _require_child_admission
    from tests.test_validate_lifecycle_v2_pilot_child_view_v1 import _published_frozen_source

    _, _, _, witness = _published_frozen_source(tmp_path, monkeypatch)
    path = tmp_path/"frozen-admission.json"
    _write_json(path, witness)
    accepted, digest = _require_child_admission(
        path, expected_train_rows=None, expected_val_rows=None,
    )
    assert accepted == witness
    assert digest == _sha256_file(path)
    with pytest.raises(RuntimeError, match="FROZEN_POPULATION_MISMATCH"):
        _require_child_admission(path, expected_train_rows=None, expected_val_rows=5508)
    with pytest.raises(RuntimeError, match="FROZEN_POPULATION_MISMATCH"):
        _require_child_admission(path, expected_train_rows=65295, expected_val_rows=None)


def test_legacy_normalization_still_requires_original_default_population(tmp_path: Path) -> None:
    from gx1.scripts.materialize_unified_exit_pilot_normalization_inputs_v1 import _require_child_admission
    path = tmp_path/"old-small-admission.json"
    value = {
        "schema_version": "gx1_lifecycle_v2_pilot_child_view_admission_v1",
        "decision": "PASS", "test_accessed": False,
        "splits": {"train": {"rows": 2}, "val": {"rows": 2}},
    }
    value["witness_sha256"] = _canonical_sha256(value)
    _write_json(path,value)
    with pytest.raises(RuntimeError, match="PILOT_NORMALIZATION_CHILD_ADMISSION_INVALID"):
        _require_child_admission(path, expected_train_rows=None, expected_val_rows=None)


def test_frozen_normalization_population_uses_calendar_end_before_source_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.scripts import materialize_unified_exit_pilot_normalization_inputs_v1 as owner
    from tests.test_validate_lifecycle_v2_pilot_child_view_v1 import _published_frozen_source

    _, _, _, witness = _published_frozen_source(tmp_path, monkeypatch)
    arguments = dict(
        child_admission=witness, child_admission_file_sha256="a"*64,
        child_sequence_audit={}, m1_source_path=tmp_path/"m1",
        m1_source_manifest_path=tmp_path/"manifest",
        m1_feature_base_path=tmp_path/"feature", m1_feature_base_manifest_path=tmp_path/"feature-manifest",
        market_closure_authority_path=tmp_path/"closure", parent_manifest={},
        mtf_cache_binding={}, mtf_cache_manifest_path=tmp_path/"mtf",
    )
    class ReachedSourceIO(Exception):
        pass
    monkeypatch.setattr(owner,"_exact_file",lambda *args: (_ for _ in ()).throw(ReachedSourceIO()))
    # The implicit and explicit correct cutoff both reach real input admission.
    for end in (None,"2025-06-01T00:00:00+00:00"):
        with pytest.raises(ReachedSourceIO):
            owner.build_train_normalization_population_witness(**arguments,train_end=end)
    # The previous default disagrees with the frozen TRAIN boundary.
    with pytest.raises(RuntimeError,match="FROZEN_TRAIN_END_MISMATCH"):
        owner.build_train_normalization_population_witness(
            **arguments,train_end="2026-06-01T00:00:00+00:00",
        )


@pytest.mark.parametrize("corruption", ["none", "rows", "hash", "bytes", "record", "clock"])
def test_adopted_full_split_reuses_only_exact_valid_parent_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str,
) -> None:
    from gx1.scripts import materialize_unified_exit_pilot_normalization_inputs_v1 as owner
    from gx1.scripts.audit_entry_sequence_source_reconstruction_v1 import audit_sequence_source_reconstruction

    fixture = _sequence_fixture(tmp_path)
    parent_manifest_path = tmp_path / "parent.manifest.json"
    _write_json(parent_manifest_path, fixture["parent"])
    parent_audit = audit_sequence_source_reconstruction(
        parquet_path=fixture["child_path"], manifest_path=parent_manifest_path,
    )
    _write_json(fixture["parent_audit_path"], parent_audit)
    _write_json(fixture["child_manifest"], {
        "source_manifest": {"path": str(parent_manifest_path), "sha256": _sha256_file(parent_manifest_path)},
    })
    child = fixture["child"]["splits"]["train"]
    child["manifest_file_sha256"] = _sha256_file(fixture["child_manifest"])
    if corruption == "rows":
        parent_audit["rows"] += 1
        _write_json(fixture["parent_audit_path"], parent_audit)
    elif corruption == "hash":
        parent_audit["parquet_sha256"] = "0" * 64
        _write_json(fixture["parent_audit_path"], parent_audit)
    elif corruption == "bytes":
        with fixture["child_path"].open("ab") as handle:
            handle.write(b"changed")
    elif corruption == "record":
        _write_json(fixture["parent_audit_path"], {"replaced": True})
    elif corruption == "clock":
        child["clock_sha256"] = "0" * 64

    def forbidden_signal_load(*args, **kwargs):
        raise AssertionError("full M5 signal must not be decoded again")
    monkeypatch.setattr(owner, "_load_surface_signal", forbidden_signal_load)
    original_parquet = pq.ParquetFile
    columns_read = []
    class ClockOnlyParquet(original_parquet):
        def iter_batches(self, *args, **kwargs):
            columns_read.append(kwargs["columns"])
            assert kwargs["columns"] == ["time"]
            yield from super().iter_batches(*args, **kwargs)
    monkeypatch.setattr(pq, "ParquetFile", ClockOnlyParquet)
    arguments = dict(
        child_admission=fixture["child"], child_admission_file_sha256="3" * 64,
        parent_manifest=fixture["parent"], parent_sequence_audit=parent_audit,
        parent_sequence_audit_path=fixture["parent_audit_path"],
    )
    if corruption != "none":
        with pytest.raises(RuntimeError):
            build_child_sequence_reconstruction_audit(**arguments)
    else:
        audit = build_child_sequence_reconstruction_audit(**arguments)
        assert audit["sequence_value_verification"] == "inherited_exact_full_parent_audit"
        assert audit["parent_sequence_source_chain_sha256"] == parent_audit["sequence_source_chain_sha256"]
        assert "sequence_value_stream_sha256" not in audit
        assert audit["child_train_rows"] == 2
        assert columns_read == [["time"]]
