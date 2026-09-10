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


def test_population_witness_scans_unique_rows_without_sampler_keys(
    tmp_path: Path,
) -> None:
    fixture = _sequence_fixture(tmp_path)
    m1_times = pd.date_range(fixture["times"][0], periods=600, freq="1min")
    m1_source = tmp_path / "m1.parquet"
    pd.DataFrame({"time": m1_times}).to_parquet(m1_source, index=False)
    m1_manifest = tmp_path / "m1.manifest.json"
    _write_json(m1_manifest, {"source": "synthetic"})

    m1_signal = np.arange(
        len(m1_times) * MODEL_NATIVE_SIGNAL_DIM, dtype=np.float32
    ).reshape(len(m1_times), MODEL_NATIVE_SIGNAL_DIM)
    m1_feature = tmp_path / "m1_feature.parquet"
    _write_surface(m1_feature, m1_times, m1_signal)
    m1_feature_manifest = tmp_path / "m1_feature.manifest.json"
    feature_manifest = {
        "output_parquet": str(m1_feature),
        "output_parquet_sha256": _sha256_file(m1_feature),
        "alignment_parquet": str(m1_source),
        "alignment_sha256": _sha256_file(m1_source),
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
        "parquet_path": str(m1_source),
        "parquet_sha256": _sha256_file(m1_source),
        "manifest_path": str(m1_manifest),
        "manifest_sha256": _sha256_file(m1_manifest),
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
    witness = build_train_normalization_population_witness(
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
        build_train_normalization_population_witness(
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


def test_view_is_explicitly_blocked_until_summary_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

    report = build_normalization_inputs(
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
        publish=False,
        expected_train_rows=2,
        expected_val_rows=2,
    )
    view = report["normalization_view"]
    assert report["published"] is False
    assert view["decision"] == "BLOCKED"
    assert view["blockers"] == ["lifetime_summary_registry_pending"]
    assert view["lifetime_summary_registry"] == {
        "schema_version": EXPECTED_SUMMARY_REGISTRY_SCHEMA,
        "status": "PENDING",
        "contract_sha256": None,
        "field_order_sha256": None,
    }
    assert view["final_normalization_published"] is False
    assert view["first_state_witness_published"] is False
    assert not output.exists()


def test_interval_merge_is_deterministic() -> None:
    assert _merge_intervals([(5, 9), (0, 2), (2, 4), (12, 13)]) == [
        (0, 4),
        (5, 9),
        (12, 13),
    ]
