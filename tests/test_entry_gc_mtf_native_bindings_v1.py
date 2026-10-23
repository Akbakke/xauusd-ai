"""Synthetic224 artifact/native joins; no genuine files, weights or forwards."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.contracts import entry_gc_mtf_history_v1 as history
from gx1.contracts import unified_exit_native_candidate_campaign_v1 as campaign
from gx1.features import htf_features as htf
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.scripts import run_unified_exit_random_access_fixed_step_v1 as factory
from gx1.scripts import run_unified_exit_random_access_full_train_v1 as native
from tests.test_entry_gc_mtf_history_v1 import gc_history_fixture

IDENTITY_KEY = "gc_mtf_history_identity"


def _ref(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return _ref(path)


@pytest.fixture(scope="module")
def native_history_case():
    return gc_history_fixture()


@pytest.fixture
def artifact(tmp_path, native_history_case):
    bound, surfaces, _selected, _start, _end = native_history_case
    manifest = {
        "schema_version": "gx1_gc_mtf_history_physical_surfaces_v1",
        "layout": _write_json(tmp_path / "LAYOUT.json", bound["layout"]),
        "normalization": _write_json(tmp_path / "NORMALIZATION.json", bound["normalization"]),
        "CONTROL_fit_rows": 0, "TEST_fit_rows": 0, "predictive_value_measured": False,
        "surfaces": {},
    }
    for tf, surface in surfaces.items():
        manifest["surfaces"][tf] = {}
        for key in ("native_row_indices", "available_ns", "features"):
            path = tmp_path / f"GC_{tf}_{key}.npy"
            value = surface[key]
            np.save(path, value, allow_pickle=False)
            manifest["surfaces"][tf][key] = {
                **_ref(path), "size_bytes": path.stat().st_size,
                "dtype": value.dtype.name, "shape": list(value.shape),
            }
    binding = _write_json(tmp_path / "SURFACES.json", manifest)
    identity = {"binding": binding, "layout_sha256": bound["layout"]["contract_sha256"],
                "normalization_sha256": bound["normalization"]["contract_sha256"]}
    return {"binding": binding, "identity": identity, "manifest": manifest,
            "history": bound, "surfaces": surfaces}


def _owner(artifact):
    return SimpleNamespace(gc_mtf_history=artifact["history"],
                           gc_mtf_history_identity=deepcopy(artifact["identity"]),
                           entry_training_phase=trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)


def test_metadata_reader_opens_only_three_json_files_and_never_arrays(artifact, monkeypatch):
    allowed = {artifact["binding"]["path"], artifact["manifest"]["layout"]["path"],
               artifact["manifest"]["normalization"]["path"]}
    opened = []
    original = os.open

    def metadata_only(path, *args, **kwargs):
        assert os.fspath(path) in allowed, "metadata reader tried opening a non-metadata file"
        opened.append(os.fspath(path))
        return original(path, *args, **kwargs)

    def forbidden(*_args, **_kwargs):
        pytest.fail("metadata-only reader reached an NPY/cache loader")

    monkeypatch.setattr(os, "open", metadata_only)
    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(htf, "_load_verified_cache_npy", forbidden)
    result = history.read_gc_mtf_history_artifact_metadata(artifact["binding"])
    assert len(opened) == 3 and set(opened) == allowed
    assert set(result) == {"history", "identity", "manifest"}
    assert result["history"] == artifact["history"]
    assert result["identity"] == artifact["identity"]
    assert result["manifest"] == artifact["manifest"]
    with pytest.raises(RuntimeError, match="ARTIFACT_JSON_HASH"):
        history.read_gc_mtf_history_artifact_metadata({**artifact["binding"], "sha256": "0" * 64})


def test_artifact_loader_reuses_fifteen_verified_arrays_and_returns_exact_readonly_surfaces(artifact, monkeypatch):
    original = htf._load_verified_cache_npy
    calls = []

    def observed(directory_fd, name, **kwargs):
        calls.append((name, kwargs))
        return original(directory_fd, name, **kwargs)

    monkeypatch.setattr(htf, "_load_verified_cache_npy", observed)
    loaded = history.load_gc_mtf_history_artifact(artifact["binding"])
    expected = {Path(d["path"]).name: d for row in artifact["manifest"]["surfaces"].values() for d in row.values()}
    assert len(calls) == 15 and {name for name, _kwargs in calls} == set(expected)
    for name, kwargs in calls:
        assert kwargs["expected_sha256"] == expected[name]["sha256"]
        assert kwargs["expected_size_bytes"] == expected[name]["size_bytes"]
    assert loaded["history"] == artifact["history"] and loaded["identity"] == artifact["identity"]
    for tf, surface in loaded["surfaces"].items():
        assert surface["field_names"] == artifact["history"]["layout"]["field_names"][tf]
        for key in ("native_row_indices", "available_ns", "features"):
            np.testing.assert_array_equal(surface[key], artifact["surfaces"][tf][key])
            assert not surface[key].flags.writeable
        with pytest.raises(ValueError):
            surface["features"][0, 0] = 123.0


@pytest.mark.parametrize("fault", ["hash", "shape", "clock"])
def test_artifact_loader_rejects_payload_or_clock_tamper_even_with_rebound_metadata(artifact, fault):
    manifest = deepcopy(artifact["manifest"])
    key = "available_ns" if fault == "clock" else "features"
    descriptor = manifest["surfaces"]["M5"][key]
    path = Path(descriptor["path"])
    changed = np.load(path, allow_pickle=False).copy()
    if fault == "shape":
        changed = changed[:-1]  # Descriptor stays internally consistent; loaded shape is wrong.
    elif fault == "clock":
        changed[0] += 1  # Correct file SHA and shape cannot authorize an off-grid close.
    else:
        changed[0, 0] += 1.0
    np.save(path, changed, allow_pickle=False)
    if fault != "hash":
        descriptor.update(_ref(path), size_bytes=path.stat().st_size)
    binding = _write_json(Path(artifact["binding"]["path"]), manifest)
    with pytest.raises(RuntimeError):
        history.load_gc_mtf_history_artifact(binding)


def test_native_identity_joins_model_train_control_and_rejects_partial_or_different_bindings(artifact):
    assert native._gc_mtf_constructor_metadata({}) is None
    assert native._gc_mtf_constructor_metadata({"gc_mtf_history_artifact": None}) is None
    for bad in (False, {}):
        with pytest.raises(RuntimeError, match="ARTIFACT_REF"): native._gc_mtf_constructor_metadata({"gc_mtf_history_artifact": bad})
    model, train, control = (_owner(artifact) for _ in range(3))
    assert trainer._gc_mtf_history_native_identity(model, train, control) == artifact["identity"]
    assert trainer._gc_mtf_history_native_identity(SimpleNamespace(), SimpleNamespace(), SimpleNamespace()) is None
    for index in range(3):
        owners = [_owner(artifact) for _ in range(3)]
        owners[index].gc_mtf_history_identity["binding"]["sha256"] = "0" * 64
        with pytest.raises(RuntimeError, match="GC_MTF_HISTORY_IDENTITY_MISMATCH"):
            trainer._gc_mtf_history_native_identity(*owners)
    for partial in (SimpleNamespace(gc_mtf_history=artifact["history"]),
                    SimpleNamespace(gc_mtf_history_identity=artifact["identity"])):
        with pytest.raises(RuntimeError, match="GC_MTF_HISTORY_IDENTITY"):
            trainer._gc_mtf_history_native_identity(model, partial, control)
    train.gc_mtf_history_identity["normalization_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="ARTIFACT_IDENTITY_HISTORY"):
        trainer._gc_mtf_history_native_identity(model, train, control)


def test_native_factory_cannot_silently_ignore_gc_artifact_without_explicit_wrapper(artifact, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("missing GC wrapper reached model allocation")
    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)
    with pytest.raises(RuntimeError, match="GC_MTF_HISTORY"):
        factory._model({"gc_mtf_history_artifact": artifact["binding"]}, {}, torch.device("cpu"),
                       entry_training_phase=trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)


@pytest.mark.parametrize("fault", ["initial_metadata", "saved_state"])
def test_fresh_state_identity_mismatch_fails_before_checkpoint_state_is_consumed(artifact, monkeypatch, fault):
    model, train, control = (_owner(artifact) for _ in range(3))
    components = {"model": model, "train_ds": train, "val_ds": control,
                  "dynamic_m5_loss_scales": {}, "model_functions": trainer._PREFIX_CURRENT_MODEL_FUNCTIONS}
    initial = {IDENTITY_KEY: deepcopy(artifact["identity"]), "dynamic_m5_loss_scales": {},
               "initial_state": {"path": "/unused.pt", "sha256": "a" * 64},
               "online_model_state_sha256": "a" * 64, "model_functions": trainer._PREFIX_CURRENT_MODEL_FUNCTIONS}
    different = deepcopy(artifact["identity"])
    different["binding"]["sha256"] = "0" * 64
    reads = []

    def fake_load(*_args, **_kwargs):
        reads.append(True)
        assert fault == "saved_state", "initial metadata mismatch reached torch.load"
        return {IDENTITY_KEY: different, "model_functions": trainer._PREFIX_CURRENT_MODEL_FUNCTIONS}

    if fault == "initial_metadata":
        initial[IDENTITY_KEY] = different
    monkeypatch.setattr(native, "_bound_artifact", lambda _binding: Path("/unused.pt"))
    monkeypatch.setattr(torch, "load", fake_load)
    with pytest.raises(RuntimeError, match="GC_HISTORY_IDENTITY_MISMATCH"):
        native._restore_prefix_initial_measurement_state(
            components=components, scope={"initialization": initial}, device=torch.device("cpu"))
    assert len(reads) == int(fault == "saved_state")


def test_measurement_identity_mismatch_precedes_cohort_or_dataframe_reads(artifact, monkeypatch):
    from gx1.contracts import unified_exit_bounded_val_cohort_v1 as cohorts
    model, dataset = _owner(artifact), _owner(artifact)
    different = deepcopy(artifact["identity"])
    different["binding"]["sha256"] = "0" * 64

    def forbidden(*_args, **_kwargs):
        pytest.fail("GC mismatch reached cohort/dataframe access")

    monkeypatch.setattr(cohorts, "require_bounded_val_cohort", forbidden)
    with pytest.raises(RuntimeError, match="GC_HISTORY_IDENTITY_MISMATCH"):
        native._require_dynamic_m5_measurement_source(
            model=model, dataset=dataset, cohort=None, context={IDENTITY_KEY: different})


def test_measurement_records_require_exact_identity_and_keep_old_off_records_unchanged(artifact):
    record = {IDENTITY_KEY: deepcopy(artifact["identity"])}
    assert campaign._require_gc_mtf_history_identity_records(None, {}, {}) == {}
    assert campaign._require_gc_mtf_history_identity_records(artifact["identity"], record, deepcopy(record)) == record
    different = deepcopy(record)
    different[IDENTITY_KEY]["binding"]["sha256"] = "0" * 64
    for expected, records in ((None, ({IDENTITY_KEY: None},)),
                              (artifact["identity"], ({},)),
                              (artifact["identity"], (record, different))):
        with pytest.raises(RuntimeError, match="GC_MTF_HISTORY_IDENTITY"):
            campaign._require_gc_mtf_history_identity_records(expected, *records)
