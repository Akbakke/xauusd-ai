from __future__ import annotations

import ast
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from gx1.contracts import local_random_access_campaign_v2 as owner
from gx1.contracts.entry_exit_production_architecture_v1 import PRODUCTION_MTF_PER_TF_WINDOW_BARS
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SEQ_LEN
from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE as PHASE
from tests.test_native_full_train_file_verification_scope_v1 import (
    SOURCE, _preflight_fixture, physical_hash_calls,
)


class _MetadataFailure(RuntimeError):
    pass


class _ExecutionBoundaryStop(RuntimeError):
    pass


def _load_dynamic_builder():
    """Use the actual function and physical-file set without importing the trainer."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    function = next(node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_dynamic_m5_entry_components")
    files = next(node for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_DYNAMIC_M5_DATA_FILES"
                for target in node.targets))
    scope_import = next(node for node in tree.body if isinstance(node, ast.ImportFrom)
        and node.module == "gx1.contracts.local_random_access_campaign_v2"
        and any(alias.name == "file_verification_scope" for alias in node.names))
    isolated = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        scope_import, files, function,
    ], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace = {"Mapping": Mapping}
    exec(compile(isolated, str(SOURCE), "exec"), namespace)
    return namespace[function.name], namespace


@pytest.mark.parametrize("boundary", ["dynamic_dataset", "initial_restore"])
@pytest.mark.parametrize("fail_metadata", [False, True])
def test_dynamic_metadata_scope_reuses_real_hashes_and_resets_before_execution(
    tmp_path, monkeypatch, physical_hash_calls, boundary, fail_metadata,
):
    # Only semantic/model boundaries are isolated. ContextVar and all counted
    # SHA streams use their genuine source owners and actual small files.
    horizon = ModuleType("gx1.contracts.entry_observed_market_horizons_v1")
    horizon.require_observed_entry_scope = lambda recipe: None
    horizon.bind_observed_entry_dataset = lambda *args, **kwargs: pytest.fail("Dataset bind reached")
    monkeypatch.setitem(sys.modules, horizon.__name__, horizon)
    cohorts = ModuleType("gx1.contracts.unified_exit_bounded_val_cohort_v1")
    cohorts.build_chronological_measurement_cohort = lambda *args, **kwargs: pytest.fail("cohort reached")
    monkeypatch.setitem(sys.modules, cohorts.__name__, cohorts)
    metadata_calls, execution_boundaries = [], []
    run_id = "SCOPE_FIXTURE_081"

    if boundary == "initial_restore":
        invoke, arguments, path, binding, _ = _preflight_fixture(tmp_path, monkeypatch)
        namespace = invoke.__globals__
        original_recipe = namespace["_require_native_full_train_recipe"]

        def recipe_boundary(*args):
            recipe, files, provenance = original_recipe(*args)
            recipe.update(dataset_run_id=run_id, chronological_prefix={"fixture": True},
                chronological_initial_measurement={"fixture": True})
            recipe["trainer_cli"].update(batch_size=16, epochs=1, seed=20260911,
                learning_rate=0.0001, weight_decay=0.0001)
            return recipe, files, provenance

        namespace["_require_native_full_train_recipe"] = recipe_boundary
        namespace["trainer"]._resolve_device = lambda _: SimpleNamespace(type="cpu")
        namespace["_build_bound_full_train_components"] = lambda **kwargs: {}
        namespace["json"] = json
        metadata_result = {"validated": "initial fixture"}
    else:
        invoke, namespace = _load_dynamic_builder()
        files = {name: tmp_path / name for name in namespace["_DYNAMIC_M5_DATA_FILES"]}
        for path in files.values():
            path.write_bytes(b"small bound fixture")
        for split in ("train", "val"):
            files[f"entry_{split}_manifest"].write_text(
                json.dumps({"extra": {"entry_run_id": run_id}}), encoding="utf-8")
        files["source_bundle_metadata"].write_text(json.dumps({
            "seq_len": MODEL_NATIVE_SEQ_LEN,
            "multi_tf": {tf.lower() + "_seq_len": count
                for tf, count in PRODUCTION_MTF_PER_TF_WINDOW_BARS},
        }), encoding="utf-8")
        files["entry_only_base_normalization"].write_text(json.dumps({
            "contract": {"contract_sha256": "b" * 64, "lineage": {"dataset_run_id": run_id}},
        }), encoding="utf-8")
        sources = {split: {kind: {"path": str(files[f"entry_{split}_{kind}"]),
            "sha256": hashlib.sha256(files[f"entry_{split}_{kind}"].read_bytes()).hexdigest()}
            for kind in ("parquet", "manifest")} for split in ("train", "val")}
        path = files["entry_train_parquet"]
        binding = sources["train"]["parquet"]
        metadata_result = {
            "physical_preprocessing": {"entry_training_phase": PHASE,
                "base_contract_sha256": "b" * 64, "physical_sources": sources},
            "entry_m5_path_single_epoch": {"witness": {"training_phase": PHASE}},
        }
        namespace["trainer"] = SimpleNamespace(MODEL_NATIVE_SEQ_LEN=MODEL_NATIVE_SEQ_LEN)
        namespace["val"] = SimpleNamespace(
            _read=lambda item: json.loads(item.read_text(encoding="utf-8")),
            _bind_multi_tf_cache_from_source_bundle_metadata=lambda metadata: None,
            file_sha256=owner.file_sha256,
        )
        arguments = {"files": files, "chronological_prefix": {"fixture": True},
            "observed_entry_scope": {"training_phase": PHASE}, "dataset_run_id": run_id,
            "device": SimpleNamespace(type="cpu"), "batch_size": 16, "epochs": 1,
            "seed": 20260911, "learning_rate": 0.0001, "weight_decay": 0.0001,
            "val_limits": dict.fromkeys(("max_model_forwards", "max_state_views",
                "max_wall_seconds", "progress_interval_forwards"), 1)}

    def metadata_boundary(*args, **kwargs):
        assert owner._FILE_VERIFICATION_MEMO.get() is not None
        before = physical_hash_calls[path]
        owner.require_binding(binding, label="first metadata use")
        owner.require_binding(binding, label="nested metadata use")
        assert physical_hash_calls[path] == before + 1
        metadata_calls.append(boundary)
        if fail_metadata:
            raise _MetadataFailure("metadata failure before execution")
        return metadata_result

    def execution_boundary(*args, **kwargs):
        assert owner._FILE_VERIFICATION_MEMO.get() is None
        before = physical_hash_calls[path]
        owner.require_binding(binding, label="outside metadata at execution boundary")
        assert physical_hash_calls[path] == before + 1
        execution_boundaries.append(boundary)
        raise _ExecutionBoundaryStop("before Dataset or fresh-state restore")

    if boundary == "initial_restore":
        sys.modules["gx1.contracts.unified_exit_native_candidate_campaign_v1"].require_chronological_initial_measurement = metadata_boundary
        namespace["_restore_prefix_initial_measurement_state"] = execution_boundary
    else:
        namespace["_require_prefix_component_bindings"] = metadata_boundary
        namespace["val"].EntryV10CtxDataset = execution_boundary

    for operation in (1, 2):
        expected = _MetadataFailure if fail_metadata else _ExecutionBoundaryStop
        with pytest.raises(expected):
            invoke(**arguments)
        assert owner._FILE_VERIFICATION_MEMO.get() is None
        assert len(metadata_calls) == operation
        assert len(execution_boundaries) == (0 if fail_metadata else operation)
        before = physical_hash_calls[path]
        owner.require_binding(binding, label="outside completed or failed operation")
        assert physical_hash_calls[path] == before + 1
    assert not (tmp_path / "unused-output").exists()
