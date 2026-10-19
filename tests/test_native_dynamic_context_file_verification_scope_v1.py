from __future__ import annotations

import ast
from collections.abc import Mapping
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from gx1.contracts import local_random_access_campaign_v2 as owner
from gx1.contracts.entry_model_native_training_objective_v1 import (
    ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE as PHASE,
)
from tests.test_native_full_train_file_verification_scope_v1 import physical_hash_calls


SOURCE = Path(__file__).resolve().parents[1] / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"


def _load_context_owner():
    """Load the genuine definition, without importing trainer/model/Dataset code."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name == "_dynamic_entry_native_context_binding")
    isolated = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        function,
    ], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace = {"Mapping": Mapping, "Path": Path,
        "ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE": PHASE,
        "_dynamic_m5_model_kwargs": lambda phase: {"dynamic_m5_query_minutes": tuple(range(1, 481))}}
    exec(compile(isolated, str(SOURCE), "exec"), namespace)
    return namespace[function.name]


@pytest.fixture
def context_unit(tmp_path, monkeypatch):
    """Completed semantic owners are isolated; every binding/hash read is real.

    This fixture qualifies the context owner's memo lifetime, not a genuine
    market cohort, normalization fit, target formula, or model execution.
    """
    def write(name, value):
        path = tmp_path / name
        raw = (json.dumps(value, sort_keys=True) + "\n").encode()
        path.write_bytes(raw)
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

    sources = {split: {"parquet": write(split + ".source", {"unit_source": split}),
        "manifest": write(split + ".manifest.json", {"unit_manifest": split}),
        "physical_rows": 4, "clock_sha256": "c" * 64} for split in ("train", "val")}
    audits = {split: write(split + ".audit.json", {"unit_audit": split}) for split in sources}
    native = write("native-design.json", {"unit_completed_design": True})
    cost = write("cost.json", {"unit_completed_cost": True})
    surfaces = {name: {"field_names": [name + "_fixture"]}
        for name in ("signal", "ctx_cont", "mtf_m5", "mtf_m15", "mtf_h1", "mtf_h4", "mtf_d1")}
    raw_contract = {"contract_sha256": "b" * 64, "surfaces": surfaces,
        "ctx_cat": {"field_names": ["category_fixture"]},
        "lineage": {"train_parquet_path": sources["train"]["parquet"]["path"],
            "train_parquet_sha256": sources["train"]["parquet"]["sha256"],
            "train_manifest_path": sources["train"]["manifest"]["path"],
            "train_manifest_sha256": sources["train"]["manifest"]["sha256"],
            "entry_train_decision_row_count": sources["train"]["physical_rows"]}}
    base = write("base.json", {"schema_version": "gx1_entry_only_dynamic_m5_base_normalization_v1",
        "decision": "PASS", "native_training_admitted": False, "entry_training_phase": PHASE,
        "contract_sha256": raw_contract["contract_sha256"], "contract": raw_contract,
        "sequence_source_audit": audits["train"]})
    prefix = {"design": native, "normalization_result": base}
    observed = write("observed.json", {"chronological_prefix": prefix,
        "files": {"train_cost_authority": cost}, "unit_marker": "AAAA"})
    target_contract = {"unit_completed_target_owner": True}
    identity = {"design": observed, "training_phase": PHASE,
        "target_contract": target_contract, "query_minutes": list(range(1, 481))}

    cohorts = ModuleType("gx1.contracts.unified_exit_bounded_val_cohort_v1")
    cohorts.require_bounded_val_cohort = lambda value: dict(value)
    monkeypatch.setitem(sys.modules, cohorts.__name__, cohorts)
    normalization = ModuleType("gx1.contracts.entry_model_native_input_normalization_v1")
    normalization.EXPECTED_SURFACES = tuple(surfaces)

    def completed_normalization(value, **kwargs):
        assert value == raw_contract
        assert kwargs == {"expected_field_names": {k: v["field_names"] for k, v in surfaces.items()},
            "expected_ctx_cat_names": ["category_fixture"], "expected_entry_training_phase": PHASE}
        return value

    normalization.require_input_normalization_contract = completed_normalization
    monkeypatch.setitem(sys.modules, normalization.__name__, normalization)
    market = ModuleType("gx1.contracts.entry_observed_market_v1")
    market.entry_observed_dynamic_m5_path_contract = lambda: target_contract

    def completed_observed_scope(value):
        assert value == {"entry_observed_market": observed, "chronological_prefix": prefix,
            "files": {**{f"entry_{split}_{kind}": sources[split][kind]
                for split in sources for kind in ("parquet", "manifest")}, "train_cost_authority": cost}}
        return {"physical_sources": sources}

    market.require_entry_observed_market_scope = completed_observed_scope
    market.entry_observed_market_identity = lambda scope: dict(identity)
    monkeypatch.setitem(sys.modules, market.__name__, market)
    context = {"entry_training_phase": PHASE, "entry_supervision_identity": identity,
        "physical_sources": sources, "entry_only_base_normalization": base,
        "base_contract_sha256": raw_contract["contract_sha256"], "val_sequence_audit": audits["train"],
        "evaluation_cohort": {"plan": native, "training_phase": PHASE, "source_split": "train",
            "measurement_role": "train", "exit_evaluation": False,
            "parent_entry_parquet": sources["train"]["parquet"],
            "parent_entry_manifest": sources["train"]["manifest"], "population_rows": 4,
            "source_clock_sha256": sources["train"]["clock_sha256"]},
        "max_model_forwards": 16, "max_state_views": 256,
        "max_wall_seconds": 10800, "progress_interval_forwards": 16}
    return SimpleNamespace(bind=_load_context_owner(), context=context, sources=sources,
        observed=observed, base=base, native=native, audits=audits)


@pytest.mark.parametrize("split", ["train", "val"])
def test_dynamic_context_reuses_real_digests_and_closes_before_caller_work(
    context_unit, physical_hash_calls, split,
):
    case = context_unit
    source = case.sources[split]
    case.context["val_sequence_audit"] = case.audits[split]
    case.context["evaluation_cohort"].update(source_split=split,
        measurement_role="train" if split == "train" else "control",
        parent_entry_parquet=source["parquet"], parent_entry_manifest=source["manifest"],
        source_clock_sha256=source["clock_sha256"])
    before_context = copy.deepcopy(case.context)
    bindings = (case.observed, case.native, case.base, case.audits[split])
    paths = [Path(binding["path"]) for binding in bindings]

    first = case.bind(case.context)
    assert all(physical_hash_calls[path] == 1 for path in paths)
    assert owner._FILE_VERIFICATION_MEMO.get() is None
    assert first["entry_only_base_normalization"] == case.base
    assert first["sequence_source_audit"] == case.audits[split]
    assert first["report_only"] is True and first["exit_evaluation"] is False
    assert first["native_launch_authorized"] is False and first["test_accessed"] is False
    # Work after the real function returns must perform its own physical read.
    observed_path = Path(case.observed["path"])
    owner.require_binding(case.observed, label="caller after context return")
    assert physical_hash_calls[observed_path] == 2
    before = physical_hash_calls.copy()
    assert case.bind(case.context) == first
    assert all(physical_hash_calls[path] == before[path] + 1 for path in paths)

    before = physical_hash_calls.copy()
    with owner.file_verification_scope():
        assert case.bind(case.context) == first
        with owner.file_verification_scope():
            assert case.bind(case.context) == first
        assert case.bind(case.context) == first
        assert all(physical_hash_calls[path] == before[path] + 1 for path in paths)
    assert owner._FILE_VERIFICATION_MEMO.get() is None
    assert case.context == before_context


@pytest.mark.parametrize("fault", ["wrong_expected_sha", "same_size_mutation", "late_limits"])
def test_dynamic_context_rejects_stale_binding_and_resets_failed_operation(
    context_unit, physical_hash_calls, fault,
):
    case = context_unit
    path = Path(case.observed["path"])
    original = path.read_bytes()
    original_sha = case.observed["sha256"]
    original_limit = case.context["progress_interval_forwards"]
    if fault == "wrong_expected_sha":
        case.observed["sha256"] = "0" * 64
    elif fault == "late_limits":
        case.context["progress_interval_forwards"] = 0

    expected = RuntimeError if fault == "late_limits" else owner.RandomAccessCampaignError
    message = "CANDIDATE_NATIVE_VAL_LIMITS_INVALID" if fault == "late_limits" else "SHA-256 mismatch"
    with pytest.raises(expected, match=message):
        if fault == "same_size_mutation":
            with owner.file_verification_scope():
                case.bind(case.context)
                before = path.stat()
                warmed_reads = physical_hash_calls[path]
                changed = original.replace(b"AAAA", b"BBBB")
                assert changed != original and len(changed) == len(original)
                path.write_bytes(changed)
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
                assert path.stat().st_size == before.st_size
                assert path.stat().st_mtime_ns == before.st_mtime_ns
                try:
                    case.bind(case.context)
                finally:
                    assert physical_hash_calls[path] == warmed_reads + 1
        else:
            case.bind(case.context)
    assert owner._FILE_VERIFICATION_MEMO.get() is None

    case.observed["sha256"] = original_sha
    case.context["progress_interval_forwards"] = original_limit
    path.write_bytes(original)
    before = physical_hash_calls.copy()
    checked = case.bind(case.context)
    assert checked["entry_training_phase"] == PHASE
    for binding in (case.observed, case.native, case.base, case.audits["train"]):
        item = Path(binding["path"])
        assert physical_hash_calls[item] == before[item] + 1
    assert owner._FILE_VERIFICATION_MEMO.get() is None
