"""A genuine small dynamic pointer audit is sealed once per runtime Session.

The previously qualified cost-schema semantics are isolated at their existing
entry economics boundary. Both real cost hash owners still read and bind their
physical dependencies, so this is an IO/lifetime proof, not a new economics proof.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import pytest

from gx1.contracts import entry_candidate_checkpoint_policy_v1 as policy
from gx1.contracts import local_random_access_campaign_v2 as files
from tests.test_entry_observed_market_native import dynamic_physical_receipt
from tests.test_native_prefix_recipe import _dynamic_single_epoch_design, _write


def _bind(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def verified_contract(tmp_path, monkeypatch, dynamic_physical_receipt):
    import numpy as np
    import pandas as pd
    from gx1.contracts import entry_causal_m1_outcomes_v1 as quote_owner
    from gx1.contracts import unified_exit_market_closure_authority_v1 as closure
    from gx1.contracts import entry_observed_market_v1 as observed
    from gx1.contracts import entry_model_native_aux_targets_v3 as targets
    from gx1.contracts import unified_exit_broker_evidence_v1 as broker
    from gx1.contracts import unified_exit_no_cap_economic_authority_v1 as economic
    receipt, original_design = dynamic_physical_receipt()
    design = json.loads(Path(original_design["path"]).read_text())
    sources = design["calendar"]["source_bindings"]
    # The reused receipt template is intentionally only three rows. A genuine
    # whole-TRAIN witness requires >=16, so rebuild its physical payload with
    # seventeen decisions (one full batch plus the preserved one-row tail).
    rows = 17
    starts = {"train": "2025-05-01T00:00Z", "val": "2025-05-03T00:00Z"}
    raw_times = pd.DatetimeIndex(np.concatenate([
        pd.date_range(pd.Timestamp(first) + pd.Timedelta(minutes=5),
            periods=rows * 5 + 1, freq="min").asi8 for first in starts.values()]), tz="UTC")
    bid = 100. + np.arange(len(raw_times), dtype=np.float64) / 100
    raw = pd.DataFrame({"time": raw_times, "bid_open": bid, "ask_open": bid + .1,
        "bid_high": bid + .2, "ask_high": bid + .3, "bid_low": bid - .2, "ask_low": bid - .1})
    quote_path = Path(receipt["splits"]["train"]["dynamic_m5_supervision"]["source_quote_binding"]["path"])
    raw.to_parquet(quote_path, index=False)
    physical_quote = {**_bind(quote_path), "size_bytes": quote_path.stat().st_size,
        "rows": len(raw), "clock_sha256": closure.m1_clock_sha256(raw_times)}
    for split, first in starts.items():
        decisions = pd.date_range(first, periods=rows, freq="5min")
        cutoff = design["calendar"]["train_control_cutoff" if split == "train"
            else "development_control_entry_end_exclusive"]
        index = quote_owner.prepare_dynamic_m1_prefix_index(raw, split_end_exclusive=cutoff,
            source_parquet_sha256=physical_quote["sha256"])
        fill = quote_owner.build_dynamic_m1_fill_surface(index, m5_decision_times=decisions)
        query = {"query_entry_rows": np.arange(rows, dtype=np.int64),
            "horizon_minutes": np.ones(rows, dtype=np.int64)}
        terminal = quote_owner.query_dynamic_m1_prefix(index, fill_surface=fill, **query)
        path = quote_owner.query_dynamic_m1_prefix(index, fill_surface=fill,
            endpoint_kind="closed_m1_prefix", **query)
        terminal_mask, path_mask = terminal["terminal_gross_observed"], path["path_observed"]
        frame = pd.DataFrame({"time": decisions,
            "dynamic_entry_m1_row": fill.entry_m1_row.to_numpy(np.int64),
            "dynamic_first_prefix_observed": terminal_mask.any(axis=1) | path_mask.any(axis=1),
            "dynamic_first_terminal_observed_long": terminal_mask[:, 0],
            "dynamic_first_terminal_observed_short": terminal_mask[:, 1],
            "dynamic_first_path_observed_long": path_mask[:, 0],
            "dynamic_first_path_observed_short": path_mask[:, 1],
            "dynamic_first_terminal_censor_reason": terminal["terminal_censor_reason"],
            "dynamic_first_path_censor_reason": path["path_censor_reason"]})
        parquet_path = Path(sources[split]["parquet"]["path"])
        frame.to_parquet(parquet_path, index=False)
        meta = receipt["splits"][split]["dynamic_m5_supervision"]
        meta.update(source_quote_binding=copy.deepcopy(physical_quote), index_binding=index.binding())
        manifest_path = Path(sources[split]["manifest"]["path"])
        manifest = json.loads(manifest_path.read_text())
        manifest["extra"].update(rows=rows, dynamic_m5_supervision=meta)
        emitted = manifest["extra"]["aux_head_target_contract"]
        emitted.update(candidate_rows_before_primary_availability=rows, primary_valid_rows_emitted=rows)
        clock_sha = hashlib.sha256(np.asarray(decisions.asi8, dtype="<i8").tobytes()).hexdigest()
        sources[split].update(parquet=_bind(parquet_path), manifest=_write(manifest_path, manifest),
            physical_rows=rows, clock_sha256=clock_sha)
        def reason_counts(values):
            names, counts = np.unique(values, return_counts=True)
            return {str(name): int(count) for name, count in zip(names, counts)}
        receipt["splits"][split] = {"source_parquet": sources[split]["parquet"],
            "source_manifest": sources[split]["manifest"], "rows": rows,
            "physical_clock_sha256": clock_sha, "dynamic_m5_supervision": meta,
            "first_prefix_observed_rows": int(frame.dynamic_first_prefix_observed.sum()),
            "first_terminal_observed_rows_by_side": dict(zip(("long", "short"), map(int, terminal_mask.sum(axis=0)))),
            "first_path_observed_rows_by_side": dict(zip(("long", "short"), map(int, path_mask.sum(axis=0)))),
            "first_terminal_censor_reason_counts": reason_counts(terminal["terminal_censor_reason"]),
            "first_path_censor_reason_counts": reason_counts(path["path_censor_reason"]),
            "entry_m1_pointer_sha256": hashlib.sha256(np.asarray(fill.entry_m1_row, dtype="<i8").tobytes()).hexdigest()}
    population_dir = tmp_path / "population"
    population_dir.mkdir()
    population_design, witness = _dynamic_single_epoch_design(population_dir, rows=rows)
    quote_manifest = _write(tmp_path / "quotes.manifest.json", {"qualified_quote_manifest_boundary": True})
    quote = copy.deepcopy(receipt["splits"]["train"]["dynamic_m5_supervision"]["source_quote_binding"])
    quote["authority"] = {"m1_source_manifest_path": quote_manifest["path"],
        "m1_source_manifest_sha256": quote_manifest["sha256"]}
    for split in ("train", "val"):
        proof = receipt["splits"][split]
        proof["dynamic_m5_supervision"]["source_quote_binding"] = copy.deepcopy(quote)
        manifest_path = Path(sources[split]["manifest"]["path"])
        manifest = json.loads(manifest_path.read_text())
        manifest["output_data_path"] = sources[split]["parquet"]["path"]
        manifest["extra"]["dynamic_m5_supervision"] = proof["dynamic_m5_supervision"]
        sources[split]["manifest"] = _write(manifest_path, manifest)
        proof["source_manifest"] = sources[split]["manifest"]
    from gx1.contracts.unified_exit_random_access_sampler_v1 import build_random_access_sampler_contract
    from gx1.scripts.materialize_unified_exit_chronological_coordinates_v1 import _epoch0_order
    sampler = build_random_access_sampler_contract(split="train",
        source_lineage_sha256=sources["train"]["manifest"]["sha256"],
        transition_budget_per_epoch=rows, transitions_per_entry=1, entry_pair_population=rows)
    order_path = Path(witness["epoch0_parent_order"]["path"])
    np.save(order_path, _epoch0_order(sampler), allow_pickle=False)
    witness.update(train_parquet=sources["train"]["parquet"], train_manifest=sources["train"]["manifest"],
        native_sampler_contract=sampler, epoch0_parent_order=_bind(order_path),
        entry_eligibility_cutoff_utc=design["calendar"]["train_control_cutoff"])
    population_design["scope"]["entry_m5_path_single_epoch"] = _write(population_dir / "witness.json", witness)
    design["scope"].update(population_design["scope"])
    design["budget"] = population_design["budget"]
    design["calendar"]["train_entry_start_inclusive"] = "2025-05-01T00:00:00Z"
    design_binding = _write(Path(original_design["path"]), design)
    receipt["design"] = design_binding
    artifacts = {"design": design_binding,
        "labels_result": _write(tmp_path / "physical_receipt.json", receipt),
        "normalization_result": _write(tmp_path / "base.json", {"qualified_normalization_boundary": True}),
        "native_coordinates": _write(tmp_path / "coordinates.json", {"qualified_coordinates_boundary": True})}
    cost_evidence = tmp_path / "cost_evidence.json"
    cost_evidence.write_text('{"qualified_cost_evidence_boundary":true}')
    cost_component = tmp_path / "cost_component.json"
    cost_component.write_text('{"qualified_cost_component_boundary":true}')
    cost_bindings = [_bind(cost_evidence), _bind(cost_component)]
    cost_policy = _write(tmp_path / "cost_policy.json", {"executable_bid_ask": {
        "parquet": {key: quote[key] for key in ("path", "sha256")}, "manifest": quote_manifest}})
    authority = _write(tmp_path / "cost_authority.json", {"policy": {
        "path": cost_policy["path"], "file_sha256": cost_policy["sha256"], "artifact_sha256": "a" * 64}})
    economics = {"qualified_cost_semantics_unit_boundary": True}
    def checked_economics(value, **coverage):
        # Do not invent a broker package. Exercise both actual delegated hash
        # paths after their separately qualified schema boundary succeeds.
        assert value == json.loads(Path(authority["path"]).read_text())
        assert coverage == {"coverage_start_utc": design["calendar"]["train_entry_start_inclusive"],
            "coverage_end_utc": design["calendar"]["development_control_entry_end_exclusive"]}
        broker._require_local_file(cost_bindings[0], "UNIT_COST_EVIDENCE", True)
        assert economic.file_sha256(Path(cost_bindings[1]["path"])) == cost_bindings[1]["sha256"]
        return economics
    monkeypatch.setattr(observed, "entry_observed_market_economics", checked_economics)
    source_files = {f"entry_{split}_{kind}": sources[split][kind]
        for split in ("train", "val") for kind in ("parquet", "manifest")}
    source_files["train_cost_authority"] = authority
    phase = receipt["training_phase"]
    target = receipt["target_contract"]
    plan = {"schema_version": observed.DYNAMIC_M5_SCHEMA_VERSION + "_learning_design",
        "target_contract": target, "chronological_prefix": artifacts, "files": source_files,
        "training_phase": phase, "exit_updates_entry": False, "native_launch_authorized": False,
        "test_data_used": False}
    plan_binding = _write(tmp_path / "observed_plan.json", plan)
    identity = observed.entry_observed_market_identity({"binding": plan_binding, "target_contract": target,
        "economics": economics, "training_phase": phase,
        "query_minutes": tuple(target["query_contract"]["horizon_minutes"]),
        "target_m1_source_sha256": quote["sha256"]})
    prefix = {"schema_version": "gx1_candidate_prefix_training_binding_v1",
        "entry_training_phase": phase, "fixed_final_model_variant": "ONLINE", "artifacts": artifacts,
        "entry_only_base_normalization": artifacts["normalization_result"],
        "entry_m5_path_single_epoch": {"population_witness": design["scope"]["entry_m5_path_single_epoch"],
            "train_rows": rows, "last_batch_rows": rows % 16, "drop_last": False},
        "train_rows": rows, "maximum_optimizer_steps": -(-rows // 16), "maximum_completed_epochs": 1,
        "physical_sources": sources,
        "entry_observed_market": {role: {"identity": copy.deepcopy(identity)} for role in ("train", "control")}}
    monitor = policy.DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR
    expected = policy.checkpoint_policy_metadata(checkpoint_monitor=monitor, prefix_training_binding=prefix)
    contract = {"schema_version": "gx1_candidate_training_session_v1", "run_id": "UNIT",
        "chronological_prefix": prefix, "training": {"checkpoint_policy": expected}}
    calls = {"hashes": [], "pointer_audits": []}
    raw_hash = files._file_sha256_uncached
    def counted_hash(path, **kwargs):
        calls["hashes"].append(str(path))
        return raw_hash(path, **kwargs)
    monkeypatch.setattr(files, "_file_sha256_uncached", counted_hash)
    raw_audit = targets.require_m5_dynamic_physical_target_receipt
    def counted_audit(*args, **kwargs):
        calls["pointer_audits"].append(True)
        return raw_audit(*args, **kwargs)
    monkeypatch.setattr(targets, "require_m5_dynamic_physical_target_receipt", counted_audit)
    paths = {"parquet": Path(sources["train"]["parquet"]["path"]),
        "manifest": Path(sources["val"]["manifest"]["path"]),
        "npy": Path(witness["epoch0_parent_order"]["path"]),
        "cost_evidence": cost_evidence, "cost_component": cost_component}
    return contract, calls, paths


def _metadata(contract, context=None):
    return policy.checkpoint_policy_metadata(
        checkpoint_monitor=contract["training"]["checkpoint_policy"]["checkpoint_monitor"],
        prefix_training_binding=contract["chronological_prefix"], verified_context=context)


def test_verified_policy_runs_genuine_pointer_validation_once_and_returns_detached_metadata(
        verified_contract, tmp_path):
    contract, calls, paths = verified_contract
    expected = copy.deepcopy(contract["training"]["checkpoint_policy"])
    context = policy.prepare_verified_checkpoint_policy(contract=contract)
    assert len(calls["pointer_audits"]) == 1
    assert set(paths.values()).issubset({Path(path) for path in calls["hashes"]})
    assert all(count == 1 for count in Counter(calls["hashes"]).values())
    before = copy.deepcopy(calls)
    for _ in range(3):
        context.require_contract(contract)
        actual = _metadata(contract, context)
        assert actual == expected
        actual["entry_m5_path_single_epoch"]["train_rows"] = 999
        assert policy.require_checkpoint_policy(expected, context="UNIT",
            checkpoint_monitor=expected["checkpoint_monitor"],
            prefix_training_binding=contract["chronological_prefix"], verified_context=context) == expected
    checkpoint = tmp_path / "mutable_checkpoint.pt"
    checkpoint.write_bytes(b"state changes outside the immutable metadata read set")
    context.require_contract(contract)
    assert calls == before
    with pytest.raises(TypeError, match="runtime-only"):
        pickle.dumps(context)
    with pytest.raises(AttributeError, match="immutable"):
        context._monitor = "other"


@pytest.mark.parametrize("dependency", ["parquet", "manifest", "npy", "cost_evidence", "cost_component"])
def test_verified_policy_rejects_physical_dependency_mutation_without_rereading(
        verified_contract, dependency):
    contract, calls, paths = verified_contract
    context = policy.prepare_verified_checkpoint_policy(contract=contract)
    path = paths[dependency]
    before = path.stat()
    path.write_bytes(b"x" * before.st_size)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert path.stat().st_ctime_ns != before.st_ctime_ns
    reads = copy.deepcopy(calls)
    with pytest.raises(files.RandomAccessCampaignError, match="identity changed"):
        context.require_contract(contract)
    with pytest.raises(files.RandomAccessCampaignError, match="identity changed"):
        _metadata(contract, context)
    assert calls == reads


@pytest.mark.parametrize("fault", ["prefix", "monitor", "policy", "contract", "unknown_context", "legacy_factory"])
def test_verified_policy_rejects_context_or_contract_substitution(verified_contract, fault):
    contract, calls, _ = verified_contract
    context = policy.prepare_verified_checkpoint_policy(contract=contract)
    changed = copy.deepcopy(contract)
    reads = copy.deepcopy(calls)
    if fault == "prefix":
        changed["chronological_prefix"]["artifacts"]["design"]["sha256"] = "0" * 64
        with pytest.raises(RuntimeError, match="BINDING_MISMATCH"):
            _metadata(changed, context)
    elif fault == "monitor":
        changed["training"]["checkpoint_policy"]["checkpoint_monitor"] = policy.CHECKPOINT_MONITOR
        with pytest.raises(RuntimeError, match="BINDING_MISMATCH"):
            _metadata(changed, context)
    elif fault == "policy":
        changed["training"]["checkpoint_policy"]["max_epochs"] = True
        with pytest.raises(RuntimeError, match="CHECKPOINT_POLICY_INVALID"):
            policy.require_checkpoint_policy(changed["training"]["checkpoint_policy"], context="UNIT",
                checkpoint_monitor=policy.DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR,
                prefix_training_binding=contract["chronological_prefix"], verified_context=context)
    elif fault == "unknown_context":
        with pytest.raises(RuntimeError, match="CONTEXT_INVALID"):
            _metadata(contract, {"forged": True})
    elif fault == "legacy_factory":
        changed.pop("chronological_prefix")
        with pytest.raises(RuntimeError, match="CONTRACT_INVALID"):
            policy.prepare_verified_checkpoint_policy(contract=changed)
    else:
        changed["run_id"] = "another Session"
        with pytest.raises(RuntimeError, match="CONTRACT_MISMATCH"):
            context.require_contract(changed)
    assert calls == reads
    if fault in {"prefix", "monitor", "policy"}:
        with pytest.raises(RuntimeError, match="CONTRACT_MISMATCH"):
            context.require_contract(changed)


def test_verified_policy_new_context_and_default_path_revalidate_fresh_physical_bytes(verified_contract):
    contract, calls, _ = verified_contract
    one = policy.prepare_verified_checkpoint_policy(contract=contract)
    two = policy.prepare_verified_checkpoint_policy(contract=copy.deepcopy(contract))
    assert one is not two
    assert len(calls["pointer_audits"]) == 2
    assert all(count == 2 for count in Counter(calls["hashes"]).values())
    expected = contract["training"]["checkpoint_policy"]
    assert _metadata(contract) == expected
    assert _metadata(contract) == expected
    assert len(calls["pointer_audits"]) == 4
    assert all(count == 4 for count in Counter(calls["hashes"]).values())
    with files.file_verification_scope():
        with pytest.raises(files.RandomAccessCampaignError, match="no outer scope"):
            policy.prepare_verified_checkpoint_policy(contract=contract)
    # Legacy None retains its original policy and does not need a captured set.
    assert policy.checkpoint_policy_metadata() == policy.checkpoint_policy_metadata(verified_context=None)


@pytest.mark.parametrize("fault", ["wrong_expected_sha", "declared_policy", "mutation_after_pointer_audit", "semantic_failure"])
def test_verified_policy_failed_preparation_cannot_seal_or_leak_a_scope(
        verified_contract, monkeypatch, fault):
    from gx1.contracts import entry_model_native_aux_targets_v3 as targets
    contract, calls, paths = verified_contract
    original = targets.require_m5_dynamic_physical_target_receipt
    if fault == "wrong_expected_sha":
        contract = copy.deepcopy(contract)
        contract["chronological_prefix"]["artifacts"]["design"]["sha256"] = "0" * 64
    elif fault == "declared_policy":
        contract = copy.deepcopy(contract)
        contract["training"]["checkpoint_policy"]["max_epochs"] = True
    else:
        def changed_after_audit(*args, **kwargs):
            actual = original(*args, **kwargs)
            if fault == "semantic_failure":
                raise RuntimeError("unit semantic failure after actual pointer checks")
            path = paths["parquet"]
            before = path.stat()
            path.write_bytes(b"x" * before.st_size)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            return actual
        monkeypatch.setattr(targets, "require_m5_dynamic_physical_target_receipt", changed_after_audit)
    with pytest.raises(RuntimeError):
        policy.prepare_verified_checkpoint_policy(contract=contract)
    with pytest.raises(files.RandomAccessCampaignError, match="nonempty active scope"):
        files.capture_verified_file_snapshot()
    if fault == "semantic_failure":
        monkeypatch.setattr(targets, "require_m5_dynamic_physical_target_receipt", original)
        calls["hashes"].clear()
        policy.prepare_verified_checkpoint_policy(contract=contract)
        assert all(count == 1 for count in Counter(calls["hashes"]).values())
