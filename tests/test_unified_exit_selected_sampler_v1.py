from __future__ import annotations
import copy
import json
from pathlib import Path
import pytest
from gx1.contracts.unified_exit_pilot_normalization_v1 import canonical_sha256
from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    build_random_access_sampler_contract,
)
from gx1.contracts.unified_exit_selected_sampler_v1 import (
    build_selected_sampler_artifact,
    file_sha256,
    require_selected_sampler_artifact,
)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    candidate_path = tmp_path / "candidates.json"
    predecessor_path = tmp_path / "v3.json"
    root_path = tmp_path / "v4.json"
    equivalence_path = tmp_path / "equivalence.json"
    receipt_path = tmp_path / "receipt.json"
    sampler = build_random_access_sampler_contract(
        split="train",
        source_lineage_sha256="1" * 64,
        transition_budget_per_epoch=65536,
        transitions_per_entry=4,
        entry_pair_population=65295,
    )
    candidates = {
        "schema_version": "gx1_unified_exit_sampler_benchmark_candidate_set_v1",
        "candidate_set_sha256": "2" * 64,
        "candidates": [
            {
                "sampler_contract": sampler,
                "selected": False,
                "status": "BENCHMARK_PENDING",
            }
        ],
    }
    _write(candidate_path, candidates)
    predecessor = {
        "root_sha256": "f2797d10a9a97a028fe3c532a3ed10849766673e860327fb53139338512d6d64",
        "test_accessed": False,
    }
    _write(predecessor_path, predecessor)
    root = {
        "root_sha256": "25f911a0d03595ee5bbbea5fa6a66da98bed91eb24bdb0f561f6870a719d007f",
        "test_accessed": False,
    }
    _write(root_path, root)
    equivalence = {
        "schema_version": "gx1_unified_exit_random_access_index_v3_to_v4_equivalence_v1",
        "decision": "PASS",
        "benchmark_receipt_transfer_to_v4_authorized": True,
        "only_parent_entry_coordinate_and_binding_fields_added": True,
        "selection_uses_outcome_values": False,
        "test_accessed": False,
        "sampler_candidate_set_sha256": candidates["candidate_set_sha256"],
        "predecessor_root": {
            "path": str(predecessor_path),
            "sha256": file_sha256(predecessor_path),
        },
        "sampled_transition_schedules": [
            {
                "sampler_contract_sha256": sampler["contract_sha256"],
                "schedule_streams_byte_identical": True,
                "transition_budget_per_epoch": 65536,
            }
        ],
    }
    equivalence["receipt_sha256"] = canonical_sha256(equivalence)
    _write(equivalence_path, equivalence)
    receipt = {
        "schema_version": "gx1_unified_exit_random_access_train_benchmark_v2",
        "decision": "PASS",
        "candidate_selection_performed": True,
        "selection_uses_outcome_values": False,
        "test_data_used": False,
        "selected_candidate": {
            "batch_size": 16,
            "transition_budget_per_epoch": 65536,
            "sampler_contract_sha256": sampler["contract_sha256"],
        },
        "run_bindings": {
            "files": {
                "candidate_set": {
                    "path": str(candidate_path),
                    "sha256": file_sha256(candidate_path),
                },
                "root_manifest": {
                    "path": str(predecessor_path),
                    "sha256": file_sha256(predecessor_path),
                },
            }
        },
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _write(receipt_path, receipt)
    return receipt_path, candidate_path, root_path, equivalence_path


def test_selected_sampler_is_bound_to_measured_v3_and_equivalent_v4(
    tmp_path: Path,
) -> None:
    receipt, candidates, root, equivalence = _fixture(tmp_path)
    artifact = build_selected_sampler_artifact(
        benchmark_receipt_path=receipt,
        candidate_set_path=candidates,
        random_access_root_path=root,
        equivalence_receipt_path=equivalence,
    )
    assert require_selected_sampler_artifact(artifact) == artifact
    assert (
        artifact["batch_size"] == 16
        and artifact["transition_budget_per_epoch"] == 65536
    )
    tampered = copy.deepcopy(artifact)
    tampered["batch_size"] = 8
    with pytest.raises(RuntimeError):
        require_selected_sampler_artifact(tampered, verify_files=False)

# These fixtures are deliberately synthetic metadata; they prove admission and
# routing, never throughput, cost authority or native learning.
from gx1.contracts.unified_exit_pilot_normalization_v1 import build_sampler_benchmark_candidate_set
from gx1.contracts.unified_exit_random_access_index_v1 import (
    FULL_POPULATION_ROOT_SCHEMA_VERSION, RANDOM_ACCESS_INDEX_V2_SCHEMA_VERSION,
)
from gx1.scripts import benchmark_unified_exit_random_access_train_v1 as benchmark


def _seal(path, value, key):
    value.pop(key, None)
    value[key] = canonical_sha256(value)
    _write(path, value)
    return value


def _file_binding(path):
    return {"path": str(path), "sha256": file_sha256(path)}


def _direct_fixture(tmp_path, selected_budget=131072, *, population=652552, design_path=None):
    if design_path is None:
        design = json.loads(Path("configs/research/NATIVE_V38_LEARNING_DESIGN_20261001.json").read_text())
        design_path = tmp_path / "DESIGN.json"
        _write(design_path, design)
    else:
        design = json.loads(design_path.read_text())
    candidates = build_sampler_benchmark_candidate_set(
        source_lineage_sha256="1" * 64, entry_pair_population=population)
    cp = tmp_path / "candidates.json"
    _write(cp, candidates)
    bp = tmp_path / "FINAL_BINDINGS_BUNDLE.json"
    bundle = _seal(bp, {
        "schema_version": "gx1_unified_exit_pilot_final_bindings_bundle_v1",
        "decision": "BLOCKED_PENDING_TRAIN_ONLY_SAMPLER_BENCHMARK",
        "sampler_benchmark_candidates": candidates, "test_accessed": False,
    }, "bundle_sha256")
    mp = tmp_path / "train.manifest.json"
    source_binding = {"path": str(tmp_path / "TRAIN_SOURCE_NOT_READ.parquet"), "sha256": "2" * 64}
    manifest = _seal(mp, {
        "schema_version": RANDOM_ACCESS_INDEX_V2_SCHEMA_VERSION,
        "decision": "PASS", "split": "train", "storage_granularity": "one_row_per_entry",
        "full_prefix_states_stored": False, "chunk_pointers_stored": False,
        "target_q_stored": False, "economic_terminal_count": 0,
        "split_end_is_right_censor": True, "test_accessed": False,
        "entry_row_count": population, "parent_entry_source_rows": population,
        "index_parquet_path": str(tmp_path / "TRAIN_INDEX_NOT_READ.parquet"),
        "index_parquet_sha256": "3" * 64,
        "source_bindings": {
            **{key: source_binding.copy() for key in (
                "entry_parquet", "entry_manifest")},
            **{f"parent_entry_{kind}": design["calendar"]["source_bindings"]["train"][kind]
               for kind in ("parquet", "manifest")},
            "final_bindings_bundle": _file_binding(bp),
        },
        **{key: "2" * 64 for key in (
            "child_entry_clock_sha256", "parent_entry_clock_sha256",
            "parent_entry_row_indices_sha256", "parent_entry_mapping_sha256",
            )},
        "parent_entry_source_sha256": design["calendar"]["source_bindings"]["train"]["parquet"]["sha256"],
        "parent_entry_manifest_sha256": design["calendar"]["source_bindings"]["train"]["manifest"]["sha256"],
    }, "manifest_sha256")
    rp = tmp_path / "ROOT.json"
    root = _seal(rp, {
        "schema_version": FULL_POPULATION_ROOT_SCHEMA_VERSION, "decision": "PASS",
        "allowed_splits": ["train", "val"], "storage_granularity": "one_row_per_entry",
        "full_prefix_states_stored": False, "chunk_pointers_stored": False, "test_accessed": False,
        "sampler_selection_status": "BLOCKED_PENDING_TRAIN_ONLY_BENCHMARK",
        "selected_sampler_contract_sha256": None, "final_bindings_bundle_sha256": bundle["bundle_sha256"],
        "full_train_population": {"entry_row_count": population, "parent_entry_source_rows": population,
                                 "train_manifest_sha256": manifest["manifest_sha256"]},
        "splits": {
            "train": {"entry_row_count": population, "manifest_path": str(mp),
                      "manifest_sha256": manifest["manifest_sha256"],
                      "index_parquet_path": manifest["index_parquet_path"],
                      "index_parquet_sha256": manifest["index_parquet_sha256"]},
            "val": {},
        },
    }, "root_sha256")
    measurements = []
    for candidate in candidates["candidates"]:
        contract = candidate["sampler_contract"]
        budget, entries = contract["transition_budget_per_epoch"], contract["entry_pairs_per_epoch"]
        duration = 100.0 if budget <= selected_budget else 1801.0
        cycles = -(-population // entries)
        row = {
            "batch_size": 16, "measured_entry_pairs": entries, "measured_transitions": budget,
            "median_seconds": duration, "materialize_seconds": 60.0, "collate_seconds": 30.0,
            "entry_pairs_per_second": entries / duration, "transitions_per_second": budget / duration,
            "peak_python_allocation_bytes": 1000, "peak_padded_model_input_bytes": 2000,
            "population_cycle_epochs": cycles, "measured_epoch_seconds": duration,
            "projected_entry_population_cycle_seconds": duration * cycles,
        }
        measurements.append({
            "transition_budget_per_epoch": budget, "sampler_contract_sha256": contract["contract_sha256"],
            "full_selected_entry_pairs": entries, "full_budget_measured": True, "batch_size_sweep": [row],
        })
    geometry = {"seq_len": 96, "per_tf_seq_lens": {"M5": 16, "M15": 64, "H1": 96, "H4": 96, "D1": 252},
                "multi_tf_closed_bar": True}
    metadata = {"seq_len": geometry["seq_len"],
                "multi_tf": {tf.lower() + "_seq_len": n for tf,n in geometry["per_tf_seq_lens"].items()}}
    _write(tmp_path / "SOURCE_BUNDLE.json", metadata)
    receipt_path = tmp_path / "receipt.json"
    receipt = _seal(receipt_path, {
        "schema_version": "gx1_unified_exit_random_access_train_benchmark_v2", "decision": "PASS",
        "device": "cpu", "candidate_budgets": [32768, 65536, 131072], "transitions_per_entry": 4,
        "batch_sizes": [16], "repeats": 1, "selection_policy": benchmark._selection_policy(population),
        "reference_workload": benchmark._reference_workload_from_design(design),
        "candidate_selection_performed": True, "selection_uses_outcome_values": False, "test_data_used": False,
        "run_bindings": {"input_geometry": geometry, "files": {"candidate_set": _file_binding(cp), "root_manifest": _file_binding(rp),
                                  "chronological_design": _file_binding(design_path)}},
        "candidates": measurements,
        "selected_candidate": benchmark.select_measured_sampler_candidate(
            measurements, entry_pair_population=population),
    }, "receipt_sha256")
    return {"receipt_path": receipt_path, "candidate_path": cp, "root_path": rp, "manifest_path": mp,
            "bundle_path": bp, "receipt": receipt, "candidates": candidates, "root": root, "manifest": manifest}


def _direct_build(case):
    return build_selected_sampler_artifact(
        benchmark_receipt_path=case["receipt_path"], candidate_set_path=case["candidate_path"],
        random_access_root_path=case["root_path"],
    )


@pytest.mark.parametrize("budget,cycles", [(32768, 80), (65536, 40), (131072, 20)])
def test_direct_choice_replays_measurements_and_binds_exact_current_root(tmp_path, budget, cycles, monkeypatch):
    case = _direct_fixture(tmp_path, budget)
    import pandas as pd
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **kw: pytest.fail("Metadata admission must not read data"))
    artifact = _direct_build(case)
    assert artifact["transition_budget_per_epoch"] == budget
    assert artifact["entry_pairs_per_epoch"] == budget // 4
    assert artifact["selected_sampler_contract"]["entry_pair_population"] == 652552
    assert case["receipt"]["selected_candidate"]["population_cycle_epochs"] == cycles
    assert artifact["selection_mode"] == "direct_same_root_benchmark_v1"
    assert "v3_to_v4_equivalence" not in artifact
    assert require_selected_sampler_artifact(artifact) == artifact


@pytest.mark.parametrize("fault", [
    "smoke", "selection_flag", "outcomes", "test", "missing_candidate", "duplicate_budget",
    "partial", "count", "negative_memory", "negative_time", "cycle", "speed", "projection",
    "policy", "wrong_winner", "candidate_hash", "repeat", "batch",
    "float_budget", "float_policy", "all_memory_over_cap",
])
def test_direct_choice_rejects_rehashed_invalid_measurements(tmp_path, fault):
    c = _direct_fixture(tmp_path)
    r = c["receipt"]
    row = r["candidates"][0]["batch_size_sweep"][0]
    if fault == "smoke": r["decision"] = "NON_AUTHORITATIVE_CAPPED_SMOKE"
    if fault == "selection_flag": r["candidate_selection_performed"] = False
    if fault == "outcomes": r["selection_uses_outcome_values"] = True
    if fault == "test": r["test_data_used"] = True
    if fault == "missing_candidate": r["candidates"].pop()
    if fault == "duplicate_budget": r["candidates"][0] = copy.deepcopy(r["candidates"][1])
    if fault == "partial": r["candidates"][0]["full_budget_measured"] = False
    if fault == "count": row["measured_transitions"] -= 1
    if fault == "negative_memory": row["peak_python_allocation_bytes"] = -1
    if fault == "negative_time": row["median_seconds"] = -1
    if fault == "cycle": row["population_cycle_epochs"] = 8
    if fault == "speed": row["transitions_per_second"] += 1
    if fault == "projection": row["projected_entry_population_cycle_seconds"] += 1
    if fault == "policy": r["selection_policy"]["max_measured_cpu_prep_epoch_seconds"] += 1
    if fault == "wrong_winner": r["selected_candidate"]["transition_budget_per_epoch"] = 32768
    if fault == "candidate_hash": r["candidates"][0]["sampler_contract_sha256"] = "f" * 64
    if fault == "repeat": r["repeats"] = 2
    if fault == "batch": r["batch_sizes"] = [8]
    if fault == "float_budget": r["candidates"][0]["transition_budget_per_epoch"] = 32768.0
    if fault == "float_policy":
        r["selection_policy"]["max_measured_cpu_prep_epoch_seconds"] = float(r["selection_policy"]["max_measured_cpu_prep_epoch_seconds"])
    if fault == "all_memory_over_cap":
        for candidate in r["candidates"]:
            candidate["batch_size_sweep"][0]["peak_python_allocation_bytes"] = benchmark.MAX_PEAK_PYTHON_ALLOCATION_BYTES + 1
    _seal(c["receipt_path"], r, "receipt_sha256")
    with pytest.raises(RuntimeError):
        _direct_build(c)


@pytest.mark.parametrize("kind", ["candidate", "root"])
def test_direct_choice_rejects_byte_identical_unbound_paths(tmp_path, kind):
    c = _direct_fixture(tmp_path)
    key = kind + "_path"
    path = tmp_path / ("copy-" + c[key].name)
    path.write_bytes(c[key].read_bytes())
    c[key] = path
    with pytest.raises(RuntimeError, match="SOURCE_MISMATCH"):
        _direct_build(c)


def test_direct_choice_rejects_root_with_wrong_population(tmp_path):
    c = _direct_fixture(tmp_path)
    c["root"]["full_train_population"]["entry_row_count"] -= 1
    _seal(c["root_path"], c["root"], "root_sha256")
    with pytest.raises(RuntimeError):
        _direct_build(c)


def test_direct_choice_rejects_role_swapped_manifest_before_access(tmp_path, monkeypatch):
    c = _direct_fixture(tmp_path)
    sealed = tmp_path / "v10_seq513_dataset__ENTRY_FITTED_Q_test.manifest.json"
    c["root"]["splits"]["train"]["manifest_path"] = str(sealed)
    _seal(c["root_path"], c["root"], "root_sha256")
    c["receipt"]["run_bindings"]["files"]["root_manifest"] = _file_binding(c["root_path"])
    _seal(c["receipt_path"], c["receipt"], "receipt_sha256")
    original = Path.resolve
    def resolve(path, *a, **kw):
        if path == sealed: pytest.fail("Sealed TEST pointer was followed")
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(RuntimeError, match="TRAIN_MANIFEST_REQUIRED"):
        _direct_build(c)


@pytest.mark.parametrize("field", ["candidate_set", "random_access_root", "train_index_manifest", "final_bindings_bundle"])
def test_direct_artifact_requires_unchanged_complete_binding(tmp_path, field):
    c = _direct_fixture(tmp_path)
    artifact = _direct_build(c)
    artifact[field]["file_sha256"] = "f" * 64
    artifact.pop("artifact_sha256")
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    with pytest.raises(RuntimeError):
        require_selected_sampler_artifact(artifact)


def test_direct_cli_declares_mode_and_uses_immutable_publication(tmp_path, monkeypatch):
    from gx1.scripts import materialize_unified_exit_selected_sampler_v1 as cli
    c = _direct_fixture(tmp_path)
    path = tmp_path / "SELECTED.json"
    args = ["--direct-benchmark", "--benchmark-receipt", str(c["receipt_path"]),
            "--candidate-set", str(c["candidate_path"]), "--random-access-root", str(c["root_path"]),
            "--output", str(path)]
    assert cli.main(args) == 0 and not path.exists()
    calls = []
    original = cli._atomic_write_new_json
    def publish(output, value):
        calls.append(output)
        return original(output, value)
    monkeypatch.setattr(cli, "_atomic_write_new_json", publish)
    assert cli.main(args + ["--publish"]) == 0
    before = path.read_bytes()
    assert calls == [path]
    require_selected_sampler_artifact(json.loads(before))
    with pytest.raises(RuntimeError, match="OUTPUT_EXISTS"):
        cli.main(args + ["--publish"])
    assert path.read_bytes() == before
    with pytest.raises(SystemExit):
        cli.main(args[1:])


@pytest.mark.parametrize("budget", [32768, 65536, 131072])
def test_native_component_binds_measured_current_sampler(tmp_path, budget):
    from gx1.scripts import run_unified_exit_random_access_full_train_v1 as runner
    c = _direct_fixture(tmp_path, budget)
    selected = _direct_build(c)
    sp = tmp_path / "SELECTED.json"; _write(sp, selected)
    design = json.loads(Path("configs/research/NATIVE_V38_LEARNING_DESIGN_20261001.json").read_text())
    dp = tmp_path / "DESIGN.json"; _write(dp, design)
    prefix = {"design": _file_binding(dp)}
    files = {"selected_sampler": sp, "sampler_candidate_set": c["candidate_path"], "random_access_root": c["root_path"],
             "source_bundle_metadata": tmp_path / "SOURCE_BUNDLE.json"}
    assert runner._require_component_sampler(files=files, chronological_prefix=prefix) == selected
    with pytest.raises(RuntimeError, match="MEASURED_SAMPLER_REQUIRED"):
        runner._require_component_sampler(files={k:v for k,v in files.items() if k != "selected_sampler"}, chronological_prefix=prefix)
    with pytest.raises(RuntimeError, match="LEGACY_SAMPLER_CHANGE_FORBIDDEN"):
        runner._require_component_sampler(files=files, chronological_prefix=None)
    copy_path = tmp_path / "ROOT-copy.json"; copy_path.write_bytes(c["root_path"].read_bytes())
    with pytest.raises(RuntimeError, match="MEASURED_SAMPLER_SOURCE_MISMATCH"):
        runner._require_component_sampler(files={**files, "random_access_root": copy_path}, chronological_prefix=prefix)


@pytest.mark.parametrize("fault", ["missing_reference", "changed_cutoff", "missing_design", "wrong_design_path"])
def test_current_native_rejects_other_benchmark_workload(tmp_path, fault):
    from gx1.scripts import run_unified_exit_random_access_full_train_v1 as runner
    c = _direct_fixture(tmp_path)
    r = c["receipt"]
    if fault == "missing_reference": r["reference_workload"] = {}
    if fault == "changed_cutoff": r["reference_workload"]["reference_cutoff_time_ns"] -= 1
    if fault == "missing_design": r["run_bindings"]["files"].pop("chronological_design")
    if fault == "wrong_design_path": r["run_bindings"]["files"]["chronological_design"]["path"] = str(tmp_path / "unbound.json")
    _seal(c["receipt_path"], r, "receipt_sha256")
    selected = _direct_build(c)
    sp = tmp_path / "SELECTED.json"; _write(sp, selected)
    with pytest.raises(RuntimeError, match="WORKLOAD_MISMATCH"):
        runner._require_component_sampler(
            files={"selected_sampler": sp, "sampler_candidate_set": c["candidate_path"], "random_access_root": c["root_path"]},
            chronological_prefix={"design": _file_binding(tmp_path / "DESIGN.json")},
        )


def test_native_builder_stops_unmeasured_current_design_before_dataset_or_model(tmp_path, monkeypatch):
    from gx1.scripts import run_unified_exit_random_access_full_train_v1 as runner
    import torch
    design = json.loads(Path("configs/research/NATIVE_V38_LEARNING_DESIGN_20261001.json").read_text())
    dp = tmp_path / "DESIGN.json"; _write(dp, design)
    def forbidden(*a, **kw):
        pytest.fail("Current builder touched dataset/model before measured sampler admission")
    monkeypatch.setattr(runner.val, "EntryV10CtxDataset", forbidden)
    monkeypatch.setattr(runner.val, "_model", forbidden)
    with pytest.raises(RuntimeError, match="MEASURED_SAMPLER_REQUIRED"):
        runner._build_bound_full_train_components(
            files={}, dataset_run_id="SYNTHETIC_ONLY", seed_launch_path=None, seed_authority_path=None,
            seed_authority_file_sha256=None, device=torch.device("cpu"), batch_size=16, epochs=30,
            seed=design["initialization"]["seed"], learning_rate=.0001, weight_decay=.0001, val_limits={},
            chronological_prefix={"design": _file_binding(dp)},
        )


def test_direct_choice_rejects_other_canonical_candidate_set_inside_bound_bundle(tmp_path):
    c = _direct_fixture(tmp_path)
    bundle = json.loads(c["bundle_path"].read_text())
    bundle["sampler_benchmark_candidates"] = build_sampler_benchmark_candidate_set(
        source_lineage_sha256="e" * 64, entry_pair_population=652552)
    _seal(c["bundle_path"], bundle, "bundle_sha256")
    manifest = c["manifest"]
    manifest["source_bindings"]["final_bindings_bundle"] = _file_binding(c["bundle_path"])
    _seal(c["manifest_path"], manifest, "manifest_sha256")
    root = c["root"]
    root["final_bindings_bundle_sha256"] = bundle["bundle_sha256"]
    root["splits"]["train"]["manifest_sha256"] = manifest["manifest_sha256"]
    root["full_train_population"]["train_manifest_sha256"] = manifest["manifest_sha256"]
    _seal(c["root_path"], root, "root_sha256")
    c["receipt"]["run_bindings"]["files"]["root_manifest"] = _file_binding(c["root_path"])
    _seal(c["receipt_path"], c["receipt"], "receipt_sha256")
    with pytest.raises(RuntimeError, match="FINAL_BINDINGS_INVALID"):
        _direct_build(c)


@pytest.mark.parametrize("fault", ["missing", "seq_len", "mtf_len", "partial_bar"])
def test_native_current_sampler_requires_measured_input_geometry(tmp_path, fault):
    from gx1.scripts import run_unified_exit_random_access_full_train_v1 as runner
    c = _direct_fixture(tmp_path)
    geometry = c["receipt"]["run_bindings"]["input_geometry"]
    if fault == "missing": c["receipt"]["run_bindings"].pop("input_geometry")
    if fault == "seq_len": geometry["seq_len"] += 1
    if fault == "mtf_len": geometry["per_tf_seq_lens"]["H1"] += 1
    if fault == "partial_bar": geometry["multi_tf_closed_bar"] = False
    _seal(c["receipt_path"], c["receipt"], "receipt_sha256")
    selected = _direct_build(c)
    sp = tmp_path / "SELECTED.json"; _write(sp, selected)
    with pytest.raises(RuntimeError, match="INPUT_GEOMETRY_MISMATCH"):
        runner._require_component_sampler(
            files={"selected_sampler": sp, "sampler_candidate_set": c["candidate_path"],
                   "random_access_root": c["root_path"], "source_bundle_metadata": tmp_path / "SOURCE_BUNDLE.json"},
            chronological_prefix={"design": _file_binding(tmp_path / "DESIGN.json")},
        )
