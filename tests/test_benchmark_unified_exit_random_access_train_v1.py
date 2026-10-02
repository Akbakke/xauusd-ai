from __future__ import annotations

import json
import math
from functools import partial

import pytest

from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    build_sampler_benchmark_candidate_set,
    canonical_sha256,
)
from gx1.contracts.unified_exit_random_access_train_factory_v1 import _candidate_contracts
from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    build_random_access_sampler_contract,
)
from gx1.scripts import benchmark_unified_exit_random_access_train_v1 as benchmark


class _Adapter:
    def __init__(self, budget: int, *, population: int = 65_295) -> None:
        self.contract = build_random_access_sampler_contract(
            split="train",
            source_lineage_sha256="1" * 64,
            transition_budget_per_epoch=budget,
            transitions_per_entry=4,
            entry_pair_population=population,
        )

    def random_access_training_bindings_v1(self):
        return {
            "sampler_contract": self.contract,
            "normalization_artifact": {},
            "m1_source_sha256": "2" * 64,
            "market_closure_authority_sha256": "3" * 64,
            "economic_step_manifest_sha256": "4" * 64,
            "economics_objective_contract_sha256": "5" * 64,
        }

    def random_access_selected_entry_rows_v1(self):
        return tuple(range(self.contract["entry_pairs_per_epoch"]))

    def materialize_random_access_training_item_v1(
        self, entry: int, *, outer_batch_index: int
    ):
        return {"entry": entry, "outer": outer_batch_index}


def test_benchmark_sweeps_preregistered_budgets_and_marks_cap(monkeypatch) -> None:
    def fake_collate(items, **_kwargs):
        return {
            "transition_count": len(items) * 4,
            "online_model_inputs": {},
            "target_model_inputs": {},
        }

    monkeypatch.setattr(benchmark, "collate_random_access_training_items", fake_collate)
    receipt = benchmark.benchmark_random_access_train_candidates_v1(
        adapter_factory=_Adapter,
        batch_sizes=(1, 2),
        repeats=1,
        max_selected_entries=2,
    )
    assert receipt["decision"] == "NON_AUTHORITATIVE_CAPPED_SMOKE"
    assert receipt["candidate_budgets"] == [32_768, 65_536, 131_072]
    assert receipt["transitions_per_entry"] == 4
    assert all(not item["full_budget_measured"] for item in receipt["candidates"])
    assert all(
        [row["batch_size"] for row in item["batch_size_sweep"]] == [1, 2]
        for item in receipt["candidates"]
    )
    assert receipt["candidate_selection_performed"] is False
    assert receipt["selected_candidate"] is None


@pytest.mark.parametrize("population,cycles", [(65_295, [8, 4, 2]), (652_552, [80, 40, 20])])
def test_authoritative_benchmark_selects_only_after_complete_receipt(
    monkeypatch, population, cycles,
) -> None:
    clock = iter(float(index) for index in range(100_000))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(benchmark.gc, "collect", lambda: None)
    monkeypatch.setattr(benchmark.tracemalloc, "start", lambda: None)
    monkeypatch.setattr(benchmark.tracemalloc, "stop", lambda: None)
    monkeypatch.setattr(benchmark.tracemalloc, "get_traced_memory", lambda: (0, 123))
    monkeypatch.setattr(benchmark, "MAX_MEASURED_CPU_PREP_EPOCH_SECONDS", 100_000)

    def fake_collate(items, **_kwargs):
        import torch

        return {
            "transition_count": len(items) * 4,
            "online_model_inputs": {"x": torch.zeros((len(items), 2))},
            "target_model_inputs": {},
        }

    monkeypatch.setattr(benchmark, "collate_random_access_training_items", fake_collate)
    receipt = benchmark.benchmark_random_access_train_candidates_v1(
        adapter_factory=partial(_Adapter, population=population),
        batch_sizes=(16,),
        repeats=1,
    )
    assert receipt["decision"] == "PASS"
    assert receipt["candidate_selection_performed"] is True
    assert receipt["selected_candidate"]["population_cycle_epochs"] == cycles[-1]
    assert receipt["selected_candidate"]["transition_budget_per_epoch"] == 131_072
    assert [
        candidate["batch_size_sweep"][0]["population_cycle_epochs"]
        for candidate in receipt["candidates"]
    ] == cycles
    assert receipt["selection_policy"]["selection_uses_outcome_values"] is False
    assert receipt["test_data_used"] is False


def test_uncapped_benchmark_rejects_non_preregistered_shape() -> None:
    import pytest

    with pytest.raises(RuntimeError, match="PREREGISTRATION_MISMATCH"):
        benchmark.benchmark_random_access_train_candidates_v1(
            adapter_factory=_Adapter,
            batch_sizes=(8,),
            repeats=1,
        )



def _fake_collate(items, **kwargs):
    return {
        "transition_count": len(items) * 4,
        "online_model_inputs": {},
        "target_model_inputs": {},
    }


@pytest.mark.parametrize("population", [65_295, 652_552])
def test_benchmark_uses_canonical_population(monkeypatch, population):
    monkeypatch.setattr(benchmark, "collate_random_access_training_items", _fake_collate)
    receipt = benchmark.benchmark_random_access_train_candidates_v1(
        adapter_factory=partial(_Adapter, population=population),
        batch_sizes=(16,), repeats=1, max_selected_entries=1,
    )
    assert receipt["selection_policy"] == benchmark._selection_policy(population)
    assert [
        c["batch_size_sweep"][0]["population_cycle_epochs"]
        for c in receipt["candidates"]
    ] == [math.ceil(population / n) for n in (8192, 16384, 32768)]
    assert receipt["selected_candidate"] is None


@pytest.mark.parametrize("fault", [
    "population", "lineage", "split", "budget", "transitions", "hash", "schedule_rule",
])
def test_benchmark_rejects_incompatible_contract_before_materializing(monkeypatch, fault):
    monkeypatch.setattr(benchmark, "collate_random_access_training_items", _fake_collate)
    seen = []
    def factory(budget):
        adapter = _Adapter(budget)
        if budget == 65_536:
            fields = {
                key: adapter.contract[key] for key in (
                    "split", "source_lineage_sha256", "transition_budget_per_epoch",
                    "transitions_per_entry", "entry_pair_population",
                )
            }
            if fault == "population": fields["entry_pair_population"] += 1
            if fault == "lineage": fields["source_lineage_sha256"] = "a" * 64
            if fault == "split": fields["split"] = "val"
            if fault == "budget": fields["transition_budget_per_epoch"] = 32_768
            if fault == "transitions": fields["transitions_per_entry"] = 8
            adapter.contract = build_random_access_sampler_contract(**fields)
            if fault == "hash": adapter.contract["contract_sha256"] = "f" * 64
            if fault == "schedule_rule": adapter.contract["entry_schedule"] = "outcome_rank"
        def materialize(entry, *, outer_batch_index):
            seen.append(budget)
            return {"entry": entry}
        adapter.materialize_random_access_training_item_v1 = materialize
        return adapter
    with pytest.raises(RuntimeError, match="UNIFIED_EXIT_RANDOM_ACCESS"):
        benchmark.benchmark_random_access_train_candidates_v1(
            adapter_factory=factory, batch_sizes=(16,), repeats=1, max_selected_entries=1,
        )
    assert seen == [32_768]


@pytest.mark.parametrize("population", [65_295, 652_552])
def test_candidate_factory_uses_canonical_candidate_owner(tmp_path, population):
    value = build_sampler_benchmark_candidate_set(
        source_lineage_sha256="1" * 64, entry_pair_population=population,
    )
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(value))
    contracts = _candidate_contracts(path)
    assert tuple(contracts) == (32_768, 65_536, 131_072)
    assert all(c["entry_pair_population"] == population for c in contracts.values())


@pytest.mark.parametrize("fault", [
    "population", "lineage", "transitions", "candidate_population", "candidate_lineage",
    "candidate_split", "candidate_transitions", "selected",
])
def test_candidate_factory_rejects_self_hashed_population_or_lineage_drift(tmp_path, fault):
    value = build_sampler_benchmark_candidate_set(
        source_lineage_sha256="1" * 64, entry_pair_population=652_552,
    )
    if fault == "population": value["entry_pair_population"] += 1
    if fault == "lineage": value["source_lineage_sha256"] = "a" * 64
    if fault == "transitions": value["transitions_per_entry"] = 8
    if fault == "selected": value["candidates"][0]["selected"] = True
    if fault.startswith("candidate_"):
        fields = {
            key: value["candidates"][0]["sampler_contract"][key] for key in (
                "split", "source_lineage_sha256", "transition_budget_per_epoch",
                "transitions_per_entry", "entry_pair_population",
            )
        }
        if fault == "candidate_population": fields["entry_pair_population"] += 1
        if fault == "candidate_lineage": fields["source_lineage_sha256"] = "a" * 64
        if fault == "candidate_split": fields["split"] = "val"
        if fault == "candidate_transitions": fields["transitions_per_entry"] = 8
        value["candidates"][0]["sampler_contract"] = build_random_access_sampler_contract(**fields)
    value.pop("candidate_set_sha256")
    value["candidate_set_sha256"] = canonical_sha256(value)
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="CANDIDATES_INVALID"):
        _candidate_contracts(path)

def test_benchmark_receipt_publication_is_durable(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    events = []
    original_sync = benchmark.os.fsync
    original_publish = benchmark._publish_file_noreplace
    original_dir_sync = benchmark._fsync_directory
    def sync(fd):
        events.append("file_fsync")
        original_sync(fd)
    def publish(source, destination):
        events.append("publish")
        assert json.loads(source.read_text()) == {"decision": "PASS"}
        original_publish(source, destination)
    def dir_sync(directory):
        events.append("directory_fsync")
        original_dir_sync(directory)
    monkeypatch.setattr(benchmark.os, "fsync", sync)
    monkeypatch.setattr(benchmark, "_publish_file_noreplace", publish)
    monkeypatch.setattr(benchmark, "_fsync_directory", dir_sync)
    benchmark._atomic_write_new_json(path, {"decision": "PASS"})
    assert json.loads(path.read_text()) == {"decision": "PASS"}
    assert events[:3] == ["file_fsync", "publish", "directory_fsync"]
    assert not list(tmp_path.glob(".receipt.json.*"))


def test_benchmark_receipt_publication_preserves_racing_winner(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    original_sync = benchmark.os.fsync
    def publish_winner(fd):
        original_sync(fd)
        path.write_text("existing winner")
    monkeypatch.setattr(benchmark.os, "fsync", publish_winner)
    with pytest.raises((RuntimeError, FileExistsError, OSError)):
        benchmark._atomic_write_new_json(path, {"decision": "PASS"})
    assert path.read_text() == "existing winner"
    stages = list(tmp_path.glob(".receipt.json.*"))
    assert len(stages) == 1  # Failed staging remains available to retention.
    assert json.loads(stages[0].read_text()) == {"decision": "PASS"}


@pytest.mark.parametrize("mode", ["file", "symlink", "dangling_symlink"])
def test_benchmark_receipt_publication_preserves_existing_name(tmp_path, mode):
    path = tmp_path / "receipt.json"
    target = tmp_path / "original.json"
    if mode == "file":
        path.write_text("existing")
    else:
        if mode == "symlink":
            target.write_text("existing")
        path.symlink_to(target)
    with pytest.raises(RuntimeError, match="OUTPUT_EXISTS"):
        benchmark._atomic_write_new_json(path, {"decision": "PASS"})
    if mode == "file":
        assert path.read_text() == "existing"
    else:
        assert path.is_symlink()
        assert target.exists() == (mode == "symlink")
        if mode == "symlink":
            assert target.read_text() == "existing"


def test_benchmark_receipt_rejects_corrupt_staging(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    original_sync = benchmark.os.fsync
    def corrupt(fd):
        original_sync(fd)
        stage, = tmp_path.glob(".receipt.json.*")
        stage.write_text('{"decision":"CORRUPTED"}')
    monkeypatch.setattr(benchmark.os, "fsync", corrupt)
    with pytest.raises(RuntimeError, match="STAGING_INVALID"):
        benchmark._atomic_write_new_json(path, {"decision": "PASS"})
    assert not path.exists()
    assert len(list(tmp_path.glob(".receipt.json.*"))) == 1


@pytest.mark.parametrize("field", ["selected", "entry_pair_population"])
def test_candidate_factory_rejects_equal_python_values_with_changed_bytes(tmp_path, field):
    value = build_sampler_benchmark_candidate_set(
        source_lineage_sha256="1" * 64, entry_pair_population=652_552,
    )
    if field == "selected":
        value["candidates"][0]["selected"] = 0
    else:
        contract = value["candidates"][0]["sampler_contract"]
        contract["entry_pair_population"] = float(contract["entry_pair_population"])
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="CANDIDATES_INVALID"):
        _candidate_contracts(path)


def test_benchmark_forwards_and_records_actual_reference_workload(monkeypatch):
    from pathlib import Path
    design = json.loads(Path("configs/research/NATIVE_V38_LEARNING_DESIGN_20261001.json").read_text())
    workload = benchmark._reference_workload_from_design(design)
    calls = []
    class ReferenceAdapter(_Adapter):
        def random_access_training_bindings_v1(self):
            return {**super().random_access_training_bindings_v1(), **workload}
    def collate(items, **kwargs):
        calls.append({key: kwargs[key] for key in workload})
        return _fake_collate(items)
    monkeypatch.setattr(benchmark, "collate_random_access_training_items", collate)
    receipt = benchmark.benchmark_random_access_train_candidates_v1(
        adapter_factory=ReferenceAdapter, batch_sizes=(16,), repeats=1, max_selected_entries=1)
    assert calls == [workload, workload, workload]
    assert receipt["reference_workload"] == workload
    assert receipt["candidate_selection_performed"] is False


def test_benchmark_rejects_candidate_reference_workload_drift(monkeypatch):
    from pathlib import Path
    design = json.loads(Path("configs/research/NATIVE_V38_LEARNING_DESIGN_20261001.json").read_text())
    workload = benchmark._reference_workload_from_design(design)
    seen = []
    def factory(budget):
        adapter = _Adapter(budget)
        get_binding = adapter.random_access_training_bindings_v1
        adapter.random_access_training_bindings_v1 = lambda: {
            **get_binding(), **workload,
            "reference_cutoff_time_ns": workload["reference_cutoff_time_ns"] - (budget != 32768)}
        return adapter
    def collate(items, **kwargs):
        seen.append(kwargs["sampler_contract"]["transition_budget_per_epoch"])
        return _fake_collate(items)
    monkeypatch.setattr(benchmark, "collate_random_access_training_items", collate)
    with pytest.raises(RuntimeError, match="REFERENCE_DRIFT"):
        benchmark.benchmark_random_access_train_candidates_v1(
            adapter_factory=factory, batch_sizes=(16,), repeats=1, max_selected_entries=1)
    assert seen == [32768]


def test_benchmark_cli_binds_design_to_factory_collation_and_input_geometry(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    import gx1.contracts.unified_exit_lifecycle_v1 as lifecycle
    import gx1.contracts.unified_exit_random_access_train_factory_v1 as factory_owner
    import gx1.models.entry_v10.entry_v10_ctx_train_v3 as trainer
    file_keys = ("root_manifest", "candidate_set", "composite_normalization", "economics_readiness",
                 "train_cost_authority", "feature_lifecycle_root", "entry_train_parquet",
                 "entry_train_manifest", "sequence_source_audit", "m5_prebuilt")
    paths = {key: tmp_path / (key + ".json") for key in file_keys}
    for path in paths.values(): path.write_text("{}")
    design = json.loads(Path("configs/research/NATIVE_V38_LEARNING_DESIGN_20261001.json").read_text())
    design["calendar"]["source_bindings"]["train"].update({
        key: {"path": str(paths["entry_train_" + key]), "sha256": benchmark.file_sha256(paths["entry_train_" + key])}
        for key in ("parquet", "manifest")
    })
    dp = tmp_path / "DESIGN.json"; dp.write_text(json.dumps(design))
    args = SimpleNamespace(**paths, chronological_design=dp, output_json=tmp_path / "SMOKE.json",
        mtf_cache_dir=tmp_path, dataset_run_id="SYNTHETIC_ONLY", seq_len=96,
        m5_len=16, m15_len=64, h1_len=96, h4_len=96, d1_len=252,
        batch_size=16, repeats=1, max_selected_entries=1)
    monkeypatch.setenv("GX1_V10_MULTI_TF_V4_CACHE_DIR", "")
    monkeypatch.setattr(benchmark, "_parser", lambda: SimpleNamespace(parse_args=lambda argv: args))
    def indexed_corpus(**kwargs):
        assert kwargs["root_manifest_path"] == paths["root_manifest"]
        assert kwargs["splits"] == ("train",)
        return SimpleNamespace(splits={"train": object()})
    monkeypatch.setattr(lifecycle, "UnifiedExitLifecycleCorpus",
                        SimpleNamespace(from_random_access_index=indexed_corpus))
    def dataset(**kwargs):
        assert kwargs["multi_tf_closed_bar"] is True
        return SimpleNamespace(seq_len=kwargs["seq_len"], per_tf_seq_lens=kwargs["per_tf_seq_lens"])
    monkeypatch.setattr(trainer, "EntryV10CtxDataset", dataset)
    seen = {}
    def factory(**kwargs):
        seen["factory"] = kwargs
        return object()
    monkeypatch.setattr(factory_owner, "build_random_access_train_adapter_factory_v1", factory)
    def measure(**kwargs):
        seen["measure"] = kwargs
        return {"decision": "NON_AUTHORITATIVE_CAPPED_SMOKE"}
    monkeypatch.setattr(benchmark, "benchmark_random_access_train_candidates_v1", measure)
    assert benchmark.main([]) == 0
    workload = benchmark._reference_workload_from_design(design)
    assert {key: seen["factory"][key] for key in workload} == workload
    bindings = seen["measure"]["run_bindings"]
    assert bindings["files"]["chronological_design"] == {"path": str(dp), "sha256": benchmark.file_sha256(dp)}
    assert bindings["input_geometry"] == {
        "seq_len": 96, "per_tf_seq_lens": {"M5":16, "M15":64, "H1":96, "H4":96, "D1":252},
        "multi_tf_closed_bar": True,
    }

@pytest.fixture
def indexed_feature_source(tmp_path, monkeypatch):
    """Synthetic source/clock test; exact surface schema has separate tests."""
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import gx1.contracts.unified_exit_lifecycle_v1 as owner
    import gx1.contracts.unified_exit_random_access_index_v1 as index_owner
    def write(name, value, key=None):
        if key:
            value = dict(value)
            value[key] = owner.canonical_json_sha256(value)
        p = tmp_path / name
        p.write_text(json.dumps(value))
        return {"path": str(p), "sha256": owner.sha256_file(p)}
    def raw(name, value):
        p=tmp_path/name; p.write_bytes(value)
        return {"path":str(p), "sha256":owner.sha256_file(p)}
    times=pd.date_range("2025-01-01",periods=12,freq="min",tz="UTC").delete(6)
    parent=raw("parent.parquet",b"synthetic parent")
    feature=raw("feature.parquet",b"synthetic features")
    fm=write("feature.parquet.manifest.json",{"pair_generation_id":"pair"})
    feature.update(manifest_path=fm["path"],manifest_sha256=fm["sha256"],rows=len(times)-1)
    parent_manifest=write("parent.manifest.json",{})
    population=write("population.json",{
        "decision":"PASS","test_accessed":False,
        "m1_source":{"parent_full_tape":{
            "parquet_path":parent["path"],"parquet_sha256":parent["sha256"],
            "manifest_path":parent_manifest["path"],"manifest_sha256":parent_manifest["sha256"]}},
        "m1_feature_base":feature}, "contract_sha256")
    view=write("view.json",{"test_accessed":False,
        "train_normalization_population_witness":{"path":population["path"],"file_sha256":population["sha256"]}},
        "contract_sha256")
    recipe=write("recipe.json",{"test_accessed":False,"normalization_view":view},"recipe_sha256")
    bundle=write("bundle.json",{},"bundle_sha256")
    ep={}; em={}; slots={}; manifests={}; child_paths={}
    for split in ("train","val"):
        ep[split]=Path(raw(split+".entry.parquet",b"synthetic entry")["path"])
        em[split]=write(split+".entry.manifest.json",{})
        cp=tmp_path/(split+".child.parquet")
        pd.DataFrame({"time":times[2:9]}).to_parquet(cp,index=False)
        child_paths[split]=cp
        sources={"parent_m1":parent,"parent_m1_manifest":parent_manifest,
            "final_recipe":recipe,"final_bindings_bundle":bundle,
            "parent_entry_parquet":{"path":str(ep[split]),"sha256":owner.sha256_file(ep[split])},
            "parent_entry_manifest":em[split],
            "m1_child":{"path":str(cp),"sha256":owner.sha256_file(cp)}}
        manifests[split]={"split":split,"parent_m1_row_offset":2,"source_bindings":sources}
        mp=write(split+".index.json",manifests[split],"manifest_sha256")
        md=json.loads(Path(mp["path"]).read_text())
        manifests[split]=md
        slots[split]={"manifest_path":mp["path"],"manifest_sha256":md["manifest_sha256"]}
    root=write("ROOT.json",{"splits":slots,
        "final_bindings_bundle_sha256":json.loads(Path(bundle["path"]).read_text())["bundle_sha256"]})
    monkeypatch.setattr(index_owner,"require_random_access_index_root",lambda x:x)
    monkeypatch.setattr(index_owner,"require_random_access_index_manifest",lambda x,**kw:x)
    calls=[]
    monkeypatch.setattr(owner,"require_exact_m1_feature_surface_manifest",lambda **kw:calls.append(("admission",kw)))
    prices={name:np.arange(len(times),dtype=float)+100 for name in ("open","high","low","close")}
    features={"signal":np.ones((len(times)-1,owner.MODEL_NATIVE_SIGNAL_DIM),dtype=np.float32),
              "ctx_cont":np.ones((len(times)-1,owner.MODEL_NATIVE_CTX_CONT_DIM),dtype=np.float32),
              "ctx_cat":np.zeros((len(times)-1,owner.MODEL_NATIVE_CTX_CAT_DIM),dtype=np.int64)}
    monkeypatch.setattr(owner,"_validated_m1_arrays",lambda p:(calls.append(("prices",p)) or (times,prices)))
    monkeypatch.setattr(owner,"load_m1_feature_surface",lambda p,**kw:(calls.append(("features",p)) or (times[1:],features)))
    return dict(owner=owner,kwargs={"root_manifest_path":Path(root["path"]),"entry_parquets":ep,
        "entry_manifest_bindings":em,"dataset_run_id":"SYNTHETIC","splits":("train","val")},
        root=root,manifests=manifests,times=times,features=features,calls=calls,feature=feature,
        child_paths=child_paths)


def test_index_feature_source_reuses_normalized_tape_for_both_splits(indexed_feature_source):
    c=indexed_feature_source
    corpus=c["owner"].UnifiedExitLifecycleCorpus.from_random_access_index(**c["kwargs"])
    assert [x[0] for x in c["calls"]]==["admission","prices","features"]
    for split in ("train","val"):
        source=corpus.splits[split]
        assert source._m1_times.equals(c["times"])
        assert source._m1_feature_times.equals(c["times"][1:])
        assert source._feature_row_offset==1
        assert source._m1["mid_close"] is source._m1["close"]
        assert source._m1_features["signal"] is c["features"]["signal"]
        assert not source._m1_features["signal"].flags.writeable
    assert corpus.evidence["test_accessed"] is False
    corpus._m1_feature_tempdir.cleanup()


@pytest.mark.parametrize("mutation",["feature_bytes","entry_binding","test_split","index_identity"])
def test_index_feature_source_rejects_changed_identity_before_arrays(indexed_feature_source,mutation):
    from pathlib import Path
    c=indexed_feature_source
    if mutation=="feature_bytes":Path(c["feature"]["path"]).write_bytes(b"changed")
    elif mutation=="entry_binding":c["kwargs"]["entry_manifest_bindings"]["train"]={"path":"/wrong","sha256":"0"*64}
    elif mutation=="test_split":c["kwargs"]["splits"]=("test",)
    else:
        p=Path(c["kwargs"]["root_manifest_path"]);d=json.loads(p.read_text())
        d["splits"]["train"]["manifest_sha256"]="0"*64;p.write_text(json.dumps(d))
    with pytest.raises(RuntimeError,match="UNIFIED_EXIT_INDEX_FEATURE"):
        c["owner"].UnifiedExitLifecycleCorpus.from_random_access_index(**c["kwargs"])
    assert not c["calls"]


def test_index_feature_source_rejects_old_feature_clock(indexed_feature_source,monkeypatch):
    import pandas as pd
    c=indexed_feature_source
    monkeypatch.setattr(c["owner"],"load_m1_feature_surface",
        lambda *a,**kw:(c["times"][1:]+pd.Timedelta(minutes=1),c["features"]))
    with pytest.raises(RuntimeError,match="FEATURE_CLOCK_INVALID"):
        c["owner"].UnifiedExitLifecycleCorpus.from_random_access_index(**c["kwargs"])


def test_index_feature_source_rejects_old_parent_coordinates(indexed_feature_source):
    from pathlib import Path
    c=indexed_feature_source
    root_path=c["kwargs"]["root_manifest_path"];root=json.loads(root_path.read_text())
    m=c["manifests"]["train"];m["parent_m1_row_offset"]=3
    m["manifest_sha256"]=c["owner"].canonical_json_sha256({k:v for k,v in m.items() if k!="manifest_sha256"})
    Path(root["splits"]["train"]["manifest_path"]).write_text(json.dumps(m))
    root["splits"]["train"]["manifest_sha256"]=m["manifest_sha256"];root_path.write_text(json.dumps(root))
    with pytest.raises(RuntimeError,match="RANDOM_ACCESS_PARENT_CLOCK_INVALID"):
        c["owner"].UnifiedExitLifecycleCorpus.from_random_access_index(**c["kwargs"])
