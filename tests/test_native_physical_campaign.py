"""Real campaign owners over synthetic current physical data, no model work."""
import json
from pathlib import Path

import pytest

from gx1.contracts import unified_exit_gpu_batch_selection_v1 as gpu
from tests.test_native_prefix_recipe import (
    prefix_scope, scope, physical_recipe, physical_normalization_templates, native, campaign, materializer, runner, window, _boot, _bind, _write, _cursor,
)
from tests.test_native_fresh_prefix_components import physical_component_templates, physical_component_case


@pytest.fixture
def physical_campaign(prefix_scope, physical_component_case, tmp_path, monkeypatch):
    policy, recipe, files, seal = prefix_scope
    case = physical_component_case
    recipe["chronological_prefix"] = case["value"]
    recipe["files"].update({key: _bind(path) for key, path in case["files"].items() if path.is_file()})
    policy["chronological_learning_run"]["chronological_prefix"] = case["value"]
    rb = seal()
    repo = Path(recipe["source_repo"])
    monkeypatch.setattr(materializer, "_source_commit", lambda _: recipe["source_commit"])
    guards, controllers = {}, {}
    for names, target in [(("runner", "guard", "query", "certificate"), guards),
                          (("controller", "observer", "campaign_cli"), controllers)]:
        for name in names:
            path = repo / "sources" / name
            path.parent.mkdir(exist_ok=True)
            path.write_text(name)
            target[name] = _bind(path)
    (repo / "scripts").mkdir()
    (repo / "scripts/gx1_capped_run.sh").write_text("synthetic")
    (repo / ".venv/bin").mkdir(parents=True)
    (repo / ".venv/bin/python").write_text("synthetic")
    monkeypatch.setattr(materializer, "_sources", lambda *args: (guards, controllers))

    def forbidden(*args, **kwargs):
        pytest.fail("Historical GPU/model authority reached from fresh physical campaign")
    monkeypatch.setattr(materializer, "require_selection", forbidden)
    monkeypatch.setattr(gpu, "require_selection", forbidden)
    monkeypatch.setattr(native, "require_native_completed_smoke", forbidden)
    monkeypatch.setattr(runner.val, "_model", forbidden)
    boot = tmp_path / "boot.json"
    _write(boot, _boot(100, 0))
    args = dict(repo=repo, output=tmp_path / "campaign", runtime=tmp_path / "runtime",
                gpu_uuid="GPU-fixture", prepared_boot_path=boot,
                prepared_boot_file_sha256=native.file_sha256(boot),
                certificate_path=Path(guards["certificate"]["path"]),
                recipe_path=Path(rb["path"]), recipe_file_sha256=rb["sha256"], window_count=1)
    return policy, recipe, case, seal, args


def test_current_physical_campaign_requires_no_historical_gpu_or_model_chain(physical_campaign, tmp_path, monkeypatch):
    _, recipe, case, _, args = physical_campaign
    result = materializer.materialize_native_candidate_campaign(**args)
    plan = result["plan"]
    assert plan["prior_campaign"] is plan["final_train_checkpoint_authority"] is None
    assert plan["selection_receipt"] == case["coordinates"]["selected_sampler"]
    assert plan["selection_artifact_sha256"] == case["selected"]["artifact_sha256"]
    assert plan["entry_pairs_per_epoch"] == 32768
    invocation = plan["checked_invocations"][0]
    assert invocation["maximum_wall_seconds"] == 13800
    assert invocation["requires_fresh_windows_boot"] is invocation["signed_guard_only"] is True
    wp = invocation["native_window_policy"]
    rb = {"path": str(args["recipe_path"]), "sha256": args["recipe_file_sha256"]}
    position, _ = _cursor(tmp_path, rb, 256)
    monkeypatch.setattr(window.native.trainer, "_require_cuda_trainer_guard_execution", lambda **kw: None)
    monkeypatch.setattr(window, "_context", lambda *a: (wp, plan, invocation, {}))
    monkeypatch.setattr(window, "_expected_training_pointer", lambda **kw: None)
    def run(**kw):
        budget = json.loads(Path(kw["execution_budget_path"]).read_text())
        assert budget["stop_after_optimizer_steps"] == 256
        assert budget["stop_after_completed_val_epochs"] is None
        return {"decision": "PAUSED_RESUMABLE", "resume_state": position}
    monkeypatch.setattr(window.native, "run_guarded_native_candidate_invocation", run)
    progress = window.run_window(policy_path=tmp_path/"unused", policy_file_sha256="a"*64,
                                progress_path=Path(wp["progress_path"]))
    payload = json.loads(Path(progress["progress"]["path"]).read_text())
    assert payload["selection_receipt_sha256"] == case["selected"]["artifact_sha256"]


@pytest.mark.parametrize("fault", ["historical_prior", "old_seed", "sampler_copy", "missing_coordinates",
                                   "full_training", "full_val", "excess_windows", "unsafe_guard"])
def test_current_physical_campaign_rejects_mixed_or_unbounded_provenance(physical_campaign, fault):
    policy, recipe, case, seal, args = physical_campaign
    if fault in {"historical_prior", "old_seed", "sampler_copy", "unsafe_guard"}:
        result = materializer.materialize_native_candidate_campaign(**args)
        raw = json.loads(Path(result["path"]).read_text())
        if fault == "historical_prior": raw["prior_campaign"] = raw["selection_receipt"]
        elif fault == "old_seed": raw["final_train_checkpoint_authority"] = raw["selection_receipt"]
        elif fault == "sampler_copy":
            source = Path(raw["selection_receipt"]["path"])
            other = source.with_name("COPY.json")
            other.write_bytes(source.read_bytes())
            raw["selection_receipt"] = _bind(other)
        else: raw["policy"]["maximum_memory_junction_temperature_c"] = 90
        raw["plan_sha256"] = campaign.canonical_sha256({k:v for k,v in raw.items() if k != "plan_sha256"})
        with pytest.raises(RuntimeError): campaign.require_plan(raw, verify_files=True)
    else:
        if fault == "missing_coordinates": recipe["chronological_prefix"].pop("native_coordinates")
        elif fault == "full_training": policy["training_enabled"] = True
        elif fault == "full_val": policy["chronological_learning_run"]["full_val_allowed"] = True
        elif fault == "excess_windows": args["window_count"] = 4
        rb = seal()
        args["recipe_file_sha256"] = rb["sha256"]
        with pytest.raises(RuntimeError): materializer.materialize_native_candidate_campaign(**args)
        assert not args["output"].exists()


def _dynamic_preprocessing_case(case, tmp_path, monkeypatch):
    """Direct join fixture; each strict source validator has separate real-data tests."""
    import copy
    from gx1.contracts import entry_full_input_liveness_v1 as live
    from gx1.contracts import entry_sequence_source_reconstruction_v1 as reconstruction
    from gx1.contracts import entry_model_native_input_normalization_v1 as normalizer
    from gx1.contracts.entry_model_native_aux_targets_v3 import (
        M5_DYNAMIC_PHYSICAL_TARGET_BINDING_SCHEMA, M5_DYNAMIC_OBSERVED_EMISSION_SCOPE,
    )
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    from gx1.contracts.entry_model_native_signal_v1 import model_native_signal_contract_metadata
    from gx1.contracts.entry_observed_market_v1 import entry_observed_dynamic_m5_path_contract
    from gx1.contracts.entry_exit_production_architecture_v1 import PRODUCTION_MTF_PER_TF_WINDOW_BARS
    from gx1.features import gc_native_context_v1 as gc
    from gx1.models.entry_v10.entry_v10_input_normalization import FIT_POPULATION_PROOF_SCHEMA_VERSION, _canonical_sha256
    from tests.test_native_prefix_recipe import _dynamic_single_epoch_design
    from tests.model_native_signal_support import canonical_model_native_selected_fields
    design = copy.deepcopy(case["design"]); n = case["sources"]["train"]["physical_rows"]
    population_design, witness = _dynamic_single_epoch_design(tmp_path, rows=n)
    sources = design["calendar"]["source_bindings"]
    monkeypatch.setitem(gc.REGISTRY, "train_decision_rows", n)
    monkeypatch.setitem(gc.REGISTRY, "train_decision_clock_sha256", "c" * 64)
    train_manifest = json.loads(Path(witness["train_manifest"]["path"]).read_text())
    signal = model_native_signal_contract_metadata(canonical_model_native_selected_fields())
    for split, source in sources.items():
        manifest = copy.deepcopy(train_manifest) if split == "train" else {"extra": {}}
        manifest["output_data_path"] = source["parquet"]["path"]
        manifest["extra"].update(rows=source["physical_rows"], emission_scope=M5_DYNAMIC_OBSERVED_EMISSION_SCOPE,
            model_native_signal_contract=signal, gc_native_context=gc.contract())
        source["manifest"] = _write(tmp_path / ("dynamic-physical-" + split + ".manifest.json"), manifest)
    witness.update(train_parquet=sources["train"]["parquet"], train_manifest=sources["train"]["manifest"],
                   entry_eligibility_cutoff_utc=design["calendar"]["train_control_cutoff"])
    design["scope"].update(population_design["scope"])
    design["scope"]["entry_m5_path_single_epoch"] = _write(tmp_path / "dynamic-physical-population.json", witness)
    design["budget"].update(population_design["budget"])
    audits = {split: _write(tmp_path / (split + "-current-reconstruction.json"), {"split": split}) for split in sources}
    clock = {"exact_match": True, "rows": n, "population": "whole_physical_TRAIN_before_target_availability_selection",
        "physical_train_parquet_path": sources["train"]["parquet"]["path"],
        "physical_train_bar_clock_sha256": sources["train"]["clock_sha256"],
        "registry_path": gc.REGISTRY_PATH, "registry_sha256": gc.REGISTRY_SHA256,
        "train_decision_clock_sha256": "c" * 64}
    live_clock = {**clock, "physical_train_parquet_sha256": sources["train"]["parquet"]["sha256"],
        "registry": {"path": gc.REGISTRY_PATH, "sha256": gc.REGISTRY_SHA256}}
    liveness = _write(tmp_path / "current-input-liveness.json", {
        "materializer_provenance": {"gc_registry_whole_train_clock": live_clock}})
    design["inputs"] = {"full_input_liveness": liveness, "sequence_source_audits": audits}
    binding = _write(tmp_path / "dynamic-physical-design.json", design)
    labels = {"schema_version": M5_DYNAMIC_PHYSICAL_TARGET_BINDING_SCHEMA,
        "design": binding, "training_phase": ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        "emission_scope": M5_DYNAMIC_OBSERVED_EMISSION_SCOPE,
        "target_contract": entry_observed_dynamic_m5_path_contract(), "splits": {}}
    for split, source in sources.items():
        labels["splits"][split] = {"source_parquet": source["parquet"], "source_manifest": source["manifest"],
            "rows": source["physical_rows"], "first_prefix_observed_rows": source["physical_rows"],
            "physical_clock_sha256": source["clock_sha256"],
            "dynamic_m5_supervision": {"source_quote_binding": {"sha256": "a" * 64},
                "index_binding": {"split_end_exclusive": design["calendar"][
                    "train_control_cutoff" if split == "train" else "development_control_entry_end_exclusive"]}}}
    lineage = {"train_parquet_path": sources["train"]["parquet"]["path"],
        "train_parquet_sha256": sources["train"]["parquet"]["sha256"],
        "train_manifest_path": sources["train"]["manifest"]["path"],
        "train_manifest_sha256": sources["train"]["manifest"]["sha256"],
        "train_row_count": n, "entry_train_decision_row_count": n, "context_fit_row_count": n,
        "exit_train_decision_row_count": 0, "val_fit_row_count": 0, "test_fit_row_count": 0,
        "local_fit_row_count": n + 95, "per_tf_seq_lens": dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS)}
    contract = {"contract_sha256": "d" * 64, "lineage": lineage,
        "fit_population": normalizer.DYNAMIC_M5_FIT_POPULATION,
        "fit_start_utc": design["calendar"]["train_entry_start_inclusive"],
        "fit_end_utc": sources["train"]["declared_window"]["end"]}
    windows = {tf: {"selected_unique_row_count": n + seq_len} for tf, seq_len in PRODUCTION_MTF_PER_TF_WINDOW_BARS}
    lineage["per_tf_fit_windows"] = windows
    contract["surfaces"] = {"signal": {"fit_row_count": n + 95}, "ctx_cont": {"fit_row_count": n},
        **{"mtf_" + tf.lower(): {"fit_row_count": value["selected_unique_row_count"]} for tf, value in windows.items()}}
    proof = {"schema_version": FIT_POPULATION_PROOF_SCHEMA_VERSION, "entry_training_phase": ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        "fit_population": normalizer.DYNAMIC_M5_FIT_POPULATION, "fit_scope": "train_only",
        "train_decision_row_count": n, "entry_train_decision_row_count": n, "context_fit_row_count": n,
        "exit_train_decision_row_count": 0, "val_fit_row_count": 0, "test_fit_row_count": 0,
        "local_fit_row_count": n + 95, "local_populations": {"entry": {"selected_unique_row_count": n + 95}}, "context_populations": {"entry": {}},
        "mtf_populations": {tf: {"seq_len": seq_len, "selected_unique_row_count": windows[tf]["selected_unique_row_count"],
            "routes": {"entry": {"enabled": True, "decision_row_count": n},
                       "exit": {"enabled": False, "decision_row_count": 0}}} for tf, seq_len in PRODUCTION_MTF_PER_TF_WINDOW_BARS},
        "gc_registry_whole_train_clock": clock,
        "entry_sequence_source_provenance": {"audit_path": audits["train"]["path"], "audit_sha256": audits["train"]["sha256"],
            "audit": {"split": "train"}}}
    proof["proof_sha256"] = _canonical_sha256(proof)
    normalization = {"schema_version": "gx1_entry_only_dynamic_m5_base_normalization_v1", "decision": "PASS",
        "entry_training_phase": ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE, "contract": contract,
        "contract_sha256": contract["contract_sha256"], "fit_population_proof": proof,
        "full_input_liveness": liveness, "sequence_source_audit": audits["train"],
        "val_fit_rows": 0, "test_fit_rows": 0, "test_accessed": False, "native_training_admitted": False}
    def checked_normalization(value, **kwargs):
        assert kwargs["expected_entry_training_phase"] == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
        assert kwargs["expected_field_names"]["signal"] == signal["fields"]
        return dict(value)
    def checked_liveness(path, **kwargs):
        assert kwargs["expected_manifest_bindings"] == {split: source["manifest"] for split, source in sources.items()}
        assert kwargs["expected_sha256"] == native.file_sha256(Path(path))
        return {"ok": True}
    def checked_reconstruction(value, **kwargs):
        split = value["split"]; source = sources[split]
        assert str(kwargs["expected_parquet_path"]) == source["parquet"]["path"]
        assert kwargs["expected_parquet_sha256"] == source["parquet"]["sha256"]
        assert str(kwargs["expected_manifest_path"]) == source["manifest"]["path"]
        assert kwargs["expected_manifest_sha256"] == source["manifest"]["sha256"]
        assert kwargs["expected_rows"] == source["physical_rows"]
        return dict(value)
    monkeypatch.setattr(normalizer, "require_input_normalization_contract", checked_normalization)
    monkeypatch.setattr(live, "validate_full_input_liveness_artifact", checked_liveness)
    monkeypatch.setattr(reconstruction, "require_sequence_source_reconstruction_audit", checked_reconstruction)
    artifacts = {**case["recipe"]["chronological_prefix"], "design": binding,
        "labels_result": _write(tmp_path / "dynamic-label-dispatch.json", labels),
        "normalization_result": _write(tmp_path / "dynamic-normalization.json", normalization)}
    return artifacts, design, labels


def test_dynamic_physical_dispatch_uses_strict_receipt_and_never_inactive_target_union(physical_recipe, tmp_path, monkeypatch):
    """Dispatch only; the aux owner separately tests its actual pointer/mask scan."""
    import pyarrow.parquet as pq
    from gx1.contracts import entry_model_native_aux_targets_v3 as auxiliary
    artifacts, design, labels = _dynamic_preprocessing_case(physical_recipe, tmp_path, monkeypatch)
    seen = []
    def checked_receipt(value, *, design_binding, context):
        assert value is labels and design_binding == artifacts["design"]
        seen.append(context)
        return dict(value)
    monkeypatch.setattr(auxiliary, "require_m5_dynamic_physical_target_receipt", checked_receipt)
    def forbidden(*args, **kwargs):
        raise AssertionError("Inactive fixed95/legacy target table was read")
    monkeypatch.setattr(pq, "read_table", forbidden)
    checked = native.require_physical_chronological_preprocessing(artifacts, design=design,
        normalization=json.loads(Path(artifacts["normalization_result"]["path"]).read_text()), labels=labels)
    assert seen == ["NATIVE_PHYSICAL_PREPROCESSING"]
    assert checked["train_rows"] == physical_recipe["sources"]["train"]["physical_rows"]
    assert checked["native_coordinates_bound"] is False
    assert checked["entry_only_base_normalization"] == artifacts["normalization_result"]
    assert checked["sequence_source_audits"] == design["inputs"]["sequence_source_audits"]
    assert checked["full_input_liveness"] == design["inputs"]["full_input_liveness"]
    assert not {"composite_normalization", "summary_normalization_sha256", "composite_normalization_sha256"}.intersection(checked)
    assert checked["input_admission_complete"] is checked["native_training_admitted"] is False


@pytest.mark.parametrize("fault", ["old_phase", "old_emission", "source", "rows", "clock", "cutoff", "shared_quote"])
def test_dynamic_physical_dispatch_keeps_source_clock_and_scope_guards(physical_recipe, tmp_path, monkeypatch, fault):
    from gx1.contracts import entry_model_native_aux_targets_v3 as auxiliary
    artifacts, design, labels = _dynamic_preprocessing_case(physical_recipe, tmp_path, monkeypatch)
    # The dedicated aux tests prove the strict scanner. These mutations must
    # also fail this caller's independently bound physical/normalization join.
    monkeypatch.setattr(auxiliary, "require_m5_dynamic_physical_target_receipt", lambda value, **kwargs: dict(value))
    if fault == "old_phase": design["scope"]["entry_only_training_phase"] = "entry_only_observed_m5_path"
    elif fault == "old_emission": design["scope"]["emission_scope"] = "m5_primary_observed"
    elif fault == "source": labels["splits"]["train"]["source_parquet"] = labels["splits"]["val"]["source_parquet"]
    elif fault == "rows": labels["splits"]["train"]["first_prefix_observed_rows"] -= 1
    elif fault == "clock": labels["splits"]["train"]["physical_clock_sha256"] = "0" * 64
    elif fault == "cutoff": labels["splits"]["train"]["dynamic_m5_supervision"]["index_binding"]["split_end_exclusive"] = "2022-01-01T00:00Z"
    elif fault == "shared_quote": labels["splits"]["val"]["dynamic_m5_supervision"]["source_quote_binding"]["sha256"] = "b" * 64
    artifacts["design"] = _write(Path(artifacts["design"]["path"]), design)
    labels["design"] = artifacts["design"]
    artifacts["labels_result"] = _write(Path(artifacts["labels_result"]["path"]), labels)
    with pytest.raises(RuntimeError):
        native.require_physical_chronological_preprocessing(artifacts, design=design,
            normalization=json.loads(Path(artifacts["normalization_result"]["path"]).read_text()), labels=labels)


def test_dynamic_phase_cannot_relabel_legacy_complete_physical_receipt(physical_recipe):
    import copy
    design = copy.deepcopy(physical_recipe["design"])
    design["scope"].update(entry_only_training_phase="entry_only_dynamic_m5_path", emission_scope="m5_dynamic_observed")
    with pytest.raises(RuntimeError, match="DYNAMIC_M5_PHASE_RECEIPT_MISMATCH"):
        native.require_physical_chronological_preprocessing(physical_recipe["recipe"]["chronological_prefix"],
            design=design, normalization=physical_recipe["normalization"], labels=physical_recipe["labels"])


@pytest.mark.parametrize("fault", ["legacy_composite", "normalization_phase", "exit_rows", "train_subset", "val_fit",
    "fit_population", "proof_hash", "registry_clock", "bar_clock", "source_audit", "liveness_binding", "argument_not_bound", "mtf_exit", "local_population"])
def test_dynamic_preprocessing_rejects_wrong_normalization_or_input_proof(physical_recipe, tmp_path, monkeypatch, fault):
    from gx1.contracts import entry_model_native_aux_targets_v3 as auxiliary
    from gx1.models.entry_v10.entry_v10_input_normalization import _canonical_sha256
    artifacts, design, labels = _dynamic_preprocessing_case(physical_recipe, tmp_path, monkeypatch)
    monkeypatch.setattr(auxiliary, "require_m5_dynamic_physical_target_receipt", lambda value, **kwargs: dict(value))
    normalization = json.loads(Path(artifacts["normalization_result"]["path"]).read_text())
    proof = normalization["fit_population_proof"]
    if fault == "legacy_composite": normalization = physical_recipe["normalization"]
    elif fault == "normalization_phase": normalization["entry_training_phase"] = "joint_fitted_q"
    elif fault == "exit_rows": normalization["contract"]["lineage"]["exit_train_decision_row_count"] = 1
    elif fault == "train_subset": normalization["contract"]["lineage"]["entry_train_decision_row_count"] -= 1
    elif fault == "val_fit": normalization["val_fit_rows"] = 1
    elif fault == "fit_population": normalization["contract"]["fit_population"] = "legacy_shared_entry_exit"
    elif fault == "proof_hash": proof["proof_sha256"] = "0" * 64
    elif fault == "registry_clock": proof["gc_registry_whole_train_clock"]["train_decision_clock_sha256"] = "0" * 64
    elif fault == "bar_clock": proof["gc_registry_whole_train_clock"]["physical_train_bar_clock_sha256"] = "0" * 64
    elif fault == "source_audit": proof["entry_sequence_source_provenance"]["audit_sha256"] = "0" * 64
    elif fault == "liveness_binding": normalization["full_input_liveness"] = artifacts["labels_result"]
    elif fault == "argument_not_bound": normalization["test_accessed"] = True
    elif fault == "mtf_exit": proof["mtf_populations"]["H4"]["routes"]["exit"]["enabled"] = True
    elif fault == "local_population": proof["local_populations"]["entry"]["selected_unique_row_count"] -= 1
    if fault != "proof_hash":
        proof["proof_sha256"] = _canonical_sha256({k: v for k, v in proof.items() if k != "proof_sha256"})
    if fault != "argument_not_bound":
        artifacts["normalization_result"] = _write(Path(artifacts["normalization_result"]["path"]), normalization)
    with pytest.raises(RuntimeError, match="NATIVE_PHYSICAL_PREPROCESSING_DYNAMIC"):
        native.require_physical_chronological_preprocessing(artifacts, design=design, normalization=normalization, labels=labels)
