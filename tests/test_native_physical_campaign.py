"""Real campaign owners over synthetic current physical data, no model work."""
import json
from pathlib import Path

import pytest

from gx1.contracts import unified_exit_gpu_batch_selection_v1 as gpu
from tests.test_native_prefix_recipe import (
    prefix_scope, scope, native, campaign, materializer, runner, window, _boot, _bind, _write, _cursor,
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
