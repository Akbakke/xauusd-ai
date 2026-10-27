"""Fixed B16 bps route checks on synthetic metadata and a tiny stochastic shell.

No genuine native constructor, saved input, later cohort, or CUDA execution.
"""
import copy
from pathlib import Path

import pytest
import torch

from gx1.contracts import entry_native_direction_research_campaign_v1 as owner
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.scripts import run_entry_native_direction_bps_qualification_v1 as worker
from tests.test_entry_native_direction_research_campaign_v1 import fixture as legacy_fixture, write, reseal


@pytest.fixture
def bps_fixture(legacy_fixture, monkeypatch):
    f = legacy_fixture
    monkeypatch.setattr(owner, "BPS_EXTRA_SOURCE_PATHS", ())
    monkeypatch.setattr(owner, "BPS_DESIGN", f["dummy"])
    monkeypatch.setattr(owner, "BPS_GC_PARITY", f["dummy"])
    recipe = f["recipe"]
    recipe.update(run_scope=owner.BPS_SCOPE, fold=dict(owner.BPS_FOLD),
        qualification_design=f["dummy"], gc_layout_metadata_parity=f["dummy"],
        max_model_constructions=4, max_forwards=6, max_backwards=5)
    recipe["numerical_recipe"].update(batch_size=16, signed_bps_supervision=True)
    reseal(recipe["numerical_recipe"], "contract_sha256")
    reseal(recipe, "recipe_sha256")
    f["recipe_ref"] = write(f["runtime"]/"RECIPE.json", recipe)
    f["window"]["recipe"] = f["recipe_ref"]
    reseal(f["window"], "policy_sha256")
    f["window_ref"] = write(f["runtime"]/"WINDOW.json", f["window"])
    f["argv"][f["argv"].index("--window-policy-file-sha256")+1] = f["window_ref"]["sha256"]
    f["execution"].update(launcher_argv_sha256=campaign.canonical_sha256(f["argv"]),
        prelaunch_manifest=f["recipe_ref"], prelaunch_manifest_sha256=recipe["recipe_sha256"],
        train_session_manifest=f["window_ref"], train_session_manifest_sha256=f["window"]["policy_sha256"])
    reseal(f["execution"], "artifact_sha256")
    f["invocation"].update(batch_size=16, launcher_argv=f["argv"],
        launcher_argv_sha256=campaign.canonical_sha256(f["argv"]),
        execution_manifest=write(f["runtime"]/"EXECUTION.json", f["execution"]))
    reseal(f["invocation"], "invocation_sha256")
    f["plan"].update(recipe=f["recipe_ref"],
        invocations=[write(f["runtime"]/"INVOCATION.json", f["invocation"])])
    reseal(f["plan"], "plan_sha256")
    f["policy"][recipe["admission_stage"]].update(recipe=f["recipe_ref"],
        signed_bps_scale_fit_authorized=True)
    write(f["repo"]/"NEXT_RUN_POLICY.json", f["policy"])
    return f


def test_bps_chain_is_exact_b16_but_keeps_same_hardware_and_new_boot(bps_fixture):
    f = bps_fixture
    plan = campaign.require_plan(f["plan"])
    assert plan["checked_invocations"][0]["batch_size"] == 16
    ready = owner.require_native_direction_research_readiness(f["repo"],
        window_path=Path(f["window_ref"]["path"]), window_file_sha256=f["window_ref"]["sha256"])
    assert ready["decision"] == "READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES"
    assert campaign.next_action(plan, [], current_boot=f["boot"])["decision"] == "REBOOT_REQUIRED"
    newer = copy.deepcopy(f["boot"])
    newer.update(boot_id=2, last_boot_utc="2026-10-24T02:00:00+00:00")
    reseal(newer, "identity_sha256")
    assert campaign.next_action(plan, [], current_boot=newer)["decision"] == "LAUNCH"
    assert plan["policy"] == campaign.campaign_hardware_policy("native_candidate")


@pytest.mark.parametrize("fault", [
    "batch2", "signed_off", "five_models", "seven_forwards", "six_reverses",
    "later_prefix", "missing_parity", "wrong_design", "old_scope",
])
def test_resealed_bps_recipe_cannot_broaden_or_mix_profile(bps_fixture, fault):
    f = bps_fixture; item = copy.deepcopy(f["recipe"])
    if fault == "batch2": item["numerical_recipe"]["batch_size"] = 2
    elif fault == "signed_off": item["numerical_recipe"]["signed_bps_supervision"] = False
    elif fault == "five_models": item["max_model_constructions"] = 5
    elif fault == "seven_forwards": item["max_forwards"] = 7
    elif fault == "six_reverses": item["max_backwards"] = 6
    elif fault == "later_prefix": item["fold"]["prefix_rows"] += 1
    elif fault == "missing_parity": del item["gc_layout_metadata_parity"]
    elif fault == "wrong_design": item["qualification_design"]["sha256"] = "0"*64
    else: item["run_scope"] = "private_four_update_resume_qualification"
    reseal(item["numerical_recipe"], "contract_sha256")
    reseal(item, "recipe_sha256")
    ref = write(f["runtime"]/"BAD_RECIPE.json", item)
    with pytest.raises(RuntimeError):
        owner.require_research_launch_recipe(ref, source_repo=f["repo"], source_commit="a"*40)


def test_old_recipe_does_not_gain_b16_authority(legacy_fixture):
    f = legacy_fixture
    item = copy.deepcopy(f["invocation"]); item["batch_size"] = 16
    reseal(item, "invocation_sha256")
    with pytest.raises(RuntimeError):
        campaign.require_invocation(item, source_repo=f["repo"], source_commit="a"*40)


@pytest.mark.parametrize("fault", ["global_stop", "scale_not_admitted", "already_claimed"])
def test_bps_private_gates_still_bind(bps_fixture, fault):
    f = bps_fixture; p = copy.deepcopy(f["policy"])
    if fault == "global_stop": p["training_enabled"] = False
    elif fault == "scale_not_admitted": p[f["recipe"]["admission_stage"]]["signed_bps_scale_fit_authorized"] = False
    else: p[f["recipe"]["admission_stage"]]["claimed_invocations"] = 1
    write(f["repo"]/"NEXT_RUN_POLICY.json", p)
    with pytest.raises(RuntimeError):
        owner.require_native_direction_research_readiness(f["repo"],
            window_path=Path(f["window_ref"]["path"]), window_file_sha256=f["window_ref"]["sha256"])


def test_b16_cursor_and_progress_bind_32_parents_and_four_physical_updates(bps_fixture):
    f = bps_fixture; plan = campaign.require_plan(f["plan"]); inv = plan["checked_invocations"][0]
    state_ref = write(f["runtime"]/"STATE.synthetic.json", {"synthetic": True})
    result_ref = write(f["runtime"]/"RESULT.json", {"synthetic": True})
    cursor = owner.build_research_cursor(recipe_ref=f["recipe_ref"], outcome="COMPLETE",
        optimizer_updates=4, parent_cursor=32, parent_order_sha256="d"*64,
        full_state=state_ref, result=result_ref, batch_size=16)
    assert cursor["schema_version"] == owner.BPS_CURSOR_SCHEMA
    with pytest.raises(RuntimeError):
        owner.require_research_cursor(cursor, expected_recipe=f["recipe_ref"], batch_size=2)
    pointer = write(Path(f["window"]["campaign_cursor_path"]), cursor)
    progress = owner.build_research_progress(window=f["window"], plan=plan, invocation=inv,
        checkpoint_pointer=pointer, optimizer_updates=4, parent_cursor=32,
        parent_order_sha256="d"*64, outcome="COMPLETE")
    assert campaign.require_progress(progress, plan_sha256=plan["plan_sha256"],
        invocation=inv, expected_selection_receipt_sha256=None)["parent_cursor"] == 32
    cursor["parent_cursor"] = 16; reseal(cursor, "cursor_sha256")
    progress["checkpoint_pointer"] = write(Path(pointer["path"]), cursor)
    reseal(progress, "progress_sha256")
    with pytest.raises(RuntimeError):
        campaign.require_progress(progress, plan_sha256=plan["plan_sha256"],
            invocation=inv, expected_selection_receipt_sha256=None)


@pytest.mark.parametrize("updates,parents,outcome", [(3,32,"COMPLETE"), (4,16,"COMPLETE"),
    (5,32,"FAILED"), (0,33,"FAILED"), (True,0,"FAILED")])
def test_cursor_rejects_inconsistent_work(bps_fixture, updates, parents, outcome):
    f = bps_fixture
    with pytest.raises(RuntimeError):
        owner.build_research_cursor(recipe_ref=f["recipe_ref"], outcome=outcome,
            optimizer_updates=updates, parent_cursor=parents, parent_order_sha256="d"*64,
            full_state=f["dummy"], result=f["dummy"], batch_size=16)


def test_b16_early_failure_is_recordable_without_a_model(bps_fixture):
    f = bps_fixture
    cursor = owner.build_research_cursor(recipe_ref=f["recipe_ref"], outcome="FAILED",
        optimizer_updates=0, parent_cursor=0, parent_order_sha256="0"*64,
        full_state=None, result=f["dummy"], batch_size=16)
    assert cursor["outcome"] == "FAILED" and cursor["full_state"] is None


def test_direct_bps_call_checks_existing_cuda_guard_first(monkeypatch, tmp_path):
    from gx1.contracts import gx1_capped_execution_v1 as capped
    events = []
    def reject():
        events.append("guard"); raise RuntimeError("synthetic guard rejection")
    monkeypatch.setattr(capped, "require_guarded_cuda_trainer_execution", reject)
    monkeypatch.setattr(owner, "require_native_direction_research_context", lambda *args: events.append("context"))
    with pytest.raises(RuntimeError, match="synthetic guard"):
        worker.run_bps_qualification(policy_path=tmp_path/"missing.json",
            policy_file_sha256="0"*64, progress_path=tmp_path/"PROGRESS.json")
    assert events == ["guard"] and list(tmp_path.iterdir()) == []


def test_existing_dispatch_reaches_bps_only_after_guard_and_context(monkeypatch, tmp_path):
    from gx1.contracts import gx1_capped_execution_v1 as capped
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    events = []
    monkeypatch.setattr(capped, "require_guarded_cuda_trainer_execution", lambda: events.append("guard"))
    def context(*args):
        events.append("context"); return {}, {}, {}, {}, {"run_scope": owner.BPS_SCOPE}
    monkeypatch.setattr(owner, "require_native_direction_research_context", context)
    def execute(**kwargs):
        events.append("bps")
        assert kwargs == dict(policy_path=tmp_path/"W.json", policy_file_sha256="0"*64,
            progress_path=tmp_path/"P.json")
        return {"synthetic_dispatch": True}
    monkeypatch.setattr(worker, "run_bps_qualification", execute)
    assert common.run_native_direction_research_window(policy_path=tmp_path/"W.json",
        policy_file_sha256="0"*64, progress_path=tmp_path/"P.json") == {"synthetic_dispatch": True}
    assert events == ["guard", "context", "bps"]


def test_bps_probe_after_update_preserves_exact_b16_resume(monkeypatch):
    from gx1.contracts import entry_model_native_train_recipe_v1 as recipes
    from gx1.contracts import entry_native_direction_research_state_v1 as state
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from tests import test_entry_native_direction_bps_v1 as component
    from tests import test_entry_native_direction_research_state_v1 as states
    # Tiny shell has no GC modules. This check covers probe/update/resume mechanics;
    # the genuine unit separately requires every existing aggregate gradient group.
    def gradients(model, torch):
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        return {"synthetic_joint": int(torch.count_nonzero(model.backbone.entry_q_joint_in.weight.grad))}
    monkeypatch.setattr(common, "_gradient_evidence", gradients)
    def make():
        case = component._candidate(monkeypatch)
        case.recipe = recipes.native_direction_research_recipe(batch_size=16, max_updates=2, signed_bps=True)
        case.optimizer, case.binding = state.build_native_direction_research_optimizer(case.model, recipe=case.recipe)
        case.order = torch.arange(32, dtype=torch.int64)
        states._seed(case.recipe["seed"])
        return case
    counts = {k: 0 for k in ("forwards", "forward_returns", "backwards", "backward_returns",
        "optimizer_step_calls", "optimizer_updates")}
    def step(case, index, probe=False):
        features = {k: v.repeat((8,)+(1,)*(v.ndim-1)) for k,v in states._features(index).items()}
        target = component._loss([[.1,2.,50.],[-.2,-3.,-40.]]*8)
        cutoff = target.pop("cutoff_ns")
        for k in ("logit_up", "mean_return_bps", "rms_bps"): del target[k]
        return worker._step(case.model, case.optimizer, features, target,
            cutoff=cutoff, counters=counts, torch=torch, probe=probe)
    first = make()
    step(first, 0)
    saved = component._capture(first, 16, 1)
    evidence = step(first, 1, probe=True)
    expected = component._capture(first, 32, 2)
    assert evidence["bps_only_probe"]["joint_projection_nonzero"] > 0
    assert evidence["bps_eligible_cells"] == 48 and evidence["bps_horizon_counts"] == [16,16,16]
    resumed = make()
    state.restore_native_direction_research_state(saved, resumed.model, resumed.optimizer,
        recipe=resumed.recipe, feature_view_identity=resumed.model.feature_view_identity,
        parent_order=resumed.order, device=torch.device("cpu"))
    step(resumed, 1)
    assert component._capture(resumed, 32, 2)["contract_sha256"] == expected["contract_sha256"]
    assert all(p.grad is None for p in resumed.model.parameters())
    assert counts == {"forwards":3,"forward_returns":3,"backwards":4,"backward_returns":4,
        "optimizer_step_calls":3,"optimizer_updates":3}
