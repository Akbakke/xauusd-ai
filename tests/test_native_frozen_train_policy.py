"""Frozen TRAIN admission and dispatch; all mutation fixtures live under tmp_path."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from gx1.contracts import unified_exit_native_candidate_campaign_v1 as native
from gx1.contracts import unified_exit_random_access_val_checkpoint_v1 as checkpoints
from gx1.contracts import unified_exit_bounded_val_cohort_v1 as cohorts
from gx1.scripts import run_unified_exit_random_access_full_train_v1 as runner
from tests.test_native_learning_calibration_scope import scope
from tests.test_native_prefix_recipe import prefix_scope, _cursor
from tests.test_native_fresh_prefix_components import _bind, _write, component_chain
from tests.test_unified_exit_random_access_val_rollout_v1 import _Policy

@pytest.fixture
def frozen_train(prefix_scope,tmp_path,monkeypatch):
    policy,recipe,files,seal=prefix_scope
    assert Path(native.__file__).is_relative_to(tmp_path)
    monkeypatch.setattr(checkpoints,"__file__",str(tmp_path/"gx1/contracts/checkpoint.py"))
    sources={}
    for name in ("gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py",
                 "gx1/models/entry_v10/entry_v10_ctx_train_v3.py",
                 "gx1/contracts/unified_exit_random_access_model_v1.py",
                 "gx1/contracts/unified_exit_random_access_training_v1.py"):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("# synthetic frozen function\n")
        sources["python:"+name]=_bind(path)
    origin=copy.deepcopy(recipe);origin["source_bindings"]=sources
    rb=_write(tmp_path/"origin-recipe.json",origin)
    state,cursor=_cursor(tmp_path,rb,512)
    model=_Policy();target=copy.deepcopy(model)
    with torch.no_grad():model.bias.add_(0.5)
    sp=tmp_path/"synthetic-session/candidate_training_state_slot_0.pt"
    torch.save({"schema_version":"gx1_candidate_training_session_v1","global_optimizer_steps":512,
        "model_state":model.state_dict(),"target_model_state":target.state_dict()},sp)
    state["training_state"]=_bind(sp)
    pp=tmp_path/"synthetic-session/CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json"
    pointer=json.loads(pp.read_text());pointer["state_sha256"]=state["training_state"]["sha256"]
    state["training_pointer"]=_write(pp,pointer)
    cursor=_write(tmp_path/"cursor.json",native.build_native_cursor(recipe=rb,resume_state=state,outcome="RESUMABLE"))
    cohort={"schema_version":cohorts.TRAIN_ROLLOUT_SCHEMA,"split":"train","source_split":"train",
        "measurement_only":False,"plan":recipe["chronological_prefix"]["design"],"measurement_coordinates":{},
        "evaluation_role":"chronological_training_policy_rollout","entry_row_indices":list(range(256)),
        "parent_entry_row_indices":list(range(10000,10256)),"population_rows":5000,
        "source_index":{"path":"/synthetic/train-index","sha256":"a"*64},
        "observation_cutoff_time_ns":1772323200000000000,"test_data_used":False}
    cohort["cohort_sha256"]=cohorts.canonical_sha256(cohort)
    monkeypatch.setattr(cohorts,"build_chronological_train_rollout_cohort",lambda *a:copy.deepcopy(cohort))
    cb=_write(tmp_path/"cohort.json",cohort)
    completion={"schema_version":"gx1_convergence512_completion_v1","learning_review_complete":True,
        "optimizer_steps":512,"teacher_refreshed":False,"test_data_used":False,"native_recipe":rb,
        "training_state":state["training_state"],"training_pointer":state["training_pointer"],
        "model_state_sha256":checkpoints.canonical_model_state_sha256(model.state_dict()),
        "target_model_state_sha256":checkpoints.canonical_model_state_sha256(target.state_dict()),
        "model_functions":{"online":"main_encoder_and_fuse_final_layernorm_no_affine_v1","target":"main_encoder_no_final_norm_v1"},
        "train_observation":_write(tmp_path/"train-observation.json",{})}
    completed=_write(tmp_path/"completion.json",completion)
    footprint=_write(tmp_path/"footprint.json",{"schema_version":"gx1_frozen_exit_train_policy_footprint_v1",
        "completed512":completed,"cohort":cb,"frozen_training_state":state["training_state"],
        "source_index":cohort["source_index"],"test_data_used":False,
        "bounded_native_policy_forwards_upper_bound_at_batch256":4,"bounded_state_views_upper_bound":1024})
    limits={"max_model_forwards":4,"max_state_views":1024,"max_wall_seconds":10800,
        "policy_batch_size":256,"cpu_pipeline_workers":8,"progress_interval_forwards":64}
    plan={"schema_version":"gx1_frozen_train_policy_evaluation_plan_v1",
        "decision":"FROZEN_TRAIN_POLICY_AND_COHORT_NOT_LAUNCH_AUTHORITY",
        "optimizer_steps":0,"origin_optimizer_steps":512,"max_invocations":1,"control_forwards":0,
        "training_enabled":False,"test_data_used":False,"selected_model_variant":"ONLINE",
        "teacher_refresh_allowed":False,"full_epoch_allowed":False,"full_val_allowed":False,"native_launch_authorized":False,
        "completed512":completed,"footprint":footprint,"cohort":cb,"origin_cursor":cursor,
        "val_limits":limits,"review_criteria":{"open_positions":"include"}}
    pb=_write(tmp_path/"plan.json",plan)
    policy.pop("chronological_learning_run")
    recipe["frozen_train_policy_evaluation"]=pb;recipe["val_limits"]=limits
    policy["frozen_train_policy_evaluation"]={"plan":pb,"chronological_prefix":recipe["chronological_prefix"],
        "run_id":recipe["run_id"],"out_bundle_dir":recipe["out_bundle_dir"],
        "source_bindings_sha256":recipe["source_bindings_sha256"],"optimizer_steps":0,"origin_optimizer_steps":512,
        "max_invocations":1,"train_entries":256,"control_forwards":0,"val_limits":limits,"test_data_used":False}
    seal()
    return policy,recipe,seal,state,model,cohort


def test_scope_exact_restore_and_immutable_origin(frozen_train):
    _,recipe,_,state,expected,_=frozen_train
    budget={"stop_after_optimizer_steps":512,"max_invocation_seconds":12000,
        "expected_active_pointer_sha256":None,"stop_after_completed_val_epochs":None}
    assert native.require_native_run_scope(recipe,invocation_number=1,execution_budget=budget)==512
    for bad in [0,2,True]:
        with pytest.raises(RuntimeError,match="INVOCATION_INVALID"):
            native.require_native_run_scope(recipe,invocation_number=bad)
    for bad in [0,513,512.0,True]:
        with pytest.raises(RuntimeError,match="BUDGET_INVALID"):
            native.require_native_run_scope(recipe,execution_budget={**budget,"stop_after_optimizer_steps":bad})
    model=_Policy();rng=torch.get_rng_state().clone()
    binding=checkpoints.bind_frozen_train_policy_checkpoint_v1(plan_binding=recipe["frozen_train_policy_evaluation"],model=model)
    assert torch.equal(model.bias,expected.bias) and not model.training and not model.bias.requires_grad
    assert torch.equal(rng,torch.get_rng_state()) and binding["model_variant"]=="frozen_online_train_policy"
    assert checkpoints.require_selected_weight_ema_checkpoint_binding_v1(binding)==binding
    assert _bind(Path(state["training_state"]["path"]))==state["training_state"]
    with pytest.raises(RuntimeError,match="BINDING_MISMATCH"):
        checkpoints.require_selected_weight_ema_checkpoint_binding_v1({**binding,"model_state_sha256":"0"*64})


@pytest.mark.parametrize("fault",["training","mixed","budget","source","checkpoint"])
def test_rejects_scope_or_temporary_fixture_drift(frozen_train,tmp_path,fault):
    policy,recipe,seal,_,_,_=frozen_train
    if fault=="training":policy["training_enabled"]=True
    elif fault=="mixed":recipe["chronological_learning_measurement"]={}
    elif fault=="budget":recipe["val_limits"]={**recipe["val_limits"],"max_model_forwards":5}
    else:
        target=tmp_path/("gx1/contracts/unified_exit_random_access_model_v1.py" if fault=="source"
                         else "synthetic-session/candidate_training_state_slot_0.pt")
        assert target.resolve().is_relative_to(tmp_path.resolve())
        target.write_bytes(b"changed synthetic fixture")
    seal()
    with pytest.raises(RuntimeError):native.require_native_run_scope(recipe,invocation_number=1)


def test_guarded_dispatch_never_calls_trainer(frozen_train,tmp_path,monkeypatch):
    _,recipe,_,state,model,cohort=frozen_train
    context={"frame":pd.DataFrame({"entry_row_index":cohort["entry_row_indices"],"parent_entry_row_index":cohort["parent_entry_row_indices"]}),
        "state_factory":SimpleNamespace(source_split="train"),"evaluation_cohort":cohort,
        "parent_coordinate_evidence":{},"val_sequence_audit":tmp_path/"audit",**recipe["val_limits"]}
    probe=object();components={"model":model,"train_probe_ds":probe,"val_ds":object(),"native_val_context":context}
    budget={"stop_after_optimizer_steps":512,"max_invocation_seconds":12000,
        "expected_active_pointer_sha256":None,"stop_after_completed_val_epochs":None}
    calls=[]
    monkeypatch.setattr(runner.trainer,"_require_cuda_trainer_guard_execution",lambda **kw:calls.append("guard"))
    monkeypatch.setattr(runner.trainer,"_resolve_device",lambda _:torch.device("cuda"))
    monkeypatch.setattr(runner,"_require_native_full_train_recipe",lambda *a:(recipe,{},{}))
    monkeypatch.setattr(runner.launch_owner,"require_candidate_execution_budget",lambda *a,**kw:budget)
    def build(**kw):
        assert kw["frozen_train_policy_scope"]["cohort"]==cohort and kw["val_limits"]==recipe["val_limits"]
        return components
    monkeypatch.setattr(runner,"_build_bound_full_train_components",build)
    monkeypatch.setattr(runner.trainer,"_copy_frozen_prefix_reference_model",lambda m:copy.deepcopy(m))
    def forbidden(**kw):pytest.fail("frozen evaluation reached the trainer")
    monkeypatch.setattr(runner,"_run_bound_full_train_candidate",forbidden)
    def evaluate(**kw):
        assert kw["entry_dataset"] is probe and kw["evaluation_cohort"]==cohort
        assert kw["exit_policy_batch_size"]==256 and kw["cpu_pipeline_workers"]==8
        assert kw["compute_guard_max_model_forwards"]==4 and kw["compute_guard_max_materialized_state_views"]==1024
        assert "candidate_target_model" not in kw and "exit_boundary_model" not in kw
        assert kw["checkpoint_binding"]["model_variant"]=="frozen_online_train_policy"
        for key in ("rollout_progress_path","result_path"):kw[key].write_text("{}")
        calls.append("evaluate");return {"decision":"COMPLETE_WITH_RIGHT_CENSORING"}
    monkeypatch.setattr(runner.val,"evaluate_bound_full_val_v1",evaluate)
    result=runner.run_guarded_native_candidate_invocation(recipe_path=tmp_path/"recipe.json",recipe_file_sha256="a"*64,
        execution_budget_path=tmp_path/"budget.json",execution_budget_file_sha256="b"*64)
    assert calls==["guard","evaluate"] and result["resume_state"]==state and result["bundle_written"] is False
    observation=json.loads(Path(result["observation"]["path"]).read_text())
    assert observation["optimizer_steps"]==0 and observation["training_enabled"] is False


@pytest.mark.parametrize("bad_budget",[False,True])
def test_components_use_train_labels_and_measured_support(component_chain,tmp_path,monkeypatch,bad_budget):
    args,seen,_,_=component_chain
    root=runner.val._read(args["files"]["random_access_root"])
    ip=Path(root["splits"]["train"]["index_parquet_path"])
    assert ip.resolve().is_relative_to(tmp_path.resolve())
    frame=pd.read_parquet(ip);frame["child_m1_start_row"]=0;frame.to_parquet(ip,index=False)
    root["splits"]["train"]["index_parquet_sha256"]=runner.val.file_sha256(ip)
    clock=pd.date_range("2026-02-28T23:57Z",periods=5,freq="min")
    cohort={"schema_version":cohorts.TRAIN_ROLLOUT_SCHEMA,"source_split":"train",
        "entry_row_indices":list(range(256)),"parent_entry_row_indices":list(range(256)),
        "source_index":_bind(ip),"observation_cutoff_time_ns":int(pd.Timestamp("2026-03-01T00:00Z").value)}
    monkeypatch.setattr(cohorts,"require_bounded_val_cohort",lambda value:value)
    def forbidden_control(*a,**kw):pytest.fail("frozen TRAIN built a CONTROL cohort")
    monkeypatch.setattr(cohorts,"build_chronological_control_cohort",forbidden_control)
    monkeypatch.setattr(runner.val.RandomAccessValStateFactoryV1,"from_artifacts",lambda **kw:SimpleNamespace(times=clock))
    monkeypatch.setattr(runner.trainer,"_native_candidate_val_context_binding",lambda context:seen.setdefault("train_context",context))
    limits={**args["val_limits"],"max_state_views":769 if bad_budget else 768,"max_model_forwards":3}
    args.update(val_limits=limits,frozen_train_policy_scope={"cohort":cohort,
        "origin_recipe":{"chronological_prefix":args["chronological_prefix"]},"plan":{"val_limits":limits}})
    if bad_budget:
        with pytest.raises(RuntimeError,match="ACTUAL_FOOTPRINT_MISMATCH"):
            runner._build_bound_full_train_components(**args)
        return
    result=runner._build_bound_full_train_components(**args)
    assert result["train_probe_ds"].role==result["val_ds"].role=="TRAIN"
    assert not hasattr(result["train_probe_ds"],"_unified_exit_lifecycle_v2")
    assert seen["train_context"]["frame"].entry_row_index.tolist()==cohort["entry_row_indices"]
    assert seen["train_context"]["max_state_views"]==768


@pytest.mark.parametrize("fault",[None,"value","nan"])
def test_saved_entry_predictions_checked_before_exit_rollout(frozen_train,tmp_path,monkeypatch,fault):
    _,recipe,_,_,model,cohort=frozen_train
    q=np.tile(np.array([-1.,-2.,0.],dtype=np.float32),(256,1));expected=q.copy()
    if fault=="value":expected[0,0]+=0.1
    if fault=="nan":expected[0,2]=float("nan")
    ref=_write(tmp_path/"reference-only.json",{"diagnostics":{"bounded_entry_observations":[
        {"parent_entry_row_index":parent,"predicted_q_bps":value.tolist()}
        for parent,value in zip(cohort["parent_entry_row_indices"],expected)]}})
    seen=[]
    def compose(requests):
        assert len(requests)==512 and {r["side_index"] for r in requests}=={0,1}
        assert all(r["state_index"]==0 and r["action"]=="exit_now" for r in requests)
        return [({"undiscounted_net_cash_pnl_increment_bps":-6.0},"d"*64) for _ in requests]
    def bind(**kw):
        seen.append("bound");return {"entry_pair_cohort_size":256},SimpleNamespace(compose_selected_actions=compose)
    factory=SimpleNamespace(source_split="train",artifact_file_sha256={"random_access_index":cohort["source_index"]["sha256"]},
        bind_rollout=bind,close_val_cpu_workers=lambda:None)
    monkeypatch.setattr(runner.val,"_entry_representations",lambda **kw:(torch.zeros(256,2),{},torch.from_numpy(q)))
    def evaluate(**kw):
        assert kw["entry_policy_decisions"]["model_variant"]=="frozen_online_train_policy"
        assert kw["entry_route_diagnostics"]["frozen_entry_reference_check"]["same_actions"] is True
        baseline=kw["entry_route_diagnostics"]["frozen_immediate_exit_baseline"]
        assert len(baseline["rows"])==512 and sum(row["net_cash_bps"] for row in baseline["rows"])==-3072.0
        seen.append("rollout");return {"ok":True}
    monkeypatch.setattr(runner.val,"run_resumable_random_access_val_evaluation_v1",evaluate)
    audit=tmp_path/"sequence-audit.json";audit.write_text("{}")
    kwargs=dict(model=model,entry_dataset=object(),
        frame=pd.DataFrame({"entry_row_index":cohort["entry_row_indices"],"parent_entry_row_index":cohort["parent_entry_row_indices"]}),
        state_factory=factory,checkpoint_binding={"model_variant":"frozen_online_train_policy","reference_train_observation":ref,
            "binding_sha256":"a"*64,"model_state_sha256":"b"*64,"checkpoint_file_sha256":"c"*64},
        parent_coordinate_evidence={},val_sequence_audit=audit,device=torch.device("cpu"),selected_batch_size=16,
        rollout_progress_path=tmp_path/"progress.json",result_path=tmp_path/"result.json",
        max_forwards_this_invocation=4,progress_interval_forwards=64,compute_guard_max_model_forwards=4,
        compute_guard_max_materialized_state_views=1024,compute_guard_max_wall_seconds=10800,evaluation_cohort=cohort)
    if fault:
        with pytest.raises(RuntimeError,match="SAVED_ENTRY_PREDICTION_MISMATCH"):
            runner.val.evaluate_bound_full_val_v1(**kwargs)
        assert seen==[]
    else:
        assert runner.val.evaluate_bound_full_val_v1(**kwargs)=={"ok":True}
        assert seen==["bound","rollout"]


@pytest.fixture
def entry_probe(frozen_train,tmp_path):
    policy,recipe,seal,state,model,cohort=frozen_train
    source_plan=recipe.pop("frozen_train_policy_evaluation")
    result=_write(tmp_path/"full-policy-result.json",{
        "decision":"PASS_COMPLETE","evaluation_cohort":cohort,"exited_side_trade_count":512,
        "trade_outcomes":[{"status":"EXITED"} for _ in range(512)]})
    completed=_write(tmp_path/"full-policy-completed.json",{
        "result":result,"evaluation_plan":source_plan,"learning_review_complete":True,
        "optimizer_steps":0,"exited_side_trade_count":512,
        "model_state_sha256":checkpoints.canonical_model_state_sha256(model.state_dict())})
    dataset=_write(tmp_path/"return-dataset.json",{
        "source_result":result,"source_policy_model_state_sha256":json.loads(Path(completed["path"]).read_text())["model_state_sha256"],
        "rows":[{"child_entry_row_index":child,"parent_entry_row_index":parent}
                for child,parent in zip(cohort["entry_row_indices"],cohort["parent_entry_row_indices"])]})
    reviewed=_write(tmp_path/"dataset-review.json",{"dataset":dataset,"entries_preserved":256,"negative_labels_preserved":409})
    review=_write(tmp_path/"selector-review.json",{"dataset_review_binding":reviewed})
    plan={"schema_version":"gx1_frozen_entry_representation_probe_v1",
        "stage":"cache_original_entry_representations_only","optimizer_steps":0,"new_fits":0,
        "max_entry_forwards":16,"entry_batch_size":16,"entries":256,"control_forwards":0,
        "exit_rollout_forwards":0,"training_enabled":False,"test_data_used":False,
        "source_policy_plan":source_plan,"completed_policy":completed,"source_result":result,
        "dataset_review":reviewed,"dataset":dataset,"review":review}
    pb=_write(tmp_path/"entry-probe-plan.json",plan)
    recipe["frozen_entry_selector_probe"]=pb
    admission=policy.pop("frozen_train_policy_evaluation")
    admission.update(plan=pb,new_fits=0,max_entry_forwards=16,exit_rollout_forwards=0)
    policy["frozen_entry_selector_probe"]=admission;policy["policy_consistent_entry_review"]=review
    seal()
    return policy,recipe,seal,state,model,cohort,plan


def test_entry_probe_scope_never_admits_fits_or_rollout(entry_probe,tmp_path):
    policy,recipe,seal,_,_,_,plan=entry_probe
    assert native.require_native_run_scope(recipe,invocation_number=1)==512
    for key,value in [("new_fits",1),("exit_rollout_forwards",1),("max_entry_forwards",17),("training_enabled",True)]:
        bad={**plan,key:value};binding=_write(tmp_path/"entry-probe-plan.json",bad)
        recipe["frozen_entry_selector_probe"]=binding;policy["frozen_entry_selector_probe"]["plan"]=binding;seal()
        with pytest.raises(RuntimeError,match="FROZEN_ENTRY_PROBE_PLAN_INVALID"):
            native.require_native_run_scope(recipe,invocation_number=1)
    binding=_write(tmp_path/"entry-probe-plan.json",plan)
    recipe["frozen_entry_selector_probe"]=binding;policy["frozen_entry_selector_probe"]["plan"]=binding
    recipe["frozen_train_policy_evaluation"]=plan["source_policy_plan"];seal()
    with pytest.raises(RuntimeError,match="FROZEN_ENTRY_PROBE_MIXED_SCOPE"):
        native.require_native_run_scope(recipe)


@pytest.mark.parametrize("fault",[None,"q","token","nonfinite","forward_error"])
def test_entry_capture_preserves_model_and_rejects_changed_context(monkeypatch,fault):
    model=torch.nn.Module();model.head_entry_action_q=torch.nn.Linear(128,3)
    model.eval().requires_grad_(False)
    x=torch.arange(256*128,dtype=torch.float32).reshape(256,128)/10000
    q=torch.cat([model.head_entry_action_q(batch) for batch in x.split(16)]).detach()
    tokens=x[:,:8].clone();before={k:v.clone() for k,v in model.state_dict().items()}
    cohort={"entry_row_indices":list(range(256)),"parent_entry_row_indices":list(range(1000,1256))}
    frame=pd.DataFrame({"entry_row_index":cohort["entry_row_indices"],"parent_entry_row_index":cohort["parent_entry_row_indices"]})
    scope={"cohort":cohort,"plan":{"val_limits":{"max_model_forwards":33,"max_state_views":500,"max_wall_seconds":10800}},
        "entry_selector_probe":{"source_result":{"entry_policy_decisions":{"entry_action_q_bps":q.tolist()},"contract_sha256":"a"*64}}}
    def forward(**kw):
        assert kw["model"] is model and kw["batch_size"]==16 and kw["evaluation_cohort"]==cohort
        outputs=[]
        for batch in x.split(16):
            if fault=="forward_error":raise RuntimeError("synthetic forward failed")
            value=batch.clone()
            if fault=="nonfinite":value[0,0]=float("nan")
            outputs.append(model.head_entry_action_q(value))
        actual=torch.cat(outputs)
        if fault=="q":actual[0,0]+=1
        return tokens,{},actual
    def bind(**kw):
        assert torch.equal(kw["entry_decision_representations"],tokens)
        assert kw["compute_guard_max_model_forwards"]==33 and kw["resumable_wall_limit"] is True
        return {"contract_sha256":("b" if fault=="token" else "a")*64},None
    monkeypatch.setattr(runner.val,"_entry_representations",forward)
    kwargs=dict(model=model,dataset=object(),frame=frame,state_factory=SimpleNamespace(bind_rollout=bind),
        scope=scope,binding={"model_state_sha256":"c"*64,"checkpoint_file_sha256":"d"*64},device=torch.device("cpu"))
    if fault:
        with pytest.raises(RuntimeError):runner._capture_frozen_entry_representations(**kwargs)
    else:
        cache,_=runner._capture_frozen_entry_representations(**kwargs)
        assert torch.equal(cache["entry_q_joint_hidden"],x)
        assert torch.equal(cache["original_entry_q_bps"],q) and torch.equal(cache["original_exit_tokens"],tokens)
        assert cache["entry_forward_count"]==16
    assert not model.head_entry_action_q._forward_pre_hooks
    assert all(torch.equal(value,before[key]) for key,value in model.state_dict().items())


def test_entry_probe_dispatch_excludes_training_and_exit_rollout(entry_probe,tmp_path,monkeypatch):
    _,recipe,_,state,_,_,_=entry_probe
    budget={"stop_after_optimizer_steps":512,"max_invocation_seconds":12000,
        "expected_active_pointer_sha256":None,"stop_after_completed_val_epochs":None}
    monkeypatch.setattr(runner.trainer,"_require_cuda_trainer_guard_execution",lambda **kw:None)
    monkeypatch.setattr(runner.trainer,"_resolve_device",lambda _:torch.device("cuda"))
    monkeypatch.setattr(runner,"_require_native_full_train_recipe",lambda *a:(recipe,{},{}))
    monkeypatch.setattr(runner.launch_owner,"require_candidate_execution_budget",lambda *a,**kw:budget)
    def build(**kw):
        assert "entry_selector_probe" in kw["frozen_train_policy_scope"]
        return {"synthetic":True}
    monkeypatch.setattr(runner,"_build_bound_full_train_components",build)
    def forbidden(**kw):pytest.fail("Entry extraction reached training or Exit rollout")
    monkeypatch.setattr(runner,"_run_bound_full_train_candidate",forbidden)
    monkeypatch.setattr(runner,"_run_frozen_train_policy_validation",forbidden)
    calls=[]
    def extract(**kw):
        assert kw["components"]=={"synthetic":True}
        calls.append("entry_only");return {"resume_state":state,"new_fits":0}
    monkeypatch.setattr(runner,"_run_frozen_entry_representation_probe",extract)
    result=runner.run_guarded_native_candidate_invocation(recipe_path=tmp_path/"recipe.json",recipe_file_sha256="a"*64,
        execution_budget_path=tmp_path/"budget.json",execution_budget_file_sha256="b"*64)
    assert calls==["entry_only"] and result["resume_state"]==state and result["new_fits"]==0
