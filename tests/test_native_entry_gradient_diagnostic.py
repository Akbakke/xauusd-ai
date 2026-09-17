"""Narrow synthetic checks for a read-only native TRAIN gradient contrast."""
import copy,json,time
from pathlib import Path
import pytest
import torch
from tests.test_native_prefix_initial_measurement import initial_scope
from tests.test_native_prefix_recipe import prefix_scope,scope,native,runner,_bind,_write

@pytest.fixture
def gradient_scope(initial_scope,tmp_path,monkeypatch):
    policy,recipe,files,seal=initial_scope
    old=recipe.pop("chronological_initial_measurement");policy.pop("chronological_initial_measurement")
    trainer=tmp_path/"gx1/models/entry_v10/entry_v10_ctx_train_v3.py";trainer.write_text("synthetic unchanged trainer")
    sources={"python:"+str(p.relative_to(tmp_path)):{**_bind(p),"size_bytes":p.stat().st_size,"mtime_ns":p.stat().st_mtime_ns}
             for p in [trainer,trainer.with_name("entry_v10_ctx_hybrid_transformer.py")]}
    pointer=_write(tmp_path/"pointer.json",{"steps":256});state=tmp_path/"state.pt";state.write_bytes(b"synthetic")
    resume={"training_pointer":pointer,"training_state":_bind(state),"global_optimizer_steps":256,
        "next_batch_offset":256,"epoch_index":0,"phase":"train","complete":False,"active_val_cursor":None}
    origin={"recipe":_write(tmp_path/"origin_recipe.json",{"source_bindings":sources}),
            "outcome":"RESUMABLE","resume_state":resume}
    monkeypatch.setattr(native,"require_native_cursor",lambda x,**kw:x)
    obs={"role":"train","optimizer_steps":256,"test_data_used":False,"model_state_sha256":"b"*64,
         "cohort":{"measurement_role":"train","plan":recipe["chronological_prefix"]["design"]},
         "diagnostics":{"bounded_entry_observations":[{}]*256}}
    ob=_write(tmp_path/"train.json",obs)
    final={"schema_version":"gx1_native_prefix_final_online_measurement_v1","selected_model_variant":"ONLINE",
      "optimizer_steps":256,"frozen_targets_exactly_preserved":True,"teacher_refreshed":False,"test_data_used":False,
      "initialization_result":old["initialization_result"],"observations":{"train":ob},"model_state_sha256":"b"*64}
    review={"schema_version":"gx1_native_fixed256_paired_learning_review_v1","decision":"REJECT_EXPANSION_LEARNING_GATE_FAILED",
            "final_result":_write(tmp_path/"final.json",final),**{k:resume[k] for k in ["training_pointer","training_state"]}}
    plan={"schema_version":"gx1_entry_gradient_diagnostic_plan_v1","optimizer_steps":0,"train_entries":16,
      "model_forwards":2,"control_forwards":0,"max_invocations":1,"mode":"eval","selection":"first16_existing_frozen_TRAIN_probe",
      "variants":["detached","connected"],"test_data_used":False,"review":_write(tmp_path/"review.json",review),
      "origin_cursor":_write(tmp_path/"cursor.json",origin),"train_observation":ob}
    def reseal():
        recipe["entry_gradient_diagnostic"]=_write(tmp_path/"gradient_plan.json",plan)
        policy["entry_gradient_diagnostic"]={"plan":recipe["entry_gradient_diagnostic"],
          "chronological_prefix":recipe["chronological_prefix"],"run_id":recipe["run_id"],
          "out_bundle_dir":recipe["out_bundle_dir"],"source_bindings_sha256":recipe["source_bindings_sha256"],
          "optimizer_steps":0,"origin_optimizer_steps":256,"max_invocations":1,"train_entries":16,
          "model_forwards":2,"control_forwards":0,"test_data_used":False}
        return seal()
    reseal()
    return policy,recipe,files,reseal,plan,obs,final,resume

def test_gradient_scope_admits_only_one_readonly_train16(gradient_scope):
    policy,recipe,_,seal,plan,obs,final,resume=gradient_scope
    budget={"stop_after_optimizer_steps":256,"stop_after_completed_val_epochs":None,
            "expected_active_pointer_sha256":None,"max_invocation_seconds":12000}
    assert native.require_native_run_scope(recipe,invocation_number=1,execution_budget=budget)==256
    for number in (0,2,True):
        with pytest.raises(RuntimeError,match="GRADIENT_INVOCATION"):native.require_native_run_scope(recipe,invocation_number=number)
    for key,value in [("stop_after_optimizer_steps",257),("stop_after_optimizer_steps",256.0),
                      ("expected_active_pointer_sha256","a"*64),("stop_after_completed_val_epochs",1),
                      ("max_invocation_seconds",12001)]:
        with pytest.raises(RuntimeError,match="GRADIENT_BUDGET"):native.require_native_run_scope(recipe,execution_budget={**budget,key:value})

@pytest.mark.parametrize("fault",["extra_forward","control","updates","model_changed","wrong_role","training"])
def test_gradient_scope_rejects_changed_inputs_or_authority(gradient_scope,fault):
    policy,recipe,_,seal,plan,obs,final,resume=gradient_scope
    if fault=="extra_forward":plan["model_forwards"]=3
    elif fault=="control":plan["control_forwards"]=1
    elif fault=="updates":plan["optimizer_steps"]=1
    elif fault=="model_changed":
        p=Path(recipe["source_repo"])/"gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py";p.write_text("changed")
    elif fault=="wrong_role":
        obs["role"]="control"
        plan["train_observation"]=_write(Path(plan["train_observation"]["path"]),obs)
    else:policy["training_enabled"]=True
    seal()
    with pytest.raises(RuntimeError):native.require_native_run_scope(recipe)

def test_gradient_dispatch_never_enters_training(gradient_scope,monkeypatch,tmp_path):
    _,recipe,files,seal,*_=gradient_scope;rb=seal();seen=[]
    budget={"stop_after_optimizer_steps":256,"stop_after_completed_val_epochs":None,
            "expected_active_pointer_sha256":None,"max_invocation_seconds":12000}
    monkeypatch.setattr(runner.trainer,"_require_cuda_trainer_guard_execution",lambda **kw:seen.append("guard"))
    monkeypatch.setattr(runner.trainer,"_resolve_device",lambda _:torch.device("cpu"))
    monkeypatch.setattr(runner,"_require_native_full_train_recipe",lambda *a:(recipe,files,{}))
    monkeypatch.setattr(runner.launch_owner,"require_candidate_execution_budget",lambda *a,**kw:budget)
    monkeypatch.setattr(runner,"_build_bound_full_train_components",lambda **kw:{})
    def diagnose(**kw):seen.append("diagnostic");return {"decision":"PAUSED_RESUMABLE"}
    monkeypatch.setattr(runner,"_run_entry_gradient_diagnostic",diagnose)
    def forbidden(**kw):raise AssertionError("training entered")
    monkeypatch.setattr(runner,"_run_bound_full_train_candidate",forbidden)
    result=runner.run_guarded_native_candidate_invocation(recipe_path=Path(rb["path"]),recipe_file_sha256=rb["sha256"],
        execution_budget_path=tmp_path/"budget",execution_budget_file_sha256="a"*64)
    assert seen==["guard","diagnostic"] and result["decision"]=="PAUSED_RESUMABLE"

class Small(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.family_tf_context_gate=torch.nn.Linear(4,4)
        self.family_tf_token_gate=torch.nn.Linear(4,4)
        self.entry_q_joint_in=torch.nn.Linear(4,4)
        self.head_entry_action_q=torch.nn.Linear(4,3)
        self.task_log_variances=torch.nn.ParameterDict({k:torch.nn.Parameter(torch.zeros(())) for k in runner.trainer.JOINT_TASK_NAMES})
    def forward(self,seq_x,snap_x,*,liquidation_relative_values=False,**kw):
        h=torch.tanh(self.family_tf_context_gate(seq_x)+self.family_tf_token_gate(snap_x))
        q=self.head_entry_action_q(self.entry_q_joint_in(h.detach() if liquidation_relative_values else h))
        return {"entry_action_q_bps":q,"aux":h}

@pytest.fixture
def pair(monkeypatch):
    torch.manual_seed(104)
    model=Small().eval()
    batch={k:torch.randn(16,4) for k in ["seq_x","snap_x","ctx_cat","ctx_cont","seq_m15","seq_h1","seq_h4","seq_d1"]}
    batch["y_position_size_target"]=torch.zeros(16);batch["y_position_size_mask"]=torch.zeros(16)
    monkeypatch.setattr(runner.trainer,"dip_forecast_task_losses",lambda out,*a:{"forecast_return_bps":(out["aux"]-1).square().mean()})
    monkeypatch.setattr(runner.trainer,"_side_mae_auxiliary_loss",lambda out,*a:((out["aux"]+1).square().mean(),{}))
    monkeypatch.setattr(runner.trainer,"_trendline_event_aux_loss",lambda out,*a:(out["aux"].square().mean(),{}))
    monkeypatch.setattr(runner.trainer,"_require_active_aux_head_prediction",lambda out,*a,**kw:out["aux"][:,0])
    target=torch.randn(16,3);valid=torch.ones_like(target,dtype=torch.bool)
    expected=model(batch["seq_x"],batch["snap_x"])["entry_action_q_bps"].detach()
    return dict(model=model,batch=batch,target=target,valid=valid,expected_prediction=expected,device=torch.device("cpu"))

def test_pair_opens_gradient_without_changing_outputs_heads_or_weights(pair):
    before=copy.deepcopy(pair["model"].state_dict())
    result=runner._entry_gradient_pair(**pair)
    assert result["detached"]["entry"]["routing_l2_norm"]==0
    assert result["connected"]["entry"]["routing_l2_norm"]>0
    assert result["detached"]["raw_entry_mse"]==result["connected"]["raw_entry_mse"]
    assert result["detached"]["entry"]["entry_head_l2_norm"]==result["connected"]["entry"]["entry_head_l2_norm"]
    for k,v in before.items():assert torch.equal(v,pair["model"].state_dict()[k])
    assert all(p.grad is None for p in pair["model"].parameters())

def test_cached_numeric_roundoff_keeps_strict_variant_gradient_parity(pair):
    pair["expected_prediction"] = pair["expected_prediction"] + 1e-5
    result=runner._entry_gradient_pair(**pair)
    assert result["connected"]["cached_prediction_max_abs_difference_bps"]<1e-4
    assert result["connected"]["cached_actions_equal"]

def test_cached_action_change_below_tolerance_is_rejected(pair):
    with torch.no_grad():
        pair["model"].head_entry_action_q.weight.zero_()
        pair["model"].head_entry_action_q.bias.zero_()
    pair["expected_prediction"]=torch.zeros(16,3)
    pair["expected_prediction"][0,1]=1e-5
    with pytest.raises(RuntimeError,match="FORWARD_VALUES_CHANGED"):
        runner._entry_gradient_pair(**pair)

@pytest.mark.parametrize("fault",["prediction","mask","training_mode"])
def test_pair_rejects_nonidentical_values_and_wrong_scope(pair,fault):
    if fault=="prediction":pair["expected_prediction"]=pair["expected_prediction"]+1
    elif fault=="mask":pair["valid"][0,0]=False
    else:pair["model"].train()
    with pytest.raises(RuntimeError,match="ENTRY_GRADIENT"):runner._entry_gradient_pair(**pair)


@pytest.mark.parametrize("failure",[False,True])
def test_outer_preserves_original_checkpoint_and_rng_on_success_or_failure(pair,tmp_path,monkeypatch,failure):
    model=pair["model"];model.train()
    state_path=tmp_path/"state.pt"
    torch.save({"model_state":model.state_dict(),"target_model_state":model.state_dict(),"global_optimizer_steps":256},state_path)
    pointer=_write(tmp_path/"pointer.json",{"step":256})
    state_binding=_bind(state_path);model_hash=runner.trainer._model_state_sha256(model)
    rows=[{"parent_entry_row_index":i,"target_q_bps":pair["target"][i].tolist(),
           "target_valid":[True]*3,"predicted_q_bps":pair["expected_prediction"][i].tolist()} for i in range(16)]
    scope={"origin_resume_state":{"training_state":state_binding,"training_pointer":pointer},
       "result":{"model_state_sha256":model_hash,"target_model_state_sha256":model_hash},
       "observation":{"diagnostics":{"bounded_entry_observations":rows},"cohort":{"parent_entry_row_indices":list(range(16))}},
       "plan_binding":{"path":"synthetic","sha256":"a"*64}}
    monkeypatch.setattr(native,"require_entry_gradient_diagnostic",lambda recipe:scope)
    batch={**pair["batch"],"entry_row_index":torch.arange(16)}
    monkeypatch.setattr(runner.val,"DataLoader",lambda *a,**kw:[batch])
    def measure(**kw):
        assert not kw["model"].training
        torch.rand(5)
        if failure:raise RuntimeError("synthetic failure")
        return {"synthetic":True}
    monkeypatch.setattr(runner,"_entry_gradient_pair",measure)
    rng=torch.get_rng_state().clone()
    args=dict(components={"model":model,"train_probe_ds":object()},recipe={"source_commit":"c"*40},
              device=torch.device("cpu"),output=tmp_path/"MODEL",recipe_file_sha256="b"*64,invocation_started=time.monotonic())
    if failure:
        with pytest.raises(RuntimeError,match="synthetic failure"):runner._run_entry_gradient_diagnostic(**args)
    else:
        result=runner._run_entry_gradient_diagnostic(**args)
        assert result["resume_state"]==scope["origin_resume_state"]
        report=json.loads(Path(result["observation"]["path"]).read_text())
        assert report["optimizer_steps"]==report["control_forwards"]==0
    assert torch.equal(torch.get_rng_state(),rng) and model.training
    assert runner.trainer._model_state_sha256(model)==model_hash
    assert _bind(state_path)==state_binding and _bind(Path(pointer["path"]))==pointer
    assert (tmp_path/"entry_gradient_diagnostic/TRAIN16_INPUTS_AND_TARGETS.pt").is_file()
