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
          "model_forwards":plan['model_forwards'],"control_forwards":0,"test_data_used":False}
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


def test_signal_loss_split_preserves_native_mse_and_prediction_gradient():
    torch.manual_seed(73)
    q=torch.randn(16,3,requires_grad=True);target=20*torch.randn(16,3)
    valid=torch.ones_like(q,dtype=torch.bool)
    parts=runner._entry_signal_losses(q,target,valid)
    actual=sum(parts.values());native_loss=torch.nn.functional.mse_loss(q,target)
    torch.testing.assert_close(actual,native_loss)
    torch.testing.assert_close(torch.autograd.grad(actual,q,retain_graph=True)[0],torch.autograd.grad(native_loss,q)[0])
    valid[0,0]=False
    with pytest.raises(RuntimeError,match="ENTRY_SIGNAL_INPUT_INVALID"):
        runner._entry_signal_losses(q,target,valid)


@pytest.mark.parametrize('validate_inference',[False,True])
def test_signal_pair_observes_two_frozen_states_without_accumulation(pair,validate_inference):
    class SignalSmall(Small):
        def __init__(self):
            super().__init__();self.entry_q_joint_norm=torch.nn.LayerNorm(16)
            self.entry_q_joint_in=torch.nn.Linear(16,4);self.calls=0;self.offset_gradient=False
        def forward(self,seq_x,snap_x,**kw):
            self.calls+=1
            h=torch.tanh(self.family_tf_context_gate(seq_x)+self.family_tf_token_gate(snap_x))
            hidden=torch.nn.functional.gelu(self.entry_q_joint_in(self.entry_q_joint_norm(torch.cat([h]*4,1))))
            q=self.head_entry_action_q(hidden)
            if self.offset_gradient and torch.is_grad_enabled():q=q+0.001
            return {"entry_action_q_bps":q,"entry_q_joint_hidden":hidden,"aux":h}
    model=SignalSmall().eval();batch=pair['batch'];initial=copy.deepcopy(model.state_dict())
    initial_q=model(batch['seq_x'],batch['snap_x'])['entry_action_q_bps'].detach()
    with torch.no_grad():model.head_entry_action_q.weight.add_(0.1)
    final=copy.deepcopy(model.state_dict());final_q=model(batch['seq_x'],batch['snap_x'])['entry_action_q_bps'].detach()
    model.calls=0;model.offset_gradient=validate_inference
    result=runner._entry_signal_pair(model=model,batch=batch,target=pair['target'],valid=pair['valid'],
        predictions={'initial':initial_q,'final':final_q},states={'initial':initial,'final':final},device=pair['device'],validate_inference=validate_inference)
    assert model.calls==(4 if validate_inference else 2) and result['final']['gradients']['contrast']['routing']['l2_norm']>0
    assert result['final']['gradients']['auxiliary']['entry_head']['l2_norm']==0
    assert all(p.grad is None for p in model.parameters())
    for k,v in final.items():assert torch.equal(v,model.state_dict()[k])
    if validate_inference:
        assert result['final']['inference_cached_max_abs_difference_bps']==0
        assert result['final']['cached_max_abs_difference_bps']>0.0009
        assert result['final']['gradient_cached_within_1e_minus4_bps'] is False
        with pytest.raises(RuntimeError,match='ENTRY_SIGNAL_CACHED_INFERENCE_CHANGED'):
            runner._entry_signal_pair(model=model,batch=batch,target=pair['target'],valid=pair['valid'],
                predictions={'initial':initial_q+1,'final':final_q},states={'initial':initial,'final':final},device=pair['device'],validate_inference=True)


@pytest.mark.parametrize('parity_only',[False,True,'checked','representations','main_encoder','joint'])
def test_signal_scope_binds_prior_cached_inputs_and_initial_predictions(gradient_scope,tmp_path,parity_only):
    _,recipe,_,seal,plan,obs,final,resume=gradient_scope
    plan.update(diagnostic_kind='initial_final_entry_signal',schema_version='gx1_entry_signal_diagnostic_plan_v1',variants=['initial','final'])
    if parity_only is True:plan.update(diagnostic_kind='initial_final_forward_parity',schema_version='gx1_entry_forward_parity_plan_v1',model_forwards=4,variants=['initial_inference','initial_gradient','final_inference','final_gradient'])
    if parity_only=='checked':plan.update(diagnostic_kind='initial_final_entry_signal_inference_checked',schema_version='gx1_entry_signal_diagnostic_plan_v2',model_forwards=4)
    if parity_only in ('representations','main_encoder','joint'):plan.update(diagnostic_kind='initial_final_entry_representations',schema_version='gx1_entry_representation_diagnostic_plan_v1')
    baseline={**obs,'optimizer_steps':0,'model_state_sha256':'c'*64}
    final['target_model_state_sha256']='c'*64
    final['initial_measurement']=_write(tmp_path/'saved_initial.json',{'observations':{'train':_write(tmp_path/'initial_obs.json',baseline)}})
    review=json.loads(Path(plan['review']['path']).read_text())
    review.update(schema_version='gx1_causal_entry_fixed256_train_review_v1',decision='PAIRED_METRICS_COMPLETE_VERDICT_REQUIRED',final_result=_write(tmp_path/'final.json',final))
    plan['review']=_write(tmp_path/'review.json',review)
    plan['verdict']=_write(tmp_path/'verdict.json',{'review':plan['review'],'decision':'REJECT_EXPANSION_CAUSAL_ENTRY_ALL_FLAT_EXIT_FIXED_BY_SIDE','learning_gate_passed':False})
    cache=tmp_path/'input_cache.pt';cache.write_bytes(b'synthetic cache')
    prior_recipe=_write(tmp_path/'cached_recipe.json',{'chronological_prefix':recipe['chronological_prefix'],'files':recipe['files']})
    prior_plan=_write(tmp_path/'cached_plan.json',{'origin_cursor':_write(tmp_path/'cached_cursor.json',{'recipe':prior_recipe})})
    plan['cached_input_review']=_write(tmp_path/'cache_review.json',{'schema_version':'gx1_entry_gradient_diagnostic_result_v1',
        'plan':prior_plan,'input_cache':_bind(cache),'selection':'first16_existing_frozen_TRAIN_probe','optimizer_steps':0,'test_data_used':False})
    if parity_only in ('representations','main_encoder','joint'):
        review['schema_version']='gx1_residual_normalization_fixed256_train_review_v1'
        plan['review']=_write(tmp_path/'review.json',review)
        plan['verdict']=_write(tmp_path/'verdict.json',{'review':plan['review'],'decision':'REJECT_EXPANSION_RESIDUAL_NORMALIZATION_NO_DECISION_IMPROVEMENT','learning_gate_passed':False})
        audit={'schema_version':'gx1_residual_representation_input_audit_v1',
            'decision':'EXACT_INPUT_TARGET_AND_ROW_BINDING_CONFIRMED_NATIVE_DIAGNOSTIC_EXTENSION_REQUIRED',
            'review':plan['review'],'verdict':plan['verdict'],'original_input_cache':_bind(cache),'cache':_bind(cache),
            'new_training_observation':plan['train_observation'],
            'initial_training_observation':json.loads(Path(final['initial_measurement']['path']).read_text())['observations']['train'],
            'training_state':resume['training_state'],'training_pointer':resume['training_pointer'],
            'model_state_sha256':final['model_state_sha256'],'target_model_state_sha256':final['target_model_state_sha256'],
            'entries':16,'batch_identical_to_original_cache':True,'parent_row_order_exact':True,
            'targets_exact':True,'masks_exact':True,'same_prefix_and_file_bindings':True,
            'all_floating_batch_tensors_finite':True,'new_model_forwards':0,'optimizer_steps':0,'test_data_used':False}
        plan['input_binding_audit']=_write(tmp_path/'input_audit.json',audit)
    if parity_only in ('main_encoder','joint'):
        plan['schema_version']='gx1_entry_representation_diagnostic_plan_v2'
        if parity_only=='joint':plan.update(diagnostic_kind='final_joint_update',schema_version='gx1_joint_update_diagnostic_plan_v2',variants=['final'],model_forwards=6)
        functions={'online':'normalized_online','target':'frozen_teacher'}
        initialization=json.loads(Path(final['initialization_result']['path']).read_text())
        initialization.update(online_model_state_sha256='c'*64,model_functions=functions)
        final['initialization_result']=_write(tmp_path/'initial.json',initialization)
        measurement=_write(tmp_path/'measurement.json',{'coordinate_result':{'path':'coordinates','sha256':'d'*64}})
        obs['cohort'].update(measurement_coordinates={'path':'coordinates','sha256':'d'*64},entry_row_indices=list(range(256)))
        plan['train_observation']=_write(tmp_path/'train.json',obs)
        baseline.update(target_model_state_sha256='c'*64)
        initial_result={'schema_version':'gx1_native_prefix_initial_measurement_v1',
            'decision':'FROZEN_INITIAL_TARGETS_AND_PREDICTIONS_READY_NO_LEARNING_MEASURED',
            'initialization_result':final['initialization_result'],'measurement_binding_result':measurement,
            'model_functions':functions,'model_state_sha256':'c'*64,'target_model_state_sha256':'c'*64,
            'optimizer_steps':0,'teacher_refreshed':False,'economic_rollout':False,'test_data_used':False,
            'observations':{'train':_write(tmp_path/'initial_obs.json',baseline)}}
        final.update(initial_measurement=_write(tmp_path/'saved_initial.json',initial_result),
            model_functions=functions,observations={'train':plan['train_observation']})
        initial_audit={'schema_version':'gx1_native_prefix_initial_measurement_audit_v1',
            'decision':'FROZEN_INITIAL_MEASUREMENT_VERIFIED_NO_LEARNING_MEASURED',
            'result':final['initial_measurement'],'optimizer_steps':0,'test_data_used':False,
            'fresh_model_optimizer_ema_scheduler_exactly_preserved':True,'saved_cpu_python_numpy_rng_exactly_preserved':True,
            'receipt':_write(tmp_path/'terminal.json',{'guard_decision':'PASS','outcome':'RESUMABLE',
                'trainer_guard_exit_code':0,'progress_observer_exit_code':0,'test_data_used':False})}
        final['initial_measurement_audit']=_write(tmp_path/'initial_audit.json',initial_audit)
        origin=json.loads(Path(plan['origin_cursor']['path']).read_text())
        origin_recipe=json.loads(Path(origin['recipe']['path']).read_text())
        origin_recipe.update(chronological_prefix=recipe['chronological_prefix'],files=recipe['files'],
            chronological_train_only_measurement=True,chronological_learning_measurement=final['initial_measurement_audit'])
        origin['recipe']=_write(tmp_path/'origin_recipe.json',origin_recipe)
        plan['origin_cursor']=_write(tmp_path/'cursor.json',origin)
        review.update(schema_version=('gx1_entry_fuse_normalization_fixed256_train_review_v1' if parity_only=='joint' else 'gx1_main_encoder_normalization_fixed256_train_review_v1'),
            initial_result=final['initial_measurement'],final_result=_write(tmp_path/'final.json',final))
        plan['review']=_write(tmp_path/'review.json',review)
        plan['verdict']=_write(tmp_path/'verdict.json',{'review':plan['review'],
            'decision':('REJECT_EXPANSION_ENTRY_FUSE_VALUE_FIT_IMPROVED_ACTIONS_UNCHANGED' if parity_only=='joint' else 'REJECT_EXPANSION_MAIN_ENCODER_PARTIAL_ENTRY_SIGNAL_NO_DECISION_IMPROVEMENT'),'learning_gate_passed':False})
        audit.update(schema_version=('gx1_joint_update_input_audit_v1' if parity_only=='joint' else 'gx1_main_encoder_representation_input_audit_v1'),review=plan['review'],
            verdict=plan['verdict'],new_training_observation=plan['train_observation'],
            initial_training_observation=initial_result['observations']['train'],
            initial_measurement=final['initial_measurement'],initial_measurement_audit=final['initial_measurement_audit'],
            model_functions=functions,initial_targets_and_masks_exact=True)
        plan['input_binding_audit']=_write(tmp_path/'input_audit.json',audit)
    if parity_only=='checked':
        parity={'schema_version':'gx1_entry_forward_parity_result_v1','reused_input_cache':_bind(cache),
                'training_state':resume['training_state'],'training_pointer':resume['training_pointer'],
                'optimizer_steps':0,'model_forwards':4,'test_data_used':False,'model_and_original_checkpoint_preserved':True,
                'measurements':{k:{'model_state_sha256':digest,'comparisons':{c:{'max_abs_difference_bps':0 if c.startswith('inference') else 0.0003,'changed_actions':0}
                       for c in ('inference__cached_reference','gradient__cached_reference','gradient__inference')}}
                       for k,digest in (('initial',final['target_model_state_sha256']),('final',final['model_state_sha256']))}}
        plan['forward_parity_result']=_write(tmp_path/'forward_parity.json',parity)
    seal()
    scope=native.require_entry_gradient_diagnostic(recipe)
    assert scope['cached_inputs']==_bind(cache) and scope['initial_observation']==baseline
    if parity_only=='checked':
        parity['measurements']['initial']['comparisons']['inference__cached_reference']['max_abs_difference_bps']=0.01
        plan['forward_parity_result']=_write(tmp_path/'forward_parity.json',parity);seal()
        with pytest.raises(RuntimeError,match='ENTRY_SIGNAL_PARITY_EVIDENCE_INVALID'):native.require_entry_gradient_diagnostic(recipe)
        parity['measurements']['initial']['comparisons']['inference__cached_reference']['max_abs_difference_bps']=0
        plan['forward_parity_result']=_write(tmp_path/'forward_parity.json',parity);seal()
    if parity_only in ('representations','main_encoder','joint'):
        for key,value in [('model_state_sha256','d'*64),('parent_row_order_exact',False),('new_model_forwards',1),('entries',16.0)]:
            plan['input_binding_audit']=_write(tmp_path/'input_audit.json',{**audit,key:value});seal()
            with pytest.raises(RuntimeError,match='ENTRY_REPRESENTATION_INPUT_BINDING_INVALID'):native.require_entry_gradient_diagnostic(recipe)
        plan['input_binding_audit']=_write(tmp_path/'input_audit.json',audit);seal()
    if parity_only in ('main_encoder','joint'):
        # Same weights and cohort must not admit a substituted older initial function.
        for key,value in [('model_functions',{'online':'old_online','target':'frozen_teacher'}),
                          ('initial_measurement',_write(tmp_path/'old_initial.json',initial_result))]:
            altered={**final,key:value}
            plan['review']=_write(tmp_path/'review.json',{**review,'final_result':_write(tmp_path/'final.json',altered)})
            plan['verdict']=_write(tmp_path/'verdict.json',{'review':plan['review'],
                'decision':('REJECT_EXPANSION_ENTRY_FUSE_VALUE_FIT_IMPROVED_ACTIONS_UNCHANGED' if parity_only=='joint' else 'REJECT_EXPANSION_MAIN_ENCODER_PARTIAL_ENTRY_SIGNAL_NO_DECISION_IMPROVEMENT'),'learning_gate_passed':False})
            seal()
            with pytest.raises(RuntimeError,match='ENTRY_REPRESENTATION_INITIAL_FUNCTION_INVALID'):
                native.require_entry_gradient_diagnostic(recipe)
        plan['review']=_write(tmp_path/'review.json',{**review,'final_result':_write(tmp_path/'final.json',final)})
        plan['verdict']=audit['verdict'];_write(Path(plan['verdict']['path']),{'review':plan['review'],
            'decision':('REJECT_EXPANSION_ENTRY_FUSE_VALUE_FIT_IMPROVED_ACTIONS_UNCHANGED' if parity_only=='joint' else 'REJECT_EXPANSION_MAIN_ENCODER_PARTIAL_ENTRY_SIGNAL_NO_DECISION_IMPROVEMENT'),'learning_gate_passed':False})
        seal()
    cache.write_bytes(b'tampered cache')
    with pytest.raises(RuntimeError):native.require_entry_gradient_diagnostic(recipe)


def test_forward_parity_reports_difference_without_hiding_it_or_updating_model(pair,monkeypatch):
    model=pair['model'];before=copy.deepcopy(model.state_dict());seen=[]
    original=runner.trainer._model_forward_fp32
    def differing_forward(*args,**kwargs):
        seen.append(torch.is_grad_enabled());out=original(*args,**kwargs)
        if torch.is_grad_enabled():out['entry_action_q_bps']=out['entry_action_q_bps']+0.01
        return out
    monkeypatch.setattr(runner.trainer,'_model_forward_fp32',differing_forward)
    result=runner._entry_forward_parity(model=model,batch=pair['batch'],expected=pair['expected_prediction'],device=pair['device'])
    assert seen==[False,True]
    assert result['comparisons']['gradient__inference']['max_abs_difference_bps']>0.009
    assert not result['comparisons']['gradient__inference']['within_existing_1e_minus4_bps']
    assert result['comparison_tolerance_changed'] is False and all(p.grad is None for p in model.parameters())
    for k,v in before.items():assert torch.equal(v,model.state_dict()[k])


def test_representation_pair_uses_only_two_inference_forwards_and_removes_hooks(pair):
    class RepresentationSmall(Small):
        def __init__(self):
            super().__init__();self.fuse=torch.nn.Linear(12,4)
            self.specialist_out=torch.nn.Linear(4,4);self.cross_tf_out=torch.nn.Linear(4,4)
            self.family_tf_cooperation_out=torch.nn.Linear(4,4)
            self.entry_q_joint_norm=torch.nn.LayerNorm(16);self.entry_q_joint_in=torch.nn.Linear(16,4)
            self.seen=[]
        def forward(self,seq_x,snap_x,**kw):
            self.seen.append((torch.is_grad_enabled(),torch.is_inference_mode_enabled()))
            z=self.fuse(torch.cat([seq_x,snap_x,seq_x],1));local=z+.25*self.specialist_out(z)
            fused=local+.5*self.cross_tf_out(z)+.25*self.family_tf_cooperation_out(z)
            hidden=torch.nn.functional.gelu(self.entry_q_joint_in(self.entry_q_joint_norm(torch.cat([local,fused,z,seq_x],1))))
            return {'entry_action_q_bps':self.head_entry_action_q(hidden),'entry_q_joint_hidden':hidden}
    model=RepresentationSmall().eval();batch=pair['batch'];states={};predictions={}
    with torch.inference_mode():
        for variant in ('initial','final'):
            if variant=='final':model.entry_q_joint_in.weight.mul_(0.1)
            states[variant]=copy.deepcopy(model.state_dict())
            predictions[variant]=model(batch['seq_x'],batch['snap_x'])['entry_action_q_bps'].clone()
    model.seen.clear()
    args=dict(model=model,batch=batch,target=pair['target'],valid=pair['valid'],predictions=predictions,states=states,device=pair['device'])
    report=runner._entry_representation_pair(**args)
    assert model.seen==[(False,True),(False,True)]
    assert report['final']['representations']['entry_hidden']['rms_feature_std']<report['initial']['representations']['entry_hidden']['rms_feature_std']
    assert report['final']['inference_cached_max_abs_difference_bps']==0
    assert all(p.grad is None for p in model.parameters())
    for k,v in states['final'].items():assert torch.equal(v,model.state_dict()[k])
    with pytest.raises(RuntimeError,match='ENTRY_REPRESENTATION_CACHED_INFERENCE_CHANGED'):
        runner._entry_representation_pair(**{**args,'predictions':{**predictions,'initial':predictions['initial']+1}})
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    # Values inside numeric tolerance may still change an action; reject that too.
    with torch.inference_mode():
        for state in states.values():
            state['head_entry_action_q.weight'].zero_();state['head_entry_action_q.bias'].zero_()
    tied=torch.zeros(16,3);tied[0,1]=1e-5
    with pytest.raises(RuntimeError,match='ENTRY_REPRESENTATION_CACHED_INFERENCE_CHANGED'):
        runner._entry_representation_pair(**{**args,'predictions':{'initial':tied,'final':tied}})
