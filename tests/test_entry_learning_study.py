"""Synthetic contracts only; learning and economics require the declared native run."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.contracts import entry_observed_market_v1 as observed
from gx1.contracts import unified_exit_native_candidate_campaign_v1 as native
from gx1.scripts import run_unified_exit_random_access_full_train_v1 as runner
from tests.test_native_learning_calibration_scope import scope
from tests.test_entry_observed_market import economics
from tests.test_entry_observed_market_native import dataset_scope
from tests.test_native_prefix_recipe import _bind


@pytest.fixture
def study_scope(scope,tmp_path,monkeypatch):
    policy,recipe,write,save=scope
    plan=json.loads((Path(__file__).resolve().parents[1]/"configs/research/ENTRY_LEARNING_STUDY_20261010.json").read_text())
    policy.pop("native_learning_calibration")
    policy.update(full_epoch_training_allowed=False,full_val_allowed=False)
    recipe.pop("candidate_resume_origin")
    recipe.update(chronological_prefix={"fixture":True},entry_observed_market={"fixture":True},
        initialization="fresh_existing_model_constructor_no_checkpoint_weights",
        run_id=plan["phases"]["fit"]["run_id"],out_bundle_dir=str(tmp_path/"FIT"/"CANDIDATE_BUNDLE"))
    plan.update(root=str(tmp_path),files=recipe["files"],chronological_prefix=recipe["chronological_prefix"],
                entry_observed_market=recipe["entry_observed_market"])
    order=np.arange(262145,dtype=np.int64)[::-1].copy()
    arrays={"epoch_order":order,"fit":order[:256],"train_probe":np.sort(order[:4096]),
            "control":np.linspace(0,8191,4096,dtype=np.int64)}
    for name,a in arrays.items():
        p=tmp_path/(name+".npy");np.save(p,a);plan["row_bindings"][name]=_bind(p)
    monkeypatch.setattr(native,"require_chronological_prefix_recipe",lambda _: {
        "physical_coordinates":{"artifact":{"bindings":{"TRAIN_NATIVE_EPOCH0_ORDER":plan["row_bindings"]["epoch_order"]}}}})
    monkeypatch.setattr(observed,"require_entry_observed_market_scope",lambda _: {
        "physical_sources":{"train":{"physical_rows":262145},"val":{"physical_rows":8192}}})
    def seal():
        recipe["entry_learning_study"]={"plan":write("plan.json",plan),"phase":"fit"}
        policy["entry_learning_study_run"]={
            "selection":recipe["entry_learning_study"],"run_id":recipe["run_id"],
            "out_bundle_dir":recipe["out_bundle_dir"],"source_bindings_sha256":recipe["source_bindings_sha256"],
            "optimizer_steps":1024,"max_invocations":1,"test_data_used":False}
        save()
    seal()
    return policy,recipe,plan,seal,save


def test_study_admission_exact_budget_and_unchanged_guards(study_scope):
    _,recipe,_,_,_=study_scope
    assert native.require_native_run_scope(recipe,invocation_number=1)==1024
    budget={"stop_after_optimizer_steps":1024,"stop_after_completed_val_epochs":None,"max_invocation_seconds":12000}
    assert native.require_native_run_scope(recipe,execution_budget=budget)==1024
    for field,bad in [("stop_after_optimizer_steps",1025),("stop_after_optimizer_steps",1024.),
                      ("max_invocation_seconds",12001),("stop_after_completed_val_epochs",1),
                      ("resume_probe_val_rows",32)]:
        with pytest.raises(RuntimeError,match="BUDGET"):
            native.require_native_run_scope(recipe,execution_budget={**budget,field:bad})
    for number in (0,2,True):
        with pytest.raises(RuntimeError,match="INVOCATION"):
            native.require_native_run_scope(recipe,invocation_number=number)


@pytest.mark.parametrize("fault",["missing_authority","training","full_epoch","full_val","old_origin","old_baseline","test",
                                  "more_steps","reselected_control","architecture"])
def test_study_fail_closed(study_scope,fault):
    policy,recipe,plan,seal,save=study_scope
    if fault=="old_origin":recipe["candidate_resume_origin"]={}
    elif fault=="old_baseline":recipe["chronological_learning_measurement"]={}
    elif fault=="test":plan["execution"]["test_data_used"]=True
    elif fault=="more_steps":plan["phases"]["fit"]["optimizer_steps"]=1025
    elif fault=="architecture":plan["model"]["signal_fields"]=38
    elif fault=="reselected_control":
        p=Path(plan["row_bindings"]["control"]["path"])
        a=np.load(p);a[0]=1;np.save(p,a);plan["row_bindings"]["control"]=_bind(p)
    seal()
    if fault=="missing_authority":policy.pop("entry_learning_study_run")
    elif fault=="training":policy["training_enabled"]=True
    elif fault=="full_epoch":policy["full_epoch_training_allowed"]=True
    elif fault=="full_val":policy["full_val_allowed"]=True
    save()
    with pytest.raises(RuntimeError):
        native.require_native_run_scope(recipe,invocation_number=1)


def test_fit_loader_repeats_only_fixed_cohort_preserving_durable_epoch(study_scope):
    _,recipe,_,_,_=study_scope
    study=native.require_entry_learning_study_run(recipe)
    original=torch.from_numpy(study["rows"]["epoch_order"].copy())
    before=original.clone()
    mapped=observed.entry_learning_study_loader_order(original,study=study)
    assert torch.equal(original,before)
    np.testing.assert_array_equal(mapped[:16384].numpy(),np.resize(study["rows"]["fit"],16384))
    assert torch.equal(mapped[16384:],original[16384:])
    study["phase"]="curve"
    assert observed.entry_learning_study_loader_order(original,study=study) is original
    with pytest.raises(RuntimeError,match="ORDER_CHANGED"):
        observed.entry_learning_study_loader_order(original.flip(0),study=study)


def test_control_view_keeps_original_mask_and_targets(tmp_path,economics):
    data,base=dataset_scope(tmp_path,economics)
    n=8192
    data.df=pd.DataFrame({"time":pd.date_range("2025-06-02",periods=n,freq="5min",tz="UTC"),
        observed.GROSS_TARGET_COLUMNS[0]:10., observed.GROSS_TARGET_COLUMNS[1]:-12.})
    data.df.to_parquet(data.parquet_path,index=False)
    source={**base["physical_sources"]["train"],"parquet":_bind(data.parquet_path),"physical_rows":n,
        "clock_sha256":__import__("hashlib").sha256(data.df.time.array.as_unit("ns").asi8.astype("<i8").tobytes()).hexdigest()}
    base["physical_sources"]["val"]=source
    old=data._policy_dependent_auxiliary_binding
    data._policy_dependent_auxiliary_binding={**old,"role":"CONTROL256","parent_entry_parquet":source["parquet"]}
    data._policy_dependent_auxiliary_bound_rows=np.zeros(n,dtype=bool)
    data._policy_dependent_auxiliary_bound_rows[:256]=True
    observed.bind_entry_observed_market_dataset(data,base)
    mask=data._policy_dependent_auxiliary_bound_rows.copy()
    targets=data._entry_observed_market_targets
    selected=np.linspace(0,n-1,4096,dtype=np.int64)
    study={"phase":"curve","observed_scope":base,"rows":{"control":selected},
           "plan":{"row_bindings":{"control":{"fixture":True}}},"selection":{"fixture":True}}
    view=observed.entry_learning_study_control_dataset(data,study=study)
    np.testing.assert_array_equal(data._policy_dependent_auxiliary_bound_rows,mask)
    assert data._entry_observed_market_targets is targets
    assert view._entry_observed_market_binding["admitted_rows"]==4096
    assert np.isfinite(view._entry_observed_market_targets[selected]).all()
    assert not view._policy_dependent_auxiliary_bound_rows.flags.writeable
    assert np.isnan(view._entry_observed_market_targets[~view._policy_dependent_auxiliary_bound_rows]).all()


def test_observer_is_deterministic_read_only_and_resume_collision_rejects(tmp_path,monkeypatch):
    n=32;rows=np.array([1,7,13,25],dtype=np.int64)
    class Dataset(torch.utils.data.Dataset):
        indices=np.arange(n)
        df=pd.DataFrame({"time":pd.date_range("2025-01-01",periods=n,freq="5min",tz="UTC")})
        _policy_dependent_auxiliary_bound_rows=np.ones(n,dtype=bool)
        def __len__(self):return n
        def __getitem__(self,i):
            return {"seq_x":torch.tensor([float(i),1.]),"snap_x":torch.zeros(2),
                    "ctx_cat":torch.zeros(1,dtype=torch.int64),"ctx_cont":torch.zeros(1),
                    "entry_row_index":i,"entry_observed_market_target_bps":torch.tensor([float(i),-float(i),0.])}
    model=torch.nn.Linear(2,3)
    monkeypatch.setattr(runner.trainer,"_multi_tf_kwargs_from_batch",lambda *a:{})
    monkeypatch.setattr(runner.trainer,"_model_forward_fp32",lambda m,x,*a,**kw:{"entry_action_q_bps":m(x)})
    pointer=tmp_path/"POINTER.json";pointer.write_text('{"global_optimizer_steps":0}')
    study={"phase":"fit","rows":{"fit":rows,"control":np.array([],dtype=np.int64)},
        "phase_spec":{"train_observation_steps":[0,64]},"selection":{"fixture":True},
        "observed_scope":{"physical_sources":{"train":{"fixture":True}}}}
    output=tmp_path/"CANDIDATE_BUNDLE"
    callback=runner._entry_learning_study_observer(components={"model":model,"train_probe_ds":Dataset()},
        study=study,recipe={"source_bindings_sha256":"a"*64},device=torch.device("cpu"),output=output)
    rng=torch.get_rng_state().clone();weights=copy.deepcopy(model.state_dict())
    callback(optimizer_steps=0,session=SimpleNamespace(_active_path=pointer))
    assert model.training and torch.equal(rng,torch.get_rng_state())
    for k,v in model.state_dict().items():assert torch.equal(v,weights[k])
    path=tmp_path/"OBSERVATIONS/step_000000_train.json";report=json.loads(path.read_text())
    assert report["rows"]==rows.tolist() and report["training_pointer_snapshot"]["global_optimizer_steps"]==0
    assert report["time_ns"][0]==Dataset.df.time.iloc[1].value
    before=path.read_bytes()
    callback(optimizer_steps=0,session=SimpleNamespace(_active_path=pointer))
    callback(optimizer_steps=1,session=SimpleNamespace(_active_path=pointer))
    assert path.read_bytes()==before and len(list(path.parent.glob("*.json")))==1
    with torch.no_grad():model.weight.add_(1.)
    with pytest.raises(RuntimeError,match="COLLISION"):
        callback(optimizer_steps=0,session=SimpleNamespace(_active_path=pointer))

from tests.test_native_prefix_coordinator import (
    physical_prepared, physical_component_templates, physical_component_case,
    PrefixHarness,digest,equal_tree,
)

def test_study_native_coordinator_preserves_exact_resume_and_fixed_row_mapping(physical_prepared,tmp_path):
    data=physical_prepared
    for dataset,split,role in ((data.train,"train","TRAIN"),(data.control,"val","CONTROL256")):
        dataset._entry_observed_market_binding={
            "identity":{"training_phase":"entry_only","target_sha256":"a"*64},
            "role":role,"physical_source":data.case["sources"][split]}
    order=data.case["order"].copy()
    study={"phase":"fit","phase_spec":{"optimizer_steps":4},
           "rows":{"epoch_order":order,"fit":order[:256].copy()},
           "selection":{"plan":{"path":"/synthetic/study.json","sha256":"b"*64},"phase":"fit"}}
    calls=[]
    def observer(*,optimizer_steps,session):
        pointer=json.loads(session._active_path.read_text())
        assert pointer["global_optimizer_steps"]==optimizer_steps
        calls.append(optimizer_steps)
    harness=PrefixHarness(tmp_path/"study-resume",data)
    direct,split=harness.root/"DIRECT",harness.root/"SPLIT"
    kwargs={"entry_learning_study":study,"entry_learning_observer":observer}
    harness.run_prefix(direct,4,**kwargs);expected=harness.state(direct)
    harness.run_prefix(split,2,**kwargs)
    harness.run_prefix(split,4,expected_pointer=digest(harness.pointer(split)),**kwargs)
    actual=harness.state(split)
    for key in ("model_state","target_model_state","optimizer_state","weight_ema_state",
                "lr_scheduler_state","rng_state","epoch_order","training_progress"):
        equal_tree(expected[key],actual[key])
    np.testing.assert_array_equal(actual["epoch_order"].numpy(),order)
    assert {0,2,4} <= set(calls)
    before=harness.pointer(split).read_bytes()
    changed=copy.deepcopy(study);changed["selection"]["plan"]["sha256"]="c"*64
    with pytest.raises(RuntimeError,match="CONTRACT"):
        harness.run_prefix(split,4,expected_pointer=digest(harness.pointer(split)),
            entry_learning_study=changed,entry_learning_observer=observer)
    assert harness.pointer(split).read_bytes()==before

def test_learning_review_rejects_bias_only_or_constant_actions_and_keeps_months():
    from gx1.scripts.research_ta_campaign_v1 import entry_learning_paired_review
    plan=json.loads((Path(__file__).resolve().parents[1]/"configs/research/ENTRY_LEARNING_STUDY_20261010.json").read_text())
    n=104
    clock=pd.date_range("2025-01-01",periods=n,freq="3D",tz="UTC").as_unit("ns").asi8
    value=np.arange(n,dtype=float)%7-3
    target=np.column_stack([value-5,-value-5,np.zeros(n)])
    initial=np.column_stack([np.ones(n)*-10,np.ones(n)*-10,np.zeros(n)])
    final=np.column_stack([np.ones(n)*-5,np.ones(n)*-5,np.zeros(n)])
    result=entry_learning_paired_review(initial=initial,final=final,target=target,
        constant=[-5,-5,0],time_ns=clock,review=plan["review"])
    assert not result["statistical_screen_pass"] and not result["action_gate"] and not result["centered_gate"]
    assert result["metrics"]["final"]["action_counts"]==[0,0,n]
    assert len(result["monthly"])==11 and len(result["paired_intervals"])==6
    q=target.copy()
    result=entry_learning_paired_review(initial=initial,final=q,target=target,
        constant=[-5,-5,0],time_ns=clock,review=plan["review"])
    assert result["numerical_gate"] and result["centered_gate"] and not result["action_gate"]
    assert not result["profitability_proven"] and result["period_and_operational_review_still_required"]


def test_learning_review_rejects_row_reordering_and_wrong_flat_target():
    from gx1.scripts.research_ta_campaign_v1 import entry_learning_paired_review,entry_learning_prediction_metrics
    y=np.zeros((4,3));bad=y.copy();bad[0,2]=1
    with pytest.raises(RuntimeError,match="ARRAY_INVALID"):entry_learning_prediction_metrics(y,bad)
    with pytest.raises(RuntimeError,match="CLOCK"):
        entry_learning_paired_review(initial=y,final=y,target=y,constant=[0,0,0],
            time_ns=[4,3,2,1],review={})
