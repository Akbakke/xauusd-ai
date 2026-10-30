"""Synthetic private-family guard tests; no model, market data or fit."""
from copy import deepcopy
import numpy as np
import pytest

from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
from gx1.scripts import run_entry_native_direction_dense_qualification_v1 as worker


def _batch():
    original={"x":np.asarray([[1,0,7],[3,1,8]],dtype=np.float32)}
    contract={"surfaces":{"x":{"preserved_mask_indices":[1],"mask_pairs":[[0,1]],
        "replacement_raw":[2,None,9]}},"arms":{"baseline":{"x":[]},"family":{"x":[0]}}}
    changed={"x":np.asarray([[0,0,7],[2,1,8]],dtype=np.float32)}
    return {"features":changed,"report_features":original},contract


def test_actual_value_reference_preserves_missingness_and_unrelated():
    raw,c=_batch()
    r=worker.require_family_input_intervention(raw,c,arm="family",np=np)
    assert r["all_masks_and_unrelated_values_bit_identical"]
    assert r["all_selected_values_equal_bound_reference"]
    assert r["conditional_value_cells"]==2


@pytest.mark.parametrize("row,col,value,reason",[
    (0,0,2,"FAMILY_VALUE_REFERENCE"),(1,0,3,"FAMILY_VALUE_REFERENCE"),
    (1,1,0,"FAMILY_UNRELATED_OR_MASK_CHANGED"),(0,2,8,"FAMILY_UNRELATED_OR_MASK_CHANGED")])
def test_actual_intervention_rejects_wrong_value_mask_and_collateral(row,col,value,reason):
    raw,c=_batch();raw["features"]["x"][row,col]=value
    with pytest.raises(RuntimeError,match=reason):
        worker.require_family_input_intervention(raw,c,arm="family",np=np)


def test_baseline_must_be_exact_noop():
    raw,c=_batch();raw["features"]={k:v.copy() for k,v in raw["report_features"].items()}
    r=worker.require_family_input_intervention(raw,c,arm="baseline",np=np)
    assert r["conditional_value_cells"]==r["actually_changed_cells"]==0
    raw["features"]["x"][0,0]=2
    with pytest.raises(RuntimeError,match="FAMILY_UNRELATED_OR_MASK_CHANGED"):
        worker.require_family_input_intervention(raw,c,arm="baseline",np=np)


def test_shared_mask_cannot_be_selected():
    raw,c=_batch();c["arms"]["family"]["x"].append(1)
    with pytest.raises(RuntimeError,match="FAMILY_MASK_SELECTED"):
        worker.require_family_input_intervention(raw,c,arm="family",np=np)


def test_family_power_cap_and_wrong_scope():
    r={"run_scope":authority.FAMILY_PRIVATE_SCOPE,
       "admission_stage":"dynamic_m5_direction_family_private_chart_20261030_001",
       "candidate_mode":"price_only","numerical_recipe":{"dense_minute_supervision":True}}
    assert authority.guarded_research_power_limit(r)==250
    r["candidate_mode"]="price_and_m1"
    with pytest.raises(Exception,match="FAMILY_POWER_SCOPE"):
        authority.guarded_research_power_limit(r)


def test_same_arm_initial_binds_inputs_state_and_rng(tmp_path):
    arrays={"parent_rows":np.arange(256,dtype=np.int64),"bar_start_ns":np.arange(256,dtype=np.int64),
        "decision_ns":np.arange(256,dtype=np.int64),"logit_up":np.zeros((256,60),dtype=np.float32)}
    path=tmp_path/"initial.npz";np.savez(path,**arrays)
    ref={"path":str(path),"sha256":"a"*64}
    report={"schema_version":"gx1_native_m5_family_initial_measurement_v1","role":"INITIAL_TRAIN",
        "family_arm":"chart_geometry_encoder","family_intervention_sha256":"b"*64,
        "predictions":ref,"preservation":{"model":"m","all_buffers":"b","rng":"r"},"test_data_used":False}
    measured={"arrays":arrays,"state_preservation":dict(report["preservation"])}
    kwargs={"family_arm":"chart_geometry_encoder","intervention_sha256":"b"*64,"np":np}
    worker.require_family_initial_parity(measured,report,ref,**kwargs)
    for key,bad,reason in [("family_arm","momentum_flow_encoder","FAMILY_INITIAL_IDENTITY"),
                          ("family_intervention_sha256","c"*64,"FAMILY_INITIAL_IDENTITY")]:
        altered=deepcopy(report);altered[key]=bad
        with pytest.raises(RuntimeError,match=reason):
            worker.require_family_initial_parity(measured,altered,ref,**kwargs)
    measured["state_preservation"]["rng"]="other"
    with pytest.raises(RuntimeError,match="FAMILY_INITIAL_RNG_PARITY"):
        worker.require_family_initial_parity(measured,report,ref,**kwargs)


def _qualification(monkeypatch):
    def ref(name):return {"path":"/bound/"+name,"sha256":name}
    source={"pure":{"path":"/repo/pure.py","sha256":"pure"}}
    fold={"prefix_rows":47602}
    common={"input_manifest":ref("input"),"fold":fold,"dense_target":ref("targets"),
        "dense_target_result":ref("target_result"),"dense_source_transition_proof":ref("transition"),
        "decimal_input_audit":ref("input_audit"),"decimal_target_refs":{"x":ref("x")},
        "candidate_mode":"price_only","parent_order_algorithm":"fixed","family_arm":"chart_geometry_encoder",
        "family_intervention_sha256":"f"*64}
    recipe={**common,"native_dense_qualification":ref("review"),"family_initial_report":ref("initial"),
        "decimal_initial_predictions":ref("npz"),"numerical_recipe":{"fixed":True,"max_updates":2976,"contract_sha256":"fit"}}
    prior={**common,"run_scope":authority.FAMILY_PRIVATE_SCOPE,"qualification_weights_for_main_fit":False,
        "numerical_recipe":{"fixed":True,"max_updates":2,"contract_sha256":"private"},"source_owners":source}
    review={"schema_version":"gx1_m5_family_private_root_review_v1","status":"PASS_CONSUMED_FAMILY_PRIVATE_NATIVE_MECHANICS",
        "family_arm":common["family_arm"],"family_intervention_sha256":common["family_intervention_sha256"],
        "initial_predictions":ref("npz"),"initial_report":ref("initial"),"initial_parent_count":256,
        "private_optimizer_updates":4,"unperturbed_initial_prediction_shared_state_parity":True,
        "bit_exact_resume_reported_and_bound":True,"actual_input_intervention_checked":True,
        "qualification_weights_for_main_fit":False,"test_data_used":False,"source_commit":"commit",
        **{k:ref(k) for k in ("plan","result","receipt","power_after")}}
    result={"schema_version":"gx1_native_m5_family_private_qualification_result_v1","decision":"PASS_M5_FAMILY_PRIVATE_MECHANICS",
        "recipe":ref("prior"),"family_arm":common["family_arm"],"family_intervention_sha256":common["family_intervention_sha256"],
        "initial_predictions":ref("npz"),"family_initial_report":ref("initial"),
        "unperturbed_initial_prediction_shared_state_parity":True,"actual_input_intervention_checked":True,
        "decimal_prior_against_sign_audit":True,"bit_exact_resume":True,"physical_optimizer_updates":4,
        "source_unchanged":True,"test_data_used":False,"private_weights_for_main_fit":False}
    own={"schema_version":"gx1_native_m5_family_initial_measurement_v1","role":"INITIAL_TRAIN","fold":fold,
        "family_arm":common["family_arm"],"family_intervention_sha256":common["family_intervention_sha256"],
        "predictions":ref("npz"),"test_data_used":False}
    documents={"review":review,"plan":{"source_commit":"commit","plan_sha256":"planseal","recipe":ref("prior"),"gpu_uuid":"GPU-bound"},
        "prior":prior,"result":result,"receipt":{"plan_sha256":"planseal","outcome":"COMPLETE","guard_decision":"PASS",
        "trainer_guard_exit_code":0,"progress_observer_exit_code":0,"test_data_used":False},
        "power_after":{"task_state":"Disabled","configured_power_limit_w":250,
        "gpu":["GPU-bound, 250.00, 22.77, 0, 78, 40"],
        "keeper":[{"TaskName":"GX1GpuPowerLimit","State":4},{"TaskName":"GX1HostTelemetryBridge","State":4}],
        "test_data_used":False},"initial":own}
    monkeypatch.setattr(authority.campaign,"require_binding",lambda ref,**kw:ref)
    monkeypatch.setattr(authority.campaign,"read_bound_json",lambda path,digest:documents[path.name])
    return recipe,source,documents


def test_qualification_accepts_bound_same_arm(monkeypatch):
    recipe,source,docs=_qualification(monkeypatch)
    assert authority.require_family_qualification(recipe,source,source_repo=None,source_commit="commit")==docs["review"]


@pytest.mark.parametrize("document,key,value,reason",[
    ("result","family_arm","other","FAMILY_PRIVATE_RESULT"),
    ("result","physical_optimizer_updates",3,"FAMILY_PRIVATE_RESULT"),
    ("result","actual_input_intervention_checked",False,"FAMILY_PRIVATE_RESULT"),
    ("receipt","guard_decision","FAIL","FAMILY_PRIVATE_TERMINAL"),
    ("power_after","configured_power_limit_w",300,"FAMILY_PRIVATE_POWER"),
    ("power_after","task_state","Running","FAMILY_PRIVATE_POWER"),
    ("power_after","gpu",["GPU-bound, 300.00, 22.77, 0, 78, 40"],"FAMILY_PRIVATE_POWER"),
    ("power_after","gpu",["GPU-other, 250.00, 22.77, 0, 78, 40"],"FAMILY_PRIVATE_POWER"),
    ("power_after","gpu",["GPU-bound, invalid, 22.77, 0, 78, 40"],"FAMILY_PRIVATE_POWER"),
    ("power_after","gpu",[],"FAMILY_PRIVATE_POWER"),
    ("power_after","keeper",[{"TaskName":"GX1GpuPowerLimit","State":4},{"TaskName":"GX1HostTelemetryBridge","State":3}],"FAMILY_PRIVATE_POWER"),
    ("power_after","keeper",[{"TaskName":"GX1GpuPowerLimit","State":4},{"TaskName":"wrong","State":4}],"FAMILY_PRIVATE_POWER"),
    ("power_after","test_data_used",True,"FAMILY_PRIVATE_POWER"),
    ("initial","family_intervention_sha256","wrong","FAMILY_PRIVATE_INITIAL"),
    ("prior","numerical_recipe",{"fixed":False,"max_updates":2},"FAMILY_PRIVATE_RECIPE")])
def test_qualification_rejects_unmatched_or_incomplete_evidence(monkeypatch,document,key,value,reason):
    recipe,source,docs=_qualification(monkeypatch);docs[document][key]=value
    with pytest.raises(Exception,match=reason):
        authority.require_family_qualification(recipe,source,source_repo=None,source_commit="commit")


def test_qualification_rejects_changed_source(monkeypatch):
    recipe,source,docs=_qualification(monkeypatch)
    changed=deepcopy(source);changed["pure"]["sha256"]="changed"
    with pytest.raises(Exception,match="FAMILY_PRIVATE_UNCHANGED_SOURCES"):
        authority.require_family_qualification(recipe,changed,source_repo=None,source_commit="commit")
