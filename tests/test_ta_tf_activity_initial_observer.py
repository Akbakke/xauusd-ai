"""Synthetic native-comparison gates; no model/checkpoint/market-data execution."""
from copy import deepcopy
from pathlib import Path
import hashlib

import numpy as np
import pytest

from gx1.audit import observe_market_state_ta_tf_activity_initial_v1 as worker


def arrays():
    value={k:np.zeros(shape,dtype) for k,(shape,dtype) in worker.previous._layout(16,np=np).items()}
    value['parents']=np.arange(16,dtype=np.int64)
    value['issues']=np.arange(16,dtype=np.int64)*300_000_000_000
    for task in ('state','future'):
        value[task+'_raw'][:]=np.arange(160,dtype=np.float32).reshape(16,5,2)/100
        value[task+'_allowed'][:]=4
        value[task+'_eligible'][:]=True
        value[task+'_mature'][:]=True
    return value


def plan():
    value=dict(schema_version=worker.SCHEMA,stage=worker.STAGE,source_repo=str(worker.REPO),
        source_base_commit='b'*40,worker={},cap_runner={},source_owners={},inputs=deepcopy(worker.INPUTS),
        **deepcopy(worker.BOUNDS))
    for name in ('source_view','trial_view','source_model_contract','trial_model_contract',
                 'source_loss_contract','trial_loss_contract'):value[name+'_sha256']='a'*64
    return value


def test_exact_plan_budget_and_no_old_scope():
    value=plan();assert worker._require_plan(value,worker.ROOT/'PLAN.json',worker.ROOT)==value
    for key,new in [('max_backbone_encodings',33),('max_checkpoint_loads',2),('max_optimizer_updates',1),
                    ('max_rng_restores',1),('added_parameter_tensors',18),('training_enabled',True),
                    ('panel_batch_indices',list(range(16))),('max_model_constructions',True)]:
        changed=deepcopy(value);changed[key]=new
        with pytest.raises(RuntimeError,match='EXACT_PLAN'):
            worker._require_plan(changed,worker.ROOT/'PLAN.json',worker.ROOT)
    with pytest.raises(RuntimeError,match='EXACT_PLAN'):
        worker._require_plan(value,worker.previous.ROOT/'PLAN.json',worker.previous.ROOT)


def test_fp32_tolerance_is_only_for_activity():
    baseline=arrays();actual=deepcopy(baseline)
    actual['state_raw'][...,0]+=np.float32(1e-6)
    actual['future_raw'][...,0]-=np.float32(1e-6)
    result=worker._require_trial_parity(actual,baseline,1.25+1e-6,1.25,np=np)
    assert result['tasks']['state']['unchanged_outputs_bitexact'] is True
    assert result['tasks']['future']['max_abs_activity']>0


@pytest.mark.parametrize('field,index',[
    ('parents',(0,)),('issues',(0,)),('state_query',(0,0)),('future_allowed',(0,0)),
    ('state_eligible',(0,0)),('future_mature',(0,0)),
    ('state_raw',(0,0,1)),('future_raw',(0,3,0)),('state_raw',(0,4,1)),
])
def test_actual_axes_unchanged_outputs_and_clone_limits_are_enforced(field,index):
    baseline=arrays();actual=deepcopy(baseline)
    if actual[field].dtype==np.bool_:actual[field][index]=False
    else:actual[field][index]+=1
    with pytest.raises(RuntimeError):worker._require_trial_parity(actual,baseline,1.25,1.25,np=np)


def test_loss_outside_fixed_tolerance_is_rejected():
    value=arrays()
    with pytest.raises(RuntimeError,match='CLONED_LOSS_PARITY'):
        worker._require_trial_parity(value,value,1.25001,1.25,np=np)


def test_default_view_transition_allows_only_reviewed_source_ref():
    name='gx1/contracts/market_state_time_query_model_v2.py'
    before=dict(contract_sha256='a'*64,physical=[1,2,3],source_bindings={
        'time_query_v2_model_source':{'path':str(worker.REPO/name),'sha256':'a'*64},'targets':{'sha256':'c'*64}})
    after=deepcopy(before);after['contract_sha256']='b'*64
    after['source_bindings']['time_query_v2_model_source']['sha256']='b'*64
    bound={'tf_activity_initial_source_owners':{name:after['source_bindings']['time_query_v2_model_source']}}
    worker.require_view_source_transition(before,after,bound)
    after['physical'][0]=9
    with pytest.raises(RuntimeError,match='CURRENT_PHYSICAL_VIEW'):worker.require_view_source_transition(before,after,bound)
    after['physical']=before['physical'];after['source_bindings']['targets']={'sha256':'d'*64}
    with pytest.raises(RuntimeError,match='CURRENT_VIEW_SOURCE_ONLY'):worker.require_view_source_transition(before,after,bound)


def test_trial_view_has_only_explicit_head_change():
    before=dict(contract_sha256='a'*64,physical=[1,2],research_objective={})
    after=dict(deepcopy(before),contract_sha256='b'*64,dedicated_tf_activity=True,
               research_objective={'dedicated_tf_activity':True})
    worker._require_trial_view(before,after)
    after['physical'][0]=7
    with pytest.raises(RuntimeError,match='TRIAL_VIEW_HEAD_ONLY'):worker._require_trial_view(before,after)


def test_partial_capture_keeps_both_phases_and_exact_writer(monkeypatch):
    parts={phase:[dict(arrays(),phase=np.full(16,i,np.uint8))] for i,phase in enumerate(worker.BOUNDS['phases'])}
    seen={}
    def save(name,values,counts,**kw):seen.update(name=name,values=values,kw=kw);return {'saved':True}
    monkeypatch.setattr(worker.cache,'_save_fusion_arrays',save)
    assert worker._save(parts,{'published_npz_bytes':0},partial=True,np=np)=={'saved':True}
    assert seen['name']=='TRAIN_FUSION_PARTIAL.npz' and seen['kw']['root']==worker.ROOT
    assert np.array_equal(seen['values']['phase'],np.r_[np.zeros(16,np.uint8),np.ones(16,np.uint8)])
    with pytest.raises(RuntimeError,match='FULL_CAPTURE'):worker._save(parts,{'published_npz_bytes':0},partial=False,np=np)
    monkeypatch.setitem(worker.BOUNDS,'max_cache_bytes',1)
    with pytest.raises(RuntimeError,match='OUTPUT_BYTES'):worker._save(parts,{'published_npz_bytes':0},partial=True,np=np)


def test_new_source_bridge_preserves_exact_numerical_owner_set(monkeypatch):
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as fit
    names=set(worker.NUMERICAL_CHANGES)|{f'fixed{i}.py' for i in range(85-len(worker.NUMERICAL_CHANGES))}
    original={name:dict(path=str(worker.REPO/name),sha256=hashlib.sha256(name.encode()).hexdigest()) for name in names}
    current=deepcopy(original)
    for name in worker.NUMERICAL_CHANGES:current[name]['sha256']='b'*64
    current.update({name:dict(path=str(worker.REPO/name),sha256='c'*64) for name in worker.EXTRA_OWNERS})
    monkeypatch.setattr(fit,'required_source_paths',lambda **kw:names)
    monkeypatch.setattr(worker.campaign,'require_binding',lambda *a,**kw:None)
    monkeypatch.setattr(worker.subprocess,'check_output',lambda args,**kw:args[2].partition(':')[2].encode())
    worker.current_source_bridge(original,current)
    altered=deepcopy(current);altered['fixed0.py']['sha256']='d'*64
    with pytest.raises(RuntimeError,match='UNREVIEWED_SOURCE'):worker.current_source_bridge(original,altered)
    altered=deepcopy(current);name=next(iter(worker.NUMERICAL_CHANGES));altered[name]=original[name]
    with pytest.raises(RuntimeError,match='EXACT_NUMERICAL_CHANGE_SET'):worker.current_source_bridge(original,altered)


@pytest.mark.parametrize("complete", [True, False])
def test_success_and_failure_receipts_preserve_read_only_consumed_boundary(tmp_path, complete):
    import json
    (tmp_path / "CLAIM.json").write_text("{}\n")
    ref = {"path": str(tmp_path / "PLAN.json"), "sha256": "a" * 64}
    result = worker.cache._publish_fusion_outcome(
        ref, {"source_unchanged": True, "counts": {"optimizer_updates": 0}},
        complete=complete, root=tmp_path, json_bytes_bound=16384)
    values = {k: json.loads((tmp_path / (k.upper() + ".json")).read_text())
              for k in ("result", "terminal")}
    expected = "COMPLETE" if complete else "FAILED_CONSUMED"
    assert values["result"]["status"] == values["terminal"]["status"] == expected
    assert values["terminal"]["exit_code"] == (0 if complete else 1)
    assert values["terminal"]["no_relaunch_under_this_plan"] is True
    assert values["result"]["training_enabled"] is False
    assert values["result"]["test_data_used"] is False
    assert values["result"]["promotion_authorized"] is False
    for key, binding in result.items():
        assert binding["sha256"] == hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest()
