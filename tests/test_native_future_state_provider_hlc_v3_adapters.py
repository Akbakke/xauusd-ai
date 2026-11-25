"""Synthetic V3 routing/scope checks; no model construction or genuine payload IO."""
import ast
from pathlib import Path
import subprocess
import numpy as np
import pytest
from gx1.audit import native_future_state_training_v1 as training
from gx1.audit import native_future_state_provider_hlc_baseline_v1 as baseline
from gx1.audit import native_future_state_provider_hlc_initial_v1 as initial
from gx1.audit import native_provider_hlc_replay_v1 as replay
from gx1.contracts import native_future_state_provider_hlc_v1 as hlc

PARENT='8fd74a76cc45ac7306da4b905c82e9cd9b12976d'
def target(): return hlc.all_tf_provider_hlc_state_contract()
def v3_base():
    return {'schema_version':initial.V3_BASE_SCHEMA,'stage':initial.V3_STAGE,
        'output_root':str(initial.V3_ROOT),'metadata':{'target_contract':target(),
        'dataset':{'path':'/synthetic/v3-data.json','sha256':'a'*64},
        'prior':{'path':'/synthetic/v3-prior.json','sha256':'b'*64}},
        'model_contract':{'contract_sha256':'c'*64}}

def test_v3_scoring_contract_and_legacy_rejection():
    assert baseline.require_provider_hlc_scoring_contract(target())==target()
    old=hlc.provider_hlc_state_contract()
    assert baseline.require_provider_hlc_scoring_contract(old)==old
    with pytest.raises(ValueError): baseline.require_provider_hlc_scoring_contract({'schema_version':'unbound'})

def test_v3_truth_inventory_and_dispatch(monkeypatch):
    assert training.provider_hlc_truth_fields(target())==training.TRUTH+('provider_hlc_observed','state_hlc_observed')
    sentinel=object();seen=[]
    monkeypatch.setattr(training,'load_all_tf_provider_hlc_target_truth',lambda *a,**kw:(seen.append(kw),sentinel)[1])
    assert training.load_provider_hlc_target_truth({},expected_issue_ns=np.array([1],np.int64),target_contract=target())is sentinel
    assert seen[0]['target_contract']==target()

def test_v3_slice_keeps_certificate_and_owns_immutable_arrays(monkeypatch):
    spec=target();fields=training.all_tf_provider_hlc_truth_fields(spec)
    ref={'path':'/synthetic/certificate.json','sha256':'a'*64,
         'target_contract_sha256':spec['contract_sha256'],
         'source_binding_sha256':spec['provider_hlc_source_binding_sha256']}
    truth={k:np.arange(20,dtype=np.int64).reshape(4,5)for k in fields}
    truth['issue_ns']=np.arange(4,dtype=np.int64);truth['target_observation_certificate_ref']=ref
    seen=[]
    from gx1.contracts import market_state_ta_integrated_objective_v1 as objective
    monkeypatch.setattr(objective,'_native_future_state_truth',lambda out,**kw:seen.append((out,kw)))
    take=np.array([0,2,3],np.int64)
    result=training.slice_provider_hlc_truth(truth,take,target_contract=spec)
    assert result['target_observation_certificate_ref']==ref and seen[0][1]['rows']==3
    for key in fields:
        assert np.array_equal(result[key],truth[key][take])
        assert not result[key].flags.writeable and not np.shares_memory(result[key],truth[key])
    with pytest.raises(ValueError): training.slice_provider_hlc_truth(truth,np.array([4],np.int64),target_contract=spec)

def test_v3_loader_rejects_wrong_generation_before_payload(monkeypatch):
    spec=target();data={'schema_version':hlc.SCHEMA,'contract':spec,'model_input':False,'cutoff_ns':training.UPPER,
        **{k:hlc.ALL_TF_SOURCE_BINDINGS[k]for k in ('base_dataset','base_targets','native_manifest')}}
    monkeypatch.setattr(training,'read',lambda ref:data)
    monkeypatch.setattr(training.np,'load',lambda *a,**k:pytest.fail('unexpected target payload IO'))
    with pytest.raises(ValueError,match='ALL_TF_PROVIDER_DATASET_BINDING'):
        training.load_provider_hlc_target_truth({},expected_issue_ns=np.array([1],np.int64),target_contract=spec)

def test_initial_scope_is_fixed_and_immutable():
    legacy=initial.initial_version_scope();assert not legacy['v3']and legacy['root']==initial.SEG_ROOT
    current=initial.initial_version_scope(v3_base())
    assert current['v3']and current['root']==initial.V3_ROOT and current['model_sha']=='c'*64
    with pytest.raises(TypeError): current['root']=Path('/unbound')
    bad=v3_base();bad['output_root']='/unbound'
    with pytest.raises(ValueError,match='V3_FIXED_SCOPE'):initial.initial_version_scope(bad)
    with pytest.raises(ValueError,match='INITIAL_EXACT_GENERATION'):initial.initial_version_scope({'schema_version':'unknown'})

def test_segment_roots_are_explicit_and_disjoint():
    sha='a'*64;old=initial.segment_paths(sha);new=initial.segment_paths(sha,base_root=initial.V3_ROOT)
    assert old[0].parent.parent==initial.SEG_ROOT and new[0].parent.parent==initial.V3_ROOT
    assert old!=new
    with pytest.raises(ValueError,match='SEGMENT_BASE_ROOT'):initial.segment_paths(sha,base_root=Path('/unbound'))

def test_legacy_replay_cannot_accept_v3_initial():
    with pytest.raises(ValueError,match='SEGMENTED_INITIAL_EXACT_GENERATION'):
        replay.all_tf_require_segmented_initial_metadata({}, {}, v3_base(), {})

def test_v3_replay_gate_has_no_main_fit_authority():
    b={'path':str(replay.V3_PLAN),'sha256':'a'*64};g=replay.v3_gate(b,True)
    assert g['plan']==b and g['main_fit_authorized']is False and g['full_epoch_training_allowed']is False
    assert all(v['remaining_invocations']==1 for v in g['lanes'].values())
    assert replay.V3_ROOTS['warm']!=replay.ALL_ROOTS['warm']

def test_exact_v3_cli_dispatch(monkeypatch):
    sentinel=object();monkeypatch.setattr(initial,'main_v3_segment',lambda:sentinel)
    monkeypatch.setattr(initial.sys,'argv',['owner','--all-tf-v3-score-segment-plan-sha256','a'*64])
    assert initial.main()is sentinel
    monkeypatch.setattr(replay,'main_v3',lambda:sentinel)
    monkeypatch.setattr(replay.sys,'argv',['owner','--all-tf-v3-plan-sha256','a'*64,'--lane','warm'])
    assert replay.main()is sentinel

@pytest.mark.parametrize('name',['provider_hlc_truth_fields','slice_provider_hlc_truth','load_provider_hlc_target_truth'])
def test_v2_loader_body_unchanged_after_explicit_v3_dispatch(name):
    path='gx1/audit/native_future_state_training_v1.py'
    historical=subprocess.check_output(['git','show',PARENT+':'+path],cwd=initial.REPO,text=True)
    old=next(n for n in ast.parse(historical).body if isinstance(n,ast.FunctionDef)and n.name==name)
    new=next(n for n in ast.parse((initial.REPO/path).read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name)
    branches=[i for i,n in enumerate(new.body)if isinstance(n,ast.If)and isinstance(n.test,ast.Call)and isinstance(n.test.func,ast.Name)and n.test.func.id=='_provider_hlc_all_tf_v3']
    assert len(branches)==1;new.body.pop(branches[0]);assert ast.dump(new)==ast.dump(old)

def test_v3_corrupt_npz_bytes_fail_before_final_publication(monkeypatch):
    base=v3_base();plan={**base,'schema_version':initial.V3_SEG_SCHEMA,
        'base_root':base['output_root'],'base_plan':{'path':str(initial.V3_BASE_PLAN),'sha256':'a'*64},
        'prefix':{'cursor':0}}
    plan.pop('output_root');values={k:np.ones((2,5),np.float32)for k in baseline.OUTPUT_KEYS}
    event={'batch_cursor':1,'left':0,'right':2,'gradients_none':True,
        'batch':{'output_sha256':{k:initial.core.digest(v)for k,v in values.items()}}}
    original=np.savez
    def corrupt(stream,**arrays):
        copied={k:v.copy()for k,v in arrays.items()};copied['activity_logits'][0,0]+=1
        original(stream,**copied)
    published=[];monkeypatch.setattr(initial.np,'savez',corrupt)
    monkeypatch.setattr(initial,'segment_publish',lambda *args:published.append(args))
    counts=dict.fromkeys(initial.V3_COUNT_KEYS,0)
    with pytest.raises(ValueError,match='PREPUBLICATION_OUTPUT_BYTES'):
        initial.segment_publish_batch(plan,{'path':'/synthetic/segment.json','sha256':'d'*64},
            {},event,values,None,counts=counts)
    assert published==[] and counts['in_memory_output_archive_strict_loads']==1
    assert counts['output_archive_write_calls']==counts['output_archive_readbacks']==0

def test_v3_corrupt_checkpoint_memory_state_fails_before_final_publication(monkeypatch):
    import io
    import torch
    base=v3_base();base['metadata'].update(input_binding={'schema_version':'synthetic'},
        normalization={'path':'/synthetic/norm.json','sha256':'e'*64},truth_sha256='f'*64)
    class Model:
        training=True
        def state_dict(self):return {'weight':torch.tensor([1.])}
    monkeypatch.setattr(initial.core,'training_rng',lambda:{'cpu':torch.tensor([1],dtype=torch.uint8)})
    original=torch.load
    def corrupt(source,**kwargs):
        assert isinstance(source,io.BytesIO)
        result=original(source,**kwargs);result['seed']=-1;return result
    monkeypatch.setattr(torch,'load',corrupt);published=[]
    monkeypatch.setattr(initial,'segment_publish',lambda *args:published.append(args))
    counts=dict.fromkeys(initial.V3_COUNT_KEYS,0)
    with pytest.raises(ValueError,match='PREPUBLICATION_INITIAL_FULL_INVENTORY_STATE_RNG'):
        initial.segment_write_initial_checkpoint(Model(),base,
            {'path':str(initial.V3_BASE_PLAN),'sha256':'a'*64},{},counts)
    assert published==[] and counts['in_memory_checkpoint_strict_loads']==1
    assert counts['new_checkpoint_writes']==counts['new_checkpoint_readbacks']==0

def test_v3_memory_validation_counters_are_separate_from_physical_readbacks():
    old=initial.segment_limits(0,True);new=initial.segment_limits(0,True,v3=True)
    assert set(new)==set(old)|{'in_memory_checkpoint_strict_loads','in_memory_output_archive_strict_loads'}
    assert new['in_memory_checkpoint_strict_loads']==new['new_checkpoint_readbacks']==1
    assert new['in_memory_output_archive_strict_loads']==new['output_archive_readbacks']==initial.BATCHES


def test_v3_budget_selects_14_consumed_fields_from_32_field_archive(monkeypatch):
    spec=target();fields=training.provider_hlc_truth_fields(spec);rows=initial.ROWS
    headers={key:{'dtype':'<i8'if key in ('issue_ns','target_start_ns','validity_ns')else'|b1',
                  'shape':[rows]if key=='issue_ns'else[rows,5],'sha256':'a'*64}for key in fields}
    extras={'synthetic_raw_fact_%02d'%j:{'dtype':'<f8','shape':[rows,5,4],'sha256':'b'*64}for j in range(18)}
    headers.update(extras);base_headers={k:v for k,v in headers.items()if k not in ('provider_hlc_observed','state_hlc_observed')}
    seen=[]
    def metadata_only(binding):
        assert binding==hlc.ALL_TF_SOURCE_BINDINGS['base_dataset'];seen.append(binding)
        return {'array_headers':base_headers}
    monkeypatch.setattr(initial,'read',metadata_only)
    monkeypatch.setattr(initial.np,'load',lambda *args,**kw:pytest.fail('budget must not decode arrays'))
    data={'array_headers':headers};budget=initial.v3_truth_input_budget(data,spec)
    assert budget=={'V3_truth_arrays':14,'V3_truth_ndarray_bytes':50700650,'V3_target_archive_fields':32}
    headers['synthetic_raw_fact_00']['shape']=[rows,50000,4]
    assert initial.v3_truth_input_budget(data,spec)==budget and len(seen)==2
    missing=dict(headers);missing.pop('side_known')
    with pytest.raises(ValueError,match='V3_TRUTH_ARCHIVE_AND_CONSUMED_FIELDS'):
        initial.v3_truth_input_budget({'array_headers':missing},spec)
