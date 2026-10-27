"""Synthetic saved-output arithmetic and admission checks, without any fit."""
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gx1.scripts import compare_entry_direction_saved_bps_v1 as audit
from gx1.contracts import entry_direction_probability_reference_v1 as probability
from gx1.contracts import entry_direction_block_rank_v1 as rank
from gx1.contracts import entry_native_direction_bps_v1 as bps
from gx1.contracts.entry_native_direction_context_report_v1 import CONTEXT_SPEC


def fixture_arrays(blocks=8, *, paired_classes=True):
    count = blocks*2 if paired_classes else blocks
    group = np.repeat(np.arange(blocks), 2) if paired_classes else np.arange(blocks)
    within = np.tile([0, 1], blocks) if paired_classes else np.zeros(blocks, np.int64)
    decision = np.datetime64('2025-01-06T12:00:00', 'ns').astype(np.int64) + group*7*audit.DAY + within*5*audit.MINUTE
    sign = np.tile(np.array([-1, 1], np.int8), count//2)
    truth = np.repeat((2*sign.astype(np.float64))[:, None], 3, axis=1)
    target = {'parent_rows':np.arange(count,dtype=np.int64), 'bar_start_ns':decision-5*audit.MINUTE,
        'dynamic_entry_m1_row':1000+np.arange(count,dtype=np.int64)*20, 'decision_ns':decision,
        'terminal_time_ns':decision[:,None]+np.array(audit.HORIZONS)*audit.MINUTE,
        'elapsed_wall_seconds':np.broadcast_to(np.array(audit.HORIZONS)*60,(count,3)).copy(),
        'terminal_source_row':1000+np.arange(count,dtype=np.int64)[:,None]*20+np.array(audit.HORIZONS),
        'observed':np.ones((count,3),bool), 'mid_move':truth/100, 'mid_return_bps':truth,
        'sign':np.repeat(sign[:,None],3,axis=1)}
    base={name:target[name].copy() for name in audit.SHARED}
    base['logit_up']=np.zeros((count,3),np.float32)
    base.update({'context__'+name:np.zeros(count,np.int8) for name,labels in CONTEXT_SPEC})
    candidate=copy.deepcopy(base)
    candidate.update(logit_up=target['sign'].astype(np.float32),
        mean_return_bps=truth.astype(np.float32),mid_return_bps=truth.copy())
    scale={'schema_version':bps.SCHEMA,'horizons_minutes':list(audit.HORIZONS),'fit_partition':'train',
        'prefix_rows':count,'cutoff_ns':audit.TRAIN_CUTOFF,
        'target_source':{'path':'/synthetic/targets.npz','sha256':'a'*64},
        'parent_rows_sha256':'b'*64,'decision_ns_sha256':'c'*64,'eligible_targets_sha256':'d'*64,
        'counts':[count]*3,'sum_bps':[float(count)]*3,'sumsq_bps':[float(4*count)]*3,
        'rms_bps':[2.0]*3,'mean_bps':[1.0]*3}
    scale['contract_sha256']=hashlib.sha256(json.dumps(scale,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return base,candidate,target,scale


def test_prediction_schema_and_exact_target_join_preserve_inputs():
    base,candidate,target,scale=fixture_arrays()
    before={name:value.tobytes() for name,value in candidate.items()}
    audit.validate_prediction(candidate,signed_bps=True,rows=len(candidate['parent_rows']),np=np,context_spec=CONTEXT_SPEC)
    audit.join_targets(candidate,target,np=np)
    audit.require_pair(base,candidate,np=np)
    assert before=={name:value.tobytes() for name,value in candidate.items()}


@pytest.mark.parametrize('fault', ['dtype','shape','nonfinite','clock','context','masked_target'])
def test_prediction_schema_rejects_malformed_arrays(fault):
    base,candidate,target,scale=fixture_arrays()
    if fault=='dtype': candidate['logit_up']=candidate['logit_up'].astype(np.float64)
    elif fault=='shape': candidate['sign']=candidate['sign'][:,:2]
    elif fault=='nonfinite': candidate['mean_return_bps'][0,0]=np.nan
    elif fault=='clock': candidate['terminal_time_ns'][0,0]+=audit.MINUTE
    elif fault=='context': candidate['context__'+CONTEXT_SPEC[0][0]][0]=127
    else: candidate['observed'][0,0]=False
    with pytest.raises(RuntimeError,match='PREDICTION_|CONTEXT_|BPS_TARGET_'):
        audit.validate_prediction(candidate,signed_bps=True,rows=len(candidate['parent_rows']),np=np,context_spec=CONTEXT_SPEC)


@pytest.mark.parametrize('name', ['decision_ns','observed','sign','mid_return_bps'])
def test_exact_original_target_join_rejects_shift_or_relabel(name):
    base,candidate,target,scale=fixture_arrays()
    if name=='observed': candidate[name][0,0]=False
    elif name=='sign': candidate[name][0,0]*=-1
    elif name=='decision_ns': candidate[name][0]+=audit.MINUTE
    else: candidate[name][0,0]+=1
    with pytest.raises(RuntimeError,match='TARGET_.*JOIN'):
        audit.join_targets(candidate,target,np=np)


def test_initial_same256_gate_is_bit_exact_and_requires_zero_return_head():
    base,candidate,target,scale=fixture_arrays()
    candidate['logit_up']=base['logit_up'].copy(); candidate['mean_return_bps'].fill(0)
    audit.require_pair(base,candidate,np=np,initial=True)
    candidate['logit_up'][0,0]=np.float32(-0.0)
    with pytest.raises(RuntimeError,match='INITIAL_LOGIT_BIT_PARITY'):
        audit.require_pair(base,candidate,np=np,initial=True)
    candidate['logit_up']=base['logit_up'].copy(); candidate['mean_return_bps'][0,0]=1
    with pytest.raises(RuntimeError,match='INITIAL_ZERO_BPS_HEAD'):
        audit.require_pair(base,candidate,np=np,initial=True)


def test_reader_accepts_exact_saved_archive_without_pickle(tmp_path):
    base,candidate,target,scale=fixture_arrays()
    path=tmp_path/'synthetic_predictions.npz'; np.savez(path,**candidate)
    actual=audit.read_predictions(path,signed_bps=True,rows=len(candidate['parent_rows']),np=np,context_spec=CONTEXT_SPEC)
    assert all(actual[name].tobytes()==value.tobytes() for name,value in candidate.items())
    np.savez(path,**{**candidate,'unplanned':np.ones(1)})
    with pytest.raises(RuntimeError,match='PREDICTION_MEMBERS'):
        audit.read_predictions(path,signed_bps=True,rows=len(candidate['parent_rows']),np=np,context_spec=CONTEXT_SPEC)


def test_point_reports_use_saved_prefix_constant_and_all_fixed_bands():
    base,candidate,target,scale=fixture_arrays()
    reports=audit.point_report(base,candidate,target,scale,cutoff=audit.LATER_CUTOFF,
        probability=probability,bps=bps,np=np)
    assert [item['horizon_minutes'] for item in reports]==[1,5,15]
    for item in reports:
        returned=item['signed_bps']; count=len(base['parent_rows'])
        assert returned['mse_bps2']==0 and returned['prefix_scaled_mse']==0
        assert returned['zero_reference_sse_bps2']==4*count
        assert returned['own_prefix_mean_reference_sse_bps2']==5*count
        assert item['direction']['baseline']['auc']==.5 and item['direction']['candidate']['auc']==1
        assert len(item['direction']['candidate']['reliability'])==10
        assert [band['band'] for band in item['fixed_actual_return_bands']]==list(audit.BANDS)
        assert sum(band['observed_cells'] for band in item['fixed_actual_return_bands'])==count


def test_observed_zero_is_return_supervision_but_not_binary_direction():
    base,candidate,target,scale=fixture_arrays()
    for arrays in (base,candidate,target): arrays['sign'][0,:]=0
    target['mid_return_bps'][0,:]=0; candidate['mid_return_bps'][0,:]=0; candidate['mean_return_bps'][0,:]=0
    rows=audit.point_report(base,candidate,target,scale,cutoff=audit.LATER_CUTOFF,
        probability=probability,bps=bps,np=np)
    assert rows[1]['observed_mature']==len(base['parent_rows'])
    assert rows[1]['observed_mature_zero']==1
    assert rows[1]['direction']['candidate']['rows']==len(base['parent_rows'])-1
    assert rows[1]['fixed_actual_return_bands'][0]['observed_cells']==1


def test_exact_paired_rank_and_mse_intervals_share_draws_and_fixed_seeds():
    base,candidate,target,scale=fixture_arrays()
    result=audit.paired_intervals(base,candidate,target,scale,probability=probability,rank=rank,np=np,replicates=32)
    assert result['family_size']==6 and result['combined_advance_rule_supported']
    assert [row['seed'] for row in result['groupings']]==[20261027,20261028]
    for row in result['groupings']:
        weights=audit.block_weights(row['blocks'],seed=row['seed'],np=np,replicates=32)
        assert row['weights_sha256']==audit.digest(weights,np)
        assert [item['point_difference'] for item in row['contrasts']]==[-5.0,-4.0,.5]
        assert all(item['interval']['undefined_replicates']==0 for item in row['contrasts'])
        assert all(item['interval']['per_interval_alpha']==.05/6 for item in row['contrasts'])
        assert all(item['supported_improvement'] for item in row['contrasts'])
        assert not row['proper_score_paired_descriptive']['logloss']['confirmatory_interval']


def test_undefined_auc_replicates_are_not_replaced_and_block_qualification(monkeypatch):
    base,candidate,target,scale=fixture_arrays(paired_classes=False)
    calls=[]
    def fixed(blocks,*,seed,np,replicates):
        calls.append(seed); weights=np.zeros((replicates,blocks),np.int64); weights[:,0]=blocks
        return weights
    monkeypatch.setattr(audit,'block_weights',fixed)
    result=audit.paired_intervals(base,candidate,target,scale,probability=probability,rank=rank,np=np,replicates=7)
    assert calls==[20261027,20261028]
    assert not result['combined_advance_rule_supported'] and result['undefined_primary_auc_blocks_qualification']
    assert all(row['contrasts'][2]['interval']['undefined_replicates']==7 for row in result['groupings'])
    assert all(row['contrasts'][2]['interval']['low'] is None for row in result['groupings'])


def test_blocks_without_direction_cells_keep_original_draw_multiplicities(monkeypatch):
    base,candidate,target,scale=fixture_arrays()
    for arrays in (base,candidate,target):
        arrays['observed'][:2,:]=False; arrays['sign'][:2,:]=0
    target['mid_return_bps'][:2,:]=0; candidate['mid_return_bps'][:2,:]=0
    def empty_draw(blocks,*,seed,np,replicates):
        weights=np.zeros((replicates,blocks),np.int64); weights[:,0]=blocks; return weights
    monkeypatch.setattr(audit,'block_weights',empty_draw)
    result=audit.paired_intervals(base,candidate,target,scale,probability=probability,rank=rank,np=np,replicates=3)
    assert all(row['contrasts'][2]['interval']['undefined_replicates']==3 for row in result['groupings'])


def test_block_rank_matches_explicit_duplicate_row_auc_with_score_ties():
    base,candidate,target,scale=fixture_arrays(blocks=2)
    labels=(target['sign'][:,1]==1).astype(np.int8)
    scores=np.array([0.,0.,1.,-1.]); ids=np.array([0,0,1,1],np.int64)
    weights=np.array([[2.,1.],[0.,3.]])
    summary=rank.block_rank_pairs(labels,scores,ids)
    numerator,denominator=rank.weighted_rank_pairs(summary,weights)
    for j,weight in enumerate(weights.astype(np.int64)):
        at=np.concatenate([np.tile(np.flatnonzero(ids==i),w) for i,w in enumerate(weight)])
        expected=probability.direction_rank_diagnostics(labels[at],scores[at],score_kind='logit_up')['auc']
        assert numerator[j]/(2*denominator[j])==expected


def test_bootstrap_grouping_requires_at_least_eight_blocks():
    base,candidate,target,scale=fixture_arrays(blocks=2)
    with pytest.raises(RuntimeError,match='MINIMUM_BLOCK_COUNT'):
        audit.paired_intervals(base,candidate,target,scale,probability=probability,rank=rank,np=np,replicates=3)


def test_fixed_bands_have_declared_edges_and_zero_domain():
    values=np.array([0.,-1.,1.,5.,10.,25.,50.,50.1])
    np.testing.assert_array_equal(audit.band_ids(values,np),[0,1,1,2,3,4,5,6])


def guard_for(tmp_path,*,synthetic=True):
    inputs={'baseline_oof':'/synthetic/baseline_oof.npz','candidate_oof':'/synthetic/candidate_oof.npz'}
    gate={'later_allowed':False}
    return audit.make_guard(tmp_path,inputs,[],synthetic=synthetic,
        source_paths=[str(audit.REPO/audit.TEST_FILE)],gate=gate),gate,inputs


def test_guard_blocks_even_later_hash_open_until_initial_gate(tmp_path):
    guard,gate,inputs=guard_for(tmp_path,synthetic=False)
    for path in inputs.values():
        with pytest.raises(RuntimeError,match='INITIAL_PARITY_BEFORE_LATER_OPEN'):
            guard('open',(path,'rb',0))
    gate['later_allowed']=True
    for path in inputs.values(): guard('open',(path,'rb',0))


@pytest.mark.parametrize('relative', ['../outside.npz','../../../GX1_DATA/TEST/sealed.npz'])
def test_owned_fixture_exception_rejects_traversal(tmp_path,relative):
    guard,gate,inputs=guard_for(tmp_path)
    with pytest.raises(RuntimeError,match='TEST_PATH|OUTPUT_ONLY_WRITE|DECLARED_DATA|PAYLOAD'):
        guard('open',(str(tmp_path/relative),'wb',os.O_WRONLY|os.O_CREAT))


def test_exact_readonly_test_source_and_stage_fixture_exceptions(tmp_path):
    guard,gate,inputs=guard_for(tmp_path)
    path=str(audit.REPO/audit.TEST_FILE); guard('open',(path,'rb',0))
    with pytest.raises(RuntimeError,match='TEST_PATH'): guard('open',(path,'wb',os.O_WRONLY))
    with pytest.raises(RuntimeError,match='TEST_PATH'):
        guard('open',(str(audit.REPO/'tests/test_unplanned.py'),'rb',0))
    fixture=str(tmp_path/'test_fixture'/'prices.npz')
    guard('open',(fixture,'wb',os.O_WRONLY|os.O_CREAT)); guard('open',(fixture,'rb',0))
    with pytest.raises(RuntimeError,match='TEST_PATH'):
        guard_for(tmp_path,synthetic=False)[0]('open',(fixture,'rb',0))


def test_guard_forbids_weights_raw_sources_network_and_mutation(tmp_path):
    guard,gate,inputs=guard_for(tmp_path)
    for path in ('/home/andre2/GX1_DATA/market.parquet','/synthetic/model.pt'):
        with pytest.raises(RuntimeError,match='DECLARED_DATA|PAYLOAD'): guard('open',(path,'rb',0))
    with pytest.raises(RuntimeError,match='NETWORK_PROCESS_PICKLE'): guard('socket.connect',())
    with pytest.raises(RuntimeError,match='OUTPUT_ONLY_MUTATION'): guard('os.mkdir',('/outside',0o700,-1))


def test_work_observer_rejects_native_constructor_fit_and_cuda():
    counters={'native_or_fit_calls':0,'torch_object_or_cuda_calls':0}; watch=audit.work_observer(counters)
    frame=SimpleNamespace(f_code=SimpleNamespace(co_filename='/lib/torch/nn/modules/module.py',co_name='__init__'))
    with pytest.raises(RuntimeError,match='NATIVE_FIT_OR_QUERY_CALL'): watch(frame,'call',None)
    frame=SimpleNamespace(f_code=SimpleNamespace(co_filename='/synthetic/pure.py',co_name='run'))
    with pytest.raises(RuntimeError,match='TORCH_OBJECT_OR_CUDA_CALL'):
        watch(frame,'c_call',SimpleNamespace(__name__='_cuda_lazy_init',__module__='torch'))
    assert counters=={'native_or_fit_calls':1,'torch_object_or_cuda_calls':1}


def synthetic_environment(monkeypatch):
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD','1'); monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.delenv('PYTEST_ADDOPTS',raising=False); monkeypatch.setattr(audit.sys,'dont_write_bytecode',True)


def test_synthetic_diagnostics_survive_collection_failure_and_restore_prefix(tmp_path,monkeypatch):
    synthetic_environment(monkeypatch); previous=audit.sys.pycache_prefix; published=[]
    def fake_main(args,*,plugins):
        plugin=plugins[0]
        plugin.pytest_collectreport(SimpleNamespace(nodeid=audit.TEST_FILE,outcome='failed',passed=False,longrepr='original import error'))
        plugin.pytest_collection_finish(SimpleNamespace(items=[])); return 2
    monkeypatch.setattr(pytest,'main',fake_main)
    with pytest.raises(RuntimeError,match='SYNTHETIC_ALL_PASSED'):
        audit.run_synthetic(tmp_path,publish=lambda path,value:published.append(value))
    assert audit.sys.pycache_prefix==previous and published[0]['bytecode_prefix_restored']
    assert published[0]['collection_reports'][0]['longrepr']=='original import error'


@pytest.mark.parametrize('outcome',['passed','failed','skipped'])
def test_synthetic_hooks_record_all_phases_without_losing_failures(tmp_path,monkeypatch,outcome):
    synthetic_environment(monkeypatch); published=[]
    def fake_main(args,*,plugins):
        plugin=plugins[0]; node=audit.TEST_FILE+'::synthetic_case'
        plugin.pytest_collection_finish(SimpleNamespace(items=[SimpleNamespace(nodeid=node)]))
        for phase in ('setup','call','teardown'):
            result=outcome if phase=='call' else 'passed'
            plugin.pytest_runtest_logreport(SimpleNamespace(nodeid=node,when=phase,outcome=result,passed=result=='passed',longrepr='original detail'))
        return 0 if outcome=='passed' else 1
    monkeypatch.setattr(pytest,'main',fake_main)
    if outcome=='passed': assert audit.run_synthetic(tmp_path,publish=lambda path,value:published.append(value))['cases']==1
    else:
        with pytest.raises(RuntimeError,match='SYNTHETIC_ALL_PASSED'):
            audit.run_synthetic(tmp_path,publish=lambda path,value:published.append(value))
    assert len(published[0]['runtest_reports'])==3


def test_plan_requires_exact_schema_source_test_and_protocol_refs(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    plan={'schema_version':audit.SCHEMA+'_synthetic','mode':'synthetic',
        'design':dict(audit.DESIGN_REF),'protocol':dict(audit.PROTOCOL_REF),
        'independent_protocol':dict(audit.INDEPENDENT_PROTOCOL_REF),'metadata':{},'inputs':{},
        'sources':{name:{'path':str(audit.REPO/name),'sha256':'a'*64} for name in audit.SOURCE_PATHS+(audit.TEST_FILE,)},
        'out_dir':str(tmp_path/'qualification'),'wall_seconds':180}
    assert audit.require_plan(plan)==tmp_path/'qualification'
    plan['inputs']={'weights':{'path':'/synthetic/model.pt','sha256':'b'*64}}
    with pytest.raises(RuntimeError,match='INPUT_METADATA_KEYS'): audit.require_plan(plan)


def metadata_case():
    refs={key:{'path':'/synthetic/'+key+'.json','sha256':'a'*64} for key in audit.METADATA_KEYS}
    inputs={key:{'path':'/synthetic/'+key+('.json' if key=='bps_scale' else '.npz'),'sha256':'b'*64} for key in audit.INPUT_KEYS}
    meta={}
    for arm in ('baseline','candidate'):
        numerical={'contract_sha256':arm,'research_objective_sha256':arm,'batch_size':16,'seed':20260911}
        if arm=='candidate': numerical['signed_bps_supervision']=True
        meta[arm+'_recipe']={'fold':dict(audit.FOLD),'input_manifest':refs[arm+'_input_manifest'],'numerical_recipe':numerical}
        result={'recipe':refs[arm+'_recipe'],'fold':dict(audit.FOLD),'phase_state':'COMPLETE',
            'parent_cursor':audit.PREFIX,'oof_cursor':audit.LATER,'optimizer_updates':2976,
            'parent_order_sha256':audit.ORDER_SHA,'source_unchanged':True,
            'qualification_weights_reused':False,'test_data_used':False}
        for phase,key in (('initial','initial_measurement'),('final','final_train_measurement'),('oof','oof_measurement')):
            role=arm+'_'+phase
            result[key]=refs[role]
            meta[role]={'predictions':inputs[role],'fold':dict(audit.FOLD),'test_data_used':False,
                'own_prefix_prior':{'logit_up':[0.,0.,0.]},'feature_view_identity':{'prefix_normalization':{'immutable':'same'}}}
        meta[arm+'_result']=result
        meta[arm+'_input_manifest']={'targets':inputs['price_targets'],'prefix':'same',
            'view_source_bindings':{'direction_component_source':arm,'other':'same'}}
    meta['baseline_qualification']={'technical_fit_resume_scoring_qualified':True,'test_data_used':False}
    meta['candidate_qualification']={'schema_version':'gx1_bps_prefix_fit_root_review_v1',
        'status':'COMPLETE_CONSUMED_PAIRED_COMPARISON_PENDING','recipe':refs['candidate_recipe'],
        'original_worker_result':refs['candidate_result'],'own_prefix_bps_scale':inputs['bps_scale'],
        'fold':dict(audit.FOLD),'test_data_used':False}
    protocol={'recipe':refs['candidate_recipe'],'baseline_recipe':refs['baseline_recipe'],
        'baseline_qualification':refs['baseline_qualification']}
    return meta,refs,inputs,protocol


def test_qualifying_metadata_binds_completed_run_and_all_six_archives():
    audit.require_metadata(*metadata_case())


@pytest.mark.parametrize('fault,reason', [
    ('partial_baseline','COMPLETE_ORIGINAL_RESULT'),
    ('moved_oof','PREDICTION_LINEAGE'),
    ('changed_optimizer','MATCHED_NUMERICAL_RECIPE'),
    ('changed_normalization','MATCHED_INPUT_MANIFESTS'),
    ('changed_prior','MATCHED_PREFIX_PRIOR'),
    ('changed_scale','CANDIDATE_QUALIFICATION')])
def test_qualifying_metadata_rejects_changed_comparison_scope(fault,reason):
    meta,refs,inputs,protocol=metadata_case()
    if fault=='partial_baseline': meta['baseline_result']['phase_state']='SCORING'
    elif fault=='moved_oof': meta['baseline_oof']['predictions']={'path':'/synthetic/wrong.npz','sha256':'c'*64}
    elif fault=='changed_optimizer': meta['candidate_recipe']['numerical_recipe']['batch_size']=32
    elif fault=='changed_normalization': meta['candidate_input_manifest']['prefix']='different'
    elif fault=='changed_prior': meta['candidate_initial']['own_prefix_prior']={'logit_up':[1.,0.,0.]}
    else: meta['candidate_qualification']['own_prefix_bps_scale']={'path':'/synthetic/other.json','sha256':'d'*64}
    with pytest.raises(RuntimeError,match=reason): audit.require_metadata(meta,refs,inputs,protocol)


def test_work_observer_allows_plain_c_methods_without_module_name():
    counters={'native_or_fit_calls':0,'torch_object_or_cuda_calls':0}
    frame=SimpleNamespace(f_code=SimpleNamespace(co_filename='/synthetic/pure.py',co_name='run'))
    audit.work_observer(counters)(frame,'c_call',SimpleNamespace(__name__='get',__module__=None))
    assert not any(counters.values())
