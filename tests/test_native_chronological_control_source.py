"""Synthetic source binding tests; no actual market data or production model."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.contracts.local_random_access_campaign_v2 import canonical_sha256, file_sha256
from gx1.contracts.unified_exit_bounded_val_cohort_v1 import (
    build_chronological_control_cohort, require_bounded_val_cohort,
)
from gx1.contracts.unified_exit_random_access_val_factory_v1 import RandomAccessValStateFactoryV1 as Factory
from gx1.contracts.unified_exit_pilot_final_bindings_v1 import build_composite_normalization_binding, build_split_sequence_binding
from gx1.contracts.unified_exit_pilot_normalization_v1 import build_first_state_entry_bridge_witness
from gx1.contracts.unified_exit_random_access_index_v1 import build_random_access_index_v2
from gx1.models.entry_v10.direction_decision_contract import UNIFIED_EXIT_PATH_PRICE_FIELDS
from tests.test_unified_exit_pilot_final_bindings_v1 import _base, _summary
from tests.test_unified_exit_random_access_val_rollout_v1 import _closure, _objective


def _binding(path):
    return {'path': str(path), 'sha256': file_sha256(path)}


def _control_plan(tmp_path, population=6000):
    frame = pd.DataFrame({'entry_row_index': np.arange(population, dtype='int64'),
                          'parent_entry_row_index': np.arange(population, dtype='int64') + 11,
                          'entry_time_ns': pd.date_range('2026-03-01T00:00Z', periods=population, freq='5min').asi8})
    frame['first_state_time_ns'] = frame.entry_time_ns + 300_000_000_000
    frame['parent_m1_start_row'] = np.arange(population, dtype='int64') * 5 + 479
    frame['child_m1_start_row'] = frame.parent_m1_start_row
    frame['successor_transition_count'] = np.full(population, 3, dtype='int64')
    frame['lifecycle_state_count'] = frame.successor_transition_count + 1
    frame['economic_terminal'] = False
    frame['right_censored'] = True
    frame['entry_bid'] = 100.0
    frame['entry_ask'] = 100.1
    # Include IDs above the old 5508 limit, with distinct parent coordinates.
    children = np.linspace(17, population-1, 256, dtype='int64')
    parents = frame.iloc[children].parent_entry_row_index.to_numpy(dtype='int64')
    index = tmp_path/'train.parquet';frame.to_parquet(index,index=False)
    rows = tmp_path/'CONTROL256.npy';np.save(rows, parents)
    design = {'schema_version':'gx1_frozen_chronological_learning_design_v1',
              'status':'DESIGN_AND_CONTROL_IDS_FROZEN_NOT_EXECUTABLE',
              'scope':{'test_sealed':True,'existing_control_periods_are_reused_development':True,
                       'native_launch_authorized':False,'one_experiment':True},
              'selection':{'control_entries':256},'budget':{'later_control_entries':256},
              'calendar':{'train_control_cutoff':'2026-03-01T00:00Z',
                          'development_control_entry_end_exclusive':'2026-06-01T00:00Z',
                          'index':_binding(index),'bindings':{'CONTROL256_PARENT_ROWS':_binding(rows)}}}
    path = tmp_path/'DESIGN.json';path.write_text(json.dumps(design))
    return _binding(path), frame, children, parents, design


def _factory_kwargs(count=6, split='train'):
    times = pd.date_range('2026-03-01T00:00Z', periods=count*5+510, freq='min')
    starts = np.arange(count)*5+479
    entries = times[starts]-pd.Timedelta(minutes=5)
    counts = np.arange(count,dtype='int64')%5+3
    bid = 100 + np.arange(len(times))*.001
    price_names = set(UNIFIED_EXIT_PATH_PRICE_FIELDS) | {'bid_open','ask_open','bid_close','ask_close',
                   'bid_high','ask_high','bid_low','ask_low','volume'}
    prices = {name: (np.ones(len(times)) if name=='volume' else
                    bid + (.1 if name.startswith('ask') else 0) +
                    (.03 if name.endswith('high') else -.03 if name.endswith('low') else 0))
              for name in price_names}
    child = pd.DataFrame({'time':times,**prices})
    closure = _closure(times)
    composite = build_composite_normalization_binding(base_artifact=_base(),base_path='/fixture/base.json',
        base_file_sha256='1'*64,summary_normalization=_summary(),summary_manifest_path='/fixture/summary.json',
        summary_manifest_file_sha256='2'*64,summary_manifest_sha256='3'*64)
    hashes = {k:'1'*64 for k in ('entry_parquet','entry_manifest','child_m1','child_m1_manifest',
        'successor_counts','summary_manifest','first_state_bridge','split_sequence_binding','composite_normalization',
        'closure_authority','random_access_index','random_access_index_manifest','random_access_index_root')}
    hashes['child_m1']='8'*64
    sequence = build_split_sequence_binding(split=split,entry_times=entries,m1_times=times,
        successor_transition_counts=counts,child_admission_file_sha256='1'*64,child_admission_witness_sha256='2'*64,
        child_parquet_sha256=hashes['entry_parquet'],child_manifest_file_sha256=hashes['entry_manifest'],
        child_manifest_contract_sha256='3'*64,m1_source_sha256=hashes['child_m1'],
        m1_manifest_file_sha256=hashes['child_m1_manifest'],closure_authority_file_sha256=hashes['closure_authority'],
        closure_authority_sha256=closure['artifact_sha256'])
    bridge = build_first_state_entry_bridge_witness(split=split,entry_times=entries,m1_times=times,
        child_admission_sha256='1'*64,child_parquet_sha256=hashes['entry_parquet'],entry_sequence_audit_sha256='2'*64,
        m1_source_sha256=hashes['child_m1'],closure_authority_sha256=closure['artifact_sha256'],
        state_view_source_sha256='3'*64,lifetime_summary_registry_sha256='4'*64,
        train_normalization_sha256=composite['composite_normalization_sha256'],
        m1_bid_open=prices['bid_open'],m1_ask_open=prices['ask_open'])
    index,_ = build_random_access_index_v2(split=split,entry_times=entries,parent_entry_times=entries,
        child_m1_times=times,parent_m1_times=times,successor_transition_counts=counts,
        entry_bid=prices['bid_open'][starts],entry_ask=prices['ask_open'][starts],
        episode_binding_sha256_by_entry=bridge['first_state_episode_binding_sha256_by_entry'],
        entry_fill_binding_sha256_by_entry=bridge['entry_fill_binding_sha256_by_entry'])
    from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SIGNAL_DIM,MODEL_NATIVE_CTX_CONT_DIM,MODEL_NATIVE_CTX_CAT_DIM
    features = {'signal':np.broadcast_to(np.arange(len(times),dtype='float32')[:,None],(len(times),MODEL_NATIVE_SIGNAL_DIM)),
                'ctx_cont':np.zeros((len(times),MODEL_NATIVE_CTX_CONT_DIM),dtype='float32'),
                'ctx_cat':np.zeros((len(times),MODEL_NATIVE_CTX_CAT_DIM),dtype='int64')}
    source = SimpleNamespace(_m1_times=times,_m1_feature_times=times,_m1_features=features,_m1=prices)
    manifest = {'manifest_sha256':'5'*64, 'economic_step_model_sha256':'6'*64,
                'economic_step_source_manifest_sha256':'7'*64}
    provider = SimpleNamespace(economic_exit_step_manifest=manifest,parent_m1_row_offset=0,
                               state_m1_source_sha256=hashes['child_m1'],state_m1_source_manifest_sha256=hashes['child_m1_manifest'],
                               parent_m1_source_sha256='8'*64,parent_m1_source_manifest_sha256='9'*64,
                               market_closure_authority_sha256=closure['artifact_sha256'])
    def mtf(request):
        from gx1.features.htf_features import MULTI_TF_TIMEFRAMES
        result={}
        for tf in MULTI_TF_TIMEFRAMES:
            suffix=tf.lower();dim=composite['base_feature_normalization']['artifact']['contract']['surfaces']['mtf_'+suffix]['field_count']
            result['exit_mtf_history_'+suffix]=np.zeros((1,dim),dtype='float32')
            result['exit_mtf_history_time_ns_'+suffix]=np.asarray(request,dtype='int64')
            result['exit_mtf_gather_'+suffix]=np.array([0],dtype='int64')
        return result
    return dict(entry_rows=pd.DataFrame({'time':entries}),child_m1=child,successor_transition_counts=counts,
                random_access_index=index,first_state_bridge=bridge,sequence_binding=sequence,
                composite_normalization=composite,closure_authority=closure,source_owner=source,
                mtf_materializer=mtf,economic_step_provider=provider,economic_step_manifest=manifest,
                economics_objective_contract=_objective(),artifact_file_sha256=hashes,source_split=split)


def test_train_factory_keeps_physical_ids_counts_prices_and_causal_state():
    kw=_factory_kwargs();factory=Factory(**kw)
    assert factory.source_split == factory.factory_receipt['split'] == 'train'
    assert factory.factory_receipt['entry_pair_count']==6
    assert [r['entry_row_index'] for r in factory.entries]==list(range(6))
    assert np.array_equal(factory.counts,kw['successor_transition_counts'])
    entry=factory.entries[5];state=factory.materialize_state(entry,2)
    row=int(factory.starts[5])+2
    assert state['m1_row_index']==row and state['decision_time_ns']==factory.times.asi8[row]+60_000_000_000
    assert np.array_equal(state['m1_local_history_x'],factory.signal[row-479:row+1])
    assert state['trade_path_length']==3 and state['lifetime_summary_x'].shape==(2,7)
    cached=factory.materialize_cached_cpu_batch([(entry,2)],workers=0)[0]
    assert np.array_equal(cached['trade_path_tail_x'],state['trade_path_tail_x'])
    assert np.array_equal(cached['lifetime_summary_x'],state['lifetime_summary_x'])
    with pytest.raises(RuntimeError,match='STATE_REQUEST_INVALID'):
        factory.materialize_state(entry,entry['available_state_count'])
    with pytest.raises(RuntimeError,match='TRAIN_SOURCE_REQUIRES_BOUND_COHORT'):
        factory.bind_rollout(entry_decision_representations=torch.zeros(6,2),model_state_sha256='1'*64,
                              checkpoint_file_sha256='2'*64,compute_guard_max_model_forwards=100,
                              compute_guard_max_materialized_state_views=100,compute_guard_max_wall_seconds=1.)


def test_default_val_still_requires_exact_june_population():
    kw=_factory_kwargs();kw.pop('source_split')
    with pytest.raises(RuntimeError,match='COHORT_INVALID'):Factory(**kw)
    kw=_factory_kwargs(5508,'val');kw.pop('source_split');factory=Factory(**kw)
    assert factory.source_split=='val' and len(factory.entries)==5508
    assert factory.factory_receipt['split']=='val'
    assert factory.materialize_state(factory.entries[-1],0)['state_index']==0


@pytest.mark.parametrize('fault',['test','sequence_split','bridge_split','normalization','counts','price','parent_offset'])
def test_train_source_preserves_existing_identity_checks(fault):
    kw=_factory_kwargs()
    if fault=='test':kw['source_split']='test'
    elif fault=='sequence_split':kw['sequence_binding']['split']='val'
    elif fault=='bridge_split':
        kw['first_state_bridge']['split']='val'
        kw['first_state_bridge']['witness_sha256']=canonical_sha256({k:v for k,v in kw['first_state_bridge'].items() if k!='witness_sha256'})
    elif fault=='normalization':kw['artifact_file_sha256']['entry_parquet']='9'*64
    elif fault=='counts':kw['successor_transition_counts']=kw['successor_transition_counts'][:-1]
    elif fault=='price':kw['child_m1'].loc[0,'bid_close']+=1
    elif fault=='parent_offset':kw['economic_step_provider'].parent_m1_row_offset=1
    with pytest.raises(RuntimeError):Factory(**kw)


def test_control_cohort_binds_original_sparse_parent_ids_and_reuses_native_rollout(tmp_path, monkeypatch):
    from tests import test_unified_exit_random_access_val_rollout_v1 as fixture_owner
    _fixture = fixture_owner._fixture
    monkeypatch.setattr(fixture_owner, "VAL_ENTRY_COHORT_SIZE", 6000)
    from gx1.contracts.unified_exit_random_access_val_rollout_v1 import require_random_access_val_rollout_contract
    binding,frame,children,parents,_=_control_plan(tmp_path)
    scope=build_chronological_control_cohort(binding)
    assert require_bounded_val_cohort(scope)==scope
    assert scope['source_split']=='train' and scope['split']=='val'
    assert scope['entry_row_indices']==children.tolist() and scope['parent_entry_row_indices']==parents.tolist()
    assert scope['population_rows']==6000 and scope['test_data_used'] is False
    assert 'reused_development' in scope['evaluation_role']
    # Existing rollout test owner accepts the same physical sparse IDs above 5508.
    model,reps,adapter,contract=_fixture(thresholds=np.ones((6000,2)),counts=np.full(6000,3),evaluation_cohort=scope)
    assert [e['entry_row_index'] for e in adapter.entries]==children.tolist()
    assert require_random_access_val_rollout_contract(contract)['entry_pair_cohort_size']==256
    from gx1.contracts.unified_exit_random_access_val_evaluator_v1 import run_resumable_random_access_val_evaluation_v1
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_complete_val_observation
    from tests.test_unified_exit_random_access_val_evaluator_v1 import _checkpoint_binding, _entry_policy, _with_route_outputs
    model = _with_route_outputs(model)
    checkpoint = _checkpoint_binding(contract, adapter, tmp_path)
    kwargs = dict(model=model,entry_decision_representations=reps,adapter=adapter,
                  checkpoint_binding=checkpoint,entry_route_diagnostics={},
                  entry_policy_decisions=_entry_policy(adapter,checkpoint),policy_batch_size=256)
    paused=run_resumable_random_access_val_evaluation_v1(**kwargs,progress_path=tmp_path/'resume.json',
        result_path=tmp_path/'resumed.json',max_forwards_this_invocation=1)
    assert paused['decision']=='PAUSED_RESUMABLE'
    resumed=run_resumable_random_access_val_evaluation_v1(**kwargs,progress_path=tmp_path/'resume.json',
        result_path=tmp_path/'resumed.json',max_forwards_this_invocation=100)
    direct=run_resumable_random_access_val_evaluation_v1(**kwargs,progress_path=tmp_path/'direct-progress.json',
        result_path=tmp_path/'direct.json',max_forwards_this_invocation=100)
    assert resumed['trade_outcomes']==direct['trade_outcomes']
    assert resumed['entry_exit_policy_metrics']==direct['entry_exit_policy_metrics']
    assert resumed['evaluation_cohort']==scope and resumed['source_population_fully_evaluated'] is False
    from gx1.contracts.unified_exit_random_access_val_evaluator_v1 import require_random_access_val_evaluation_result_v1
    validation=dict(rollout_contract_sha256=contract['contract_sha256'],
                    checkpoint_binding_sha256=checkpoint['binding_sha256'],
                    execution_contract_sha256=resumed['execution_contract_sha256'])
    assert require_random_access_val_evaluation_result_v1(resumed,**validation)==resumed
    bad=copy.deepcopy(resumed);bad['source_population_fully_evaluated']=True
    bad['semantic_result_sha256']=canonical_sha256({k:v for k,v in bad.items() if k!='semantic_result_sha256'})
    with pytest.raises(RuntimeError,match='COHORT_MISMATCH'):
        require_random_access_val_evaluation_result_v1(bad,**validation)
    assert sorted({r['entry_row_index'] for r in resumed['trade_outcomes']})==children.tolist()
    with pytest.raises(RuntimeError,match='FULL_VAL_OUTCOMES_REQUIRED'):require_complete_val_observation(resumed)


@pytest.mark.parametrize('fault',['reorder','duplicates','future','past','design','source_drift','changed_parent','changed_child'])
def test_control_rejects_reselection_temporal_or_source_drift(tmp_path,fault):
    binding,frame,children,parents,design=_control_plan(tmp_path)
    if fault in ('reorder','duplicates'):
        rows=parents[::-1] if fault=='reorder' else np.full(256,parents[0],dtype='int64')
        path=Path(design['calendar']['bindings']['CONTROL256_PARENT_ROWS']['path']);np.save(path,rows)
        design['calendar']['bindings']['CONTROL256_PARENT_ROWS']=_binding(path)
    elif fault in ('future','past'):
        key='development_control_entry_end_exclusive' if fault=='future' else 'train_control_cutoff'
        design['calendar'][key]='2026-03-02T00:00Z'
    elif fault=='design':design['scope']['native_launch_authorized']=True
    elif fault in ('source_drift','changed_parent','changed_child'):
        path=tmp_path/'rebound.parquet';other=frame.copy()
        key={'source_drift':'successor_transition_count','changed_parent':'parent_entry_row_index','changed_child':'entry_row_index'}[fault]
        other.loc[children[0],key]+=1;other.to_parquet(path,index=False)
        with pytest.raises(RuntimeError,match='COORDINATES_INVALID'):
            build_chronological_control_cohort(binding,source_index_binding=_binding(path))
        return
    path=Path(binding['path']);path.write_text(json.dumps(design))
    with pytest.raises(RuntimeError):build_chronological_control_cohort(_binding(path))


def test_control_metadata_cannot_be_resealed_as_train_or_new_rows(tmp_path):
    binding,_,_,_,_=_control_plan(tmp_path);scope=build_chronological_control_cohort(binding)
    for key,value in [('source_split','val'),('evaluation_role','untouched_holdout'),
                      ('parent_entry_row_indices',list(range(256))),('test_data_used',True)]:
        bad=copy.deepcopy(scope);bad[key]=value
        bad['cohort_sha256']=canonical_sha256({k:v for k,v in bad.items() if k!='cohort_sha256'})
        with pytest.raises(RuntimeError,match='BINDING_MISMATCH'):require_bounded_val_cohort(bad)
    # Hashes are rechecked even when coordinates have been cached.
    rows=Path(json.loads(Path(binding['path']).read_text())['calendar']['bindings']['CONTROL256_PARENT_ROWS']['path'])
    rows.write_bytes(b'changed-after-first-binding')
    with pytest.raises(RuntimeError):require_bounded_val_cohort(scope)


def test_train_materialization_has_no_old_june_id_ceiling():
    factory=Factory(**_factory_kwargs(5509,'train'))
    assert factory.materialize_state(factory.entries[5508],1)['m1_row_index']==factory.starts[5508]+1


def test_from_artifacts_propagates_physical_train_identity(tmp_path,monkeypatch):
    from gx1.contracts import unified_exit_random_access_val_factory_v1 as owner
    kw=_factory_kwargs()
    names=('entry_parquet','entry_manifest','child_m1','child_m1_manifest','successor_counts',
           'summary_manifest','first_state_bridge','split_sequence_binding','composite_normalization',
           'closure_authority','random_access_index','random_access_index_manifest','random_access_index_root')
    paths={name:tmp_path/name for name in names}
    for path in paths.values():path.write_text('{}')
    kw['random_access_index'].to_parquet(paths['random_access_index'],index=False)
    kw['entry_rows'].to_parquet(paths['entry_parquet'],index=False)
    kw['child_m1'].to_parquet(paths['child_m1'],index=False)
    with paths['successor_counts'].open('wb') as handle:np.save(handle,kw['successor_transition_counts'])
    for name, parquet in [('entry_manifest','entry_parquet'),('child_m1_manifest','child_m1')]:
        paths[name].write_text(json.dumps({'split':'train','decision':'PASS','test_accessed':False,
                                         'output_parquet_sha256':file_sha256(paths[parquet])}))
    import hashlib
    summary={'split':'train','decision':'PASS','entry_pair_population':6,'test_accessed':False,
             'm1_source_sha256':file_sha256(paths['child_m1']),
             'm1_manifest_sha256':file_sha256(paths['child_m1_manifest']),
             'successor_counts_sha256':hashlib.sha256(kw['successor_transition_counts'].astype('<i8').tobytes()).hexdigest()}
    paths['summary_manifest'].write_text(json.dumps(summary))
    checked=[]
    def manifest(value,**args):
        assert args['expected_split']=='train' and args['verify_sources'] is True
        checked.append('train')
        return {'manifest_sha256':'1'*64,'source_bindings':{
            key:{'sha256':file_sha256(paths[name])} for key,name in
            [('first_state_bridge','first_state_bridge'),('sequence_binding','split_sequence_binding'),
             ('composite_normalization','composite_normalization')]}}
    root={'splits':{'train':{'index_parquet_path':str(paths['random_access_index']),
                             'index_parquet_sha256':file_sha256(paths['random_access_index']),
                             'manifest_path':str(paths['random_access_index_manifest']),'manifest_sha256':'1'*64}}}
    monkeypatch.setattr(owner,'require_random_access_index_manifest',manifest)
    monkeypatch.setattr(owner,'require_random_access_index_root',lambda value:root)
    class Capture(Factory):
        def __init__(self,**args):self.args=args
    call={name+'_path':path for name,path in paths.items()}
    call.update({name:kw[name] for name in ('source_owner','mtf_materializer','economic_step_provider',
                                           'economic_step_manifest','economics_objective_contract')})
    obj=Capture.from_artifacts(**call,source_split='train')
    assert checked==['train'] and obj.args['source_split']=='train'
    assert obj.args['random_access_index'].equals(kw['random_access_index'])
    with pytest.raises(RuntimeError,match='ARTIFACT_BINDING_INVALID'):Capture.from_artifacts(**call)
    with pytest.raises(RuntimeError,match='SOURCE_SPLIT_INVALID'):Capture.from_artifacts(**call,source_split='test')
    summary['entry_pair_population']=7;paths['summary_manifest'].write_text(json.dumps(summary))
    with pytest.raises(RuntimeError,match='INDEX_ROOT_INVALID'):Capture.from_artifacts(**call,source_split='train')


def test_existing_factory_binds_only_exact_train_control_and_frozen_normalization(tmp_path):
    binding,_,children,_,design=_control_plan(tmp_path)
    kw=_factory_kwargs(6000,'train')
    index=Path(design['calendar']['index']['path'])
    kw['random_access_index'].to_parquet(index,index=False)
    rows=Path(design['calendar']['bindings']['CONTROL256_PARENT_ROWS']['path'])
    np.save(rows,children)
    design['calendar']['index']=_binding(index)
    design['calendar']['bindings']['CONTROL256_PARENT_ROWS']=_binding(rows)
    path=Path(binding['path']);path.write_text(json.dumps(design))
    scope=build_chronological_control_cohort(_binding(path))
    kw['artifact_file_sha256']['random_access_index']=file_sha256(index)
    factory=Factory(**kw)
    args=dict(entry_decision_representations=torch.zeros(256,2),model_state_sha256='1'*64,
              checkpoint_file_sha256='2'*64,compute_guard_max_model_forwards=100,
              compute_guard_max_materialized_state_views=100000,compute_guard_max_wall_seconds=1.,
              evaluation_cohort=scope)
    contract,adapter=factory.bind_rollout(**args)
    assert [r['entry_row_index'] for r in adapter.entries]==children.tolist()
    assert contract['normalization_sha256']==factory.normalization['normalization_sha256']
    assert [r['available_state_count'] for r in adapter.entries]==(kw['successor_transition_counts'][children]+1).tolist()
    factory.source_split='val'
    with pytest.raises(RuntimeError,match='SOURCE_MISMATCH'):factory.bind_rollout(**args)
    factory.source_split='train';factory.artifact_file_sha256['random_access_index']='0'*64
    with pytest.raises(RuntimeError,match='SOURCE_MISMATCH'):factory.bind_rollout(**args)


def test_evaluator_checks_physical_source_and_parent_ids_before_any_forward(tmp_path,monkeypatch):
    import inspect
    from gx1.scripts import run_unified_exit_random_access_val_v1 as evaluator
    binding,frame,children,_,_=_control_plan(tmp_path)
    scope=build_chronological_control_cohort(binding)
    factory=SimpleNamespace(source_split='train',artifact_file_sha256={'random_access_index':scope['source_index']['sha256']})
    args={name:None for name,p in inspect.signature(evaluator.evaluate_bound_full_val_v1).parameters.items()
          if p.default is inspect.Parameter.empty}
    args.update(frame=frame.iloc[children].copy(),state_factory=factory,evaluation_cohort=scope)
    class ForwardBoundary(Exception):pass
    def stop(**kwargs):raise ForwardBoundary()
    monkeypatch.setattr(evaluator,'_entry_representations',stop)
    with pytest.raises(ForwardBoundary):evaluator.evaluate_bound_full_val_v1(**args)
    factory.source_split='val'
    with pytest.raises(RuntimeError,match='COHORT_SOURCE_MISMATCH'):evaluator.evaluate_bound_full_val_v1(**args)
    factory.source_split='train';args['frame'].iloc[0,args['frame'].columns.get_loc('parent_entry_row_index')]+=1
    with pytest.raises(RuntimeError,match='BOUND_FRAME_MISMATCH'):evaluator.evaluate_bound_full_val_v1(**args)
