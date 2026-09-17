"""Synthetic binding/initialization tests for the existing native component owner."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.scripts import run_unified_exit_random_access_full_train_v1 as runner
from gx1.contracts.local_random_access_campaign_v2 import file_sha256
from gx1.contracts.unified_exit_pilot_final_bindings_v1 import build_composite_normalization_binding
from gx1.contracts.unified_exit_reference_policy_v1 import reference_policy_contract
from tests.test_unified_exit_pilot_final_bindings_v1 import _base, _summary


def _bind(path):return {'path':str(path),'sha256':file_sha256(path)}

def _write(path,data):
    path.write_text(json.dumps(data));return _bind(path)


def _bindings(tmp_path):
    files={name:tmp_path/name for name in runner._DATA_FILES}
    for path in files.values():path.write_text('{}')
    files['entry_val_parquet']=files['entry_train_parquet']
    files['entry_val_manifest']=files['entry_train_manifest']
    cutoff=int(pd.Timestamp('2026-03-01T00:00Z').value)
    design={'schema_version':'gx1_frozen_chronological_learning_design_v1',
            'status':'DESIGN_AND_CONTROL_IDS_FROZEN_NOT_EXECUTABLE',
            'scope':{'test_sealed':True,'same_architecture':True},
            'initialization':{'mode':'fresh_existing_model_constructor_no_checkpoint_weights','seed':20260911},
            'budget':{'planned_optimizer_steps':256,'maximum_trained_entry_rows':4096},
            'calendar':{'train_control_cutoff':'2026-03-01T00:00Z','bindings':{'CONTROL256_PARENT_ROWS':{'path':'fixture-control','sha256':'8'*64}}},
            'targets':{'reference_policy':reference_policy_contract()}}
    db=_write(tmp_path/'design.json',design)
    populations={
        'TRAIN_ELIGIBLE_PARENT_ROWS':np.arange(4500,dtype='int64'),
        'TRAIN_NATIVE_EPOCH0_ORDER':np.arange(4500,dtype='int64')[::-1],
        'TRAIN_NATIVE4096_PARENT_ROWS':np.arange(4500,dtype='int64')[::-1][:4096],
        'TRAIN256_PROBE_PARENT_ROWS':np.arange(4500,dtype='int64')[::-1][:256],
    }
    rb={}
    for key,values in populations.items():
        path=tmp_path/(key+'.npy');np.save(path,values);rb[key]=_bind(path)
    prep={'frozen_design':db,'support_end_inclusive':'2026-03-01T00:00Z','bindings':rb}
    pb=_write(tmp_path/'preparation.json',prep)
    composite=build_composite_normalization_binding(base_artifact=_base(),base_path='/fixture/base.json',
        base_file_sha256='1'*64,summary_normalization=_summary(),summary_manifest_path='/fixture/summary.json',
        summary_manifest_file_sha256='2'*64,summary_manifest_sha256='3'*64)
    cb=_write(files['child_composite_normalization'],composite)
    norm={'schema_version':'gx1_prefix_normalization_preparation_result_v1',
          'decision':'PREFIX_NORMALIZATION_READY_NOT_NATIVE_BOUND','frozen_design':db,
          'scope':{'control_fit_rows':0,'test_fit_rows':0,'test_accessed':False},
          'original_market_successor_counts_exact_equal':True,'parent_child_row_clock_identity_exact':True,
          'prefix_preparation':pb,'fit_cutoff_time_ns':cutoff,'fit_entry_rows':rb['TRAIN_ELIGIBLE_PARENT_ROWS'],
          'artifacts':{'COMPOSITE_NORMALIZATION.json':cb},
          'composite_normalization_sha256':composite['composite_normalization_sha256'],
          'base_contract_sha256':composite['base_feature_normalization']['contract_sha256'],
          'summary_normalization_sha256':composite['lifetime_summary_normalization']['normalization_sha256']}
    labels={'schema_version':'gx1_prefix_policy_dependent_labels_v1','prefix_preparation':pb,
            'parent_entry_parquet':_bind(files['entry_train_parquet']),
            'parent_entry_manifest':_bind(files['entry_train_manifest']),
            'labels':{'TRAIN':{'row_binding':rb['TRAIN_ELIGIBLE_PARENT_ROWS']},
                      'CONTROL256':{'row_binding':design['calendar']['bindings']['CONTROL256_PARENT_ROWS']}}}
    value={'design':db,'normalization_result':_write(tmp_path/'norm-result.json',norm),
           'labels_result':_write(tmp_path/'labels-result.json',labels)}
    return value,files,design,norm,labels,prep,composite


def _check(value,files,**overrides):
    args=dict(files=files,seed=20260911,batch_size=16,learning_rate=.0001,weight_decay=.0001)
    args.update(overrides);return runner._require_prefix_component_bindings(value,**args)


def test_completed_artifacts_join_on_design_cutoff_population_and_transform(tmp_path):
    value,files,design,norm,labels,prep,composite=_bindings(tmp_path)
    checked=_check(value,files)
    assert checked['normalization']==norm and checked['labels']==labels
    assert checked['cutoff_time_ns']==norm['fit_cutoff_time_ns']
    assert checked['eligible_parent_rows'].tolist()==list(range(4500))
    assert checked['epoch0_parent_order'].tolist()==list(range(4500))[::-1]


@pytest.mark.parametrize('fault',['control_fit','test_fit','changed_cutoff','wrong_design','wrong_rows',
                                  'old_base_transform','old_summary_transform','old_label_policy',
                                  'june_source','reordered_plan','changed_bytes','seed','learning_rate'])
def test_fresh_components_reject_exposed_mixed_or_reselected_inputs(tmp_path,fault):
    value,files,design,norm,labels,prep,composite=_bindings(tmp_path)
    overrides={}
    if fault in ('control_fit','test_fit'):norm['scope']['control_fit_rows' if fault=='control_fit' else 'test_fit_rows']=1
    elif fault=='changed_cutoff':norm['fit_cutoff_time_ns']+=1
    elif fault=='wrong_design':norm['frozen_design']={**value['design'],'sha256':'0'*64}
    elif fault=='wrong_rows':norm['fit_entry_rows']={**norm['fit_entry_rows'],'sha256':'0'*64}
    elif fault=='old_base_transform':norm['base_contract_sha256']='0'*64
    elif fault=='old_summary_transform':norm['summary_normalization_sha256']='0'*64
    elif fault=='old_label_policy':labels['prefix_preparation']={**labels['prefix_preparation'],'sha256':'0'*64}
    elif fault=='june_source':files['entry_val_parquet']=tmp_path/'june.parquet';files['entry_val_parquet'].write_bytes(b'old-val')
    elif fault=='reordered_plan':
        key='TRAIN_NATIVE4096_PARENT_ROWS';path=Path(prep['bindings'][key]['path']);rows=np.load(path);np.save(path,rows[::-1])
        prep['bindings'][key]=_bind(path)
        pb=_write(Path(norm['prefix_preparation']['path']),prep);norm['prefix_preparation']=pb;labels['prefix_preparation']=pb
    elif fault=='changed_bytes':Path(prep['bindings']['TRAIN_ELIGIBLE_PARENT_ROWS']['path']).write_bytes(b'replaced')
    elif fault=='seed':overrides['seed']=99
    elif fault=='learning_rate':overrides['learning_rate']=.001
    value['normalization_result']=_write(Path(value['normalization_result']['path']),norm)
    value['labels_result']=_write(Path(value['labels_result']['path']),labels)
    with pytest.raises(RuntimeError):_check(value,files,**overrides)


class TinyModel(torch.nn.Module):
    def __init__(self,normalization):
        super().__init__();self.backbone=torch.nn.Linear(3,3);self.head=torch.nn.Linear(3,2)
        self.task_log_variances=torch.nn.ParameterDict({'entry':torch.nn.Parameter(torch.zeros(())),
                                                     'exit':torch.nn.Parameter(torch.zeros(()))})
        self.register_buffer('normalization_identity',torch.tensor(list(bytes.fromhex(normalization['contract_sha256'])),dtype=torch.uint8))
    def forward(self,*args,**kwargs):raise AssertionError('No forward belongs in component construction')


def test_fresh_model_is_seeded_whole_constructor_and_neutral_tasks(tmp_path,monkeypatch):
    _,_,_,_,_,_,composite=_bindings(tmp_path)
    norm=composite['base_feature_normalization']['artifact']['contract'];calls=[]
    def construct(meta,normalization,device):calls.append((meta,normalization,device));return TinyModel(normalization)
    monkeypatch.setattr(runner.val,'_model',construct)
    def forbidden(*a,**kw):raise AssertionError('Old checkpoint or normalization loaded')
    monkeypatch.setattr(runner.val,'load_selected_weight_ema_checkpoint_readonly_v1',forbidden)
    monkeypatch.setattr(runner.val,'bind_preserved_v7_input_normalization',forbidden)
    kwargs=dict(metadata={'unchanged':'architecture'},normalization=norm,device=torch.device('cpu'),seed=20260911)
    model,binding=runner._fresh_prefix_model(**kwargs)
    torch.rand(17)
    again,same=runner._fresh_prefix_model(**kwargs)
    assert binding==same and binding['checkpoint_loaded'] is False
    assert all(torch.equal(v,again.state_dict()[k]) for k,v in model.state_dict().items())
    other,_=runner._fresh_prefix_model(**{**kwargs,'seed':20260912})
    assert not torch.equal(model.backbone.weight,other.backbone.weight)
    assert all(torch.count_nonzero(p)==0 for p in model.task_log_variances.parameters())
    assert all(c[0]==kwargs['metadata'] and c[1]==norm for c in calls)
    def biased(*args):
        m=TinyModel(norm);m.task_log_variances['entry'].data.fill_(1);return m
    monkeypatch.setattr(runner.val,'_model',biased)
    with pytest.raises(RuntimeError,match='FRESH_TASK_WEIGHTS_INVALID'):runner._fresh_prefix_model(**kwargs)


@pytest.fixture
def component_chain(tmp_path,monkeypatch):
    value,files,design,norm,labels,prep,composite=_bindings(tmp_path)
    prefix=_check(value,files)
    monkeypatch.setattr(runner,'_require_prefix_component_bindings',lambda *a,**kw:prefix)
    seen={};population=5000
    frame=pd.DataFrame({'entry_row_index':np.arange(population),'parent_entry_row_index':np.arange(population),
                        'lifecycle_state_count':np.full(population,8)})
    ip=tmp_path/'index.parquet';frame.to_parquet(ip,index=False)
    mp=tmp_path/'index.json';mp.write_text('{}')
    root={'schema_version':'fixture','root_sha256':'1'*64,'splits':{'train':{
          'index_parquet_path':str(ip),'index_parquet_sha256':file_sha256(ip),
          'manifest_path':str(mp),'manifest_sha256':'2'*64}}}
    meta={'seq_len':96,'multi_tf':{tf+'_seq_len':7 for tf in ['m5','m15','h1','h4','d1']}}
    raw_read=runner.val._read
    def read(path):
        if path==files['entry_train_manifest']:return {'splits':{'train':{'start':'2021-06-01T00:00Z','end':'2026-06-01T00:00Z'}}}
        if path==files['source_bundle_metadata']:return meta
        if path==files['random_access_root']:return root
        return raw_read(path)
    monkeypatch.setattr(runner.val,'_read',read)
    monkeypatch.setattr(runner.val,'_bind_multi_tf_cache_from_source_bundle_metadata',lambda m:seen.setdefault('meta',m))
    class Dataset:
        def __init__(self,path,**kw):
            seen.setdefault('dataset_constructors',[]).append((path,kw));self.role=None;self.storage=object();self.df=pd.DataFrame({'feature':[1.,2.]})
        def __len__(self):return population
        def bind_policy_dependent_auxiliary_targets(self,**kw):
            assert self.role is None;self.role=kw['role'];self.label_binding=kw
        def bind_unified_exit_lifecycle_v2(self,adapter):self._unified_exit_lifecycle_v2=adapter
        def bind_random_access_entry_coordinate_mapping_v1(self,**kw):
            self._random_access_child_index_by_parent=dict(zip(kw['parent_entry_row_indices'],kw['child_entry_row_indices']))
        def _get_exit_multi_tf_episode_histories(self,*args):raise AssertionError('No model input materialization')
    monkeypatch.setattr(runner.val,'EntryV10CtxDataset',Dataset)
    class Corpus:
        def __init__(self,**kw):
            assert kw['splits']==('train',) and set(kw['entry_parquets'])=={'train'}
            seen['corpus']=kw;self.splits={'train':object()};self.evidence={'root_manifest_sha256':'3'*64}
    monkeypatch.setattr(runner.val,'UnifiedExitLifecycleCorpus',Corpus)
    monkeypatch.setattr(runner.val,'require_random_access_index_root',lambda r:r)
    class Adapter:
        _random_access_train={};_native_random_access_index=frame
        def set_full_population_epoch_index(self,epoch):
            assert epoch==0
            return {'every_entry_pair_exactly_once':True,'entry_pair_count':population,'epoch_index':epoch}
        def random_access_selected_entry_rows_v1(self):return tuple(range(population))[::-1]
    def factory(**kw):seen['adapter_binding']=kw;return lambda budget:Adapter()
    monkeypatch.setattr(runner,'build_random_access_train_adapter_factory_v1',factory)
    def manifest(data,**kw):assert kw['expected_split']=='train';return {'manifest_sha256':'2'*64}
    monkeypatch.setattr(runner.val,'require_random_access_index_manifest',manifest)
    from gx1.contracts import unified_exit_bounded_val_cohort_v1 as cohort_owner
    cohort={'entry_row_indices':list(range(4700,4956)),'parent_entry_row_indices':list(range(4700,4956)),
            'source_split':'train','population_rows':population,'source_index':_bind(ip)}
    monkeypatch.setattr(cohort_owner,'build_chronological_control_cohort',lambda *a,**kw:cohort)
    monkeypatch.setattr(runner.val,'require_parent_entry_coordinate_equivalence',lambda **kw:seen.setdefault('parent_equivalence',kw))
    monkeypatch.setattr(runner.val,'_load_val_economics_readiness',lambda path:{'economics_objective_contract':{}})
    monkeypatch.setattr(runner.val,'_build_provider',lambda **kw:SimpleNamespace(economic_exit_step_manifest={}))
    monkeypatch.setattr(runner.val,'_source_path',lambda manifest,name:tmp_path/name)
    def val_factory(**kw):seen['control_factory']=kw;return object()
    monkeypatch.setattr(runner.val.RandomAccessValStateFactoryV1,'from_artifacts',val_factory)
    monkeypatch.setattr(runner.val,'_model',lambda meta,normalization,device:TinyModel(normalization))
    def forbidden(*a,**kw):raise AssertionError('Legacy weight, normalization, June audit or full-VAL binding reached')
    for name in ['load_selected_weight_ema_checkpoint_readonly_v1','bind_preserved_v7_input_normalization',
                 '_load_final_authority','require_launch_manifest','_val_sequence_source_audit']:
        monkeypatch.setattr(runner.val,name,forbidden)
    def bind_control(context):
        assert context['evaluation_cohort']==cohort
        seen['control_context']=context
        return {'synthetic_bounded_control':True}
    monkeypatch.setattr(runner.trainer,'_native_candidate_val_context_binding',bind_control)
    args=dict(files=files,dataset_run_id='fixture',seed_launch_path=None,seed_authority_path=None,
              seed_authority_file_sha256=None,device=torch.device('cpu'),batch_size=16,epochs=30,
              seed=20260911,learning_rate=.0001,weight_decay=.0001,
              val_limits={'max_state_views':1000000,'max_model_forwards':1000000,'max_wall_seconds':10800,
                          'progress_interval_forwards':64,'policy_batch_size':256,'cpu_pipeline_workers':8},
              exit_reference_policy=design['targets']['reference_policy'],chronological_prefix=value)
    return args,seen,prefix,composite


def test_existing_component_owner_uses_fresh_state_labels_and_physical_train(component_chain):
    args,seen,prefix,composite=component_chain
    c=runner._build_bound_full_train_components(**args)
    train,control=c['train_ds'],c['val_ds']
    assert train is not control and train.storage is control.storage
    assert train.role=='TRAIN' and control.role=='CONTROL256'
    probe=c['train_probe_ds']
    assert probe is not train and probe is not control
    assert probe.storage is train.storage and probe.df is train.df
    assert probe.role=='TRAIN' and probe.label_binding==train.label_binding
    assert not hasattr(probe,'_unified_exit_lifecycle_v2')
    assert not hasattr(probe,'_random_access_child_index_by_parent')
    assert len(seen['dataset_constructors'])==1
    assert not hasattr(control,'_unified_exit_lifecycle_v2')
    assert seen['control_factory']['source_split']=='train'
    assert seen['adapter_binding']['reference_cutoff_time_ns']==prefix['cutoff_time_ns']
    assert seen['adapter_binding']['reference_policy']==args['exit_reference_policy']
    assert c['input_normalization']==composite['base_feature_normalization']['artifact']['contract']
    assert c['seed_binding']['checkpoint_loaded'] is False
    assert c['effective_train_rows']==4500
    assert c['prefix_epoch0_parent_order'].tolist()==list(range(4500))[::-1]
    assert c['optimizer'].state_dict()['state']=={}
    assert [group['weight_decay'] for group in c['optimizer'].param_groups]==[.0001,0.]
    assert all(group['lr']==.0001 for group in c['optimizer'].param_groups)
    joint={id(p) for p in c['model'].task_log_variances.parameters()}
    assert {id(p) for p in c['optimizer'].param_groups[1]['params']}==joint
    assert c['weight_ema'].steps==0
    assert all(torch.equal(v,c['weight_ema']._shadow[k]) for k,v in c['model'].state_dict().items())
    expected=runner.trainer.resolve_weight_ema_decay(runner.trainer.ENTRY_TRAIN_WEIGHT_EMA_DECAY_DECLARED,
             train_rows=4500,batch_size=16,grad_accum_steps=1)
    assert c['weight_ema_derivation']==expected
    assert len(c['native_val_context']['frame'])==256
    assert seen['control_context'] is c['native_val_context']
    assert c['native_val_context']['val_sequence_audit']==args['files']['sequence_source_audit']
    if c['lr_scheduler'] is not None:assert c['lr_scheduler'].T_max==30


@pytest.mark.parametrize('fault',['old_weights','old_policy','changed_order'])
def test_existing_component_owner_rejects_old_initialization_or_changed_training_order(component_chain,fault):
    args,seen,prefix,_=component_chain
    if fault=='old_weights':args['seed_authority_path']=Path('/old/seed.json')
    elif fault=='old_policy':args['exit_reference_policy']=None
    else:prefix['epoch0_parent_order']=prefix['epoch0_parent_order'][::-1].copy()
    with pytest.raises(RuntimeError,match='NATIVE_PREFIX_'):runner._build_bound_full_train_components(**args)
