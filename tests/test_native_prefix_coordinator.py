"""Synthetic native session integration, not GPU or trading-quality evidence."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.contracts.entry_candidate_checkpoint_policy_v1 import MARKED_NET_CHECKPOINT_MONITOR
from gx1.contracts.unified_exit_bounded_val_cohort_v1 import build_chronological_control_cohort
from gx1.contracts.unified_exit_random_access_val_factory_v1 import RandomAccessValStateFactoryV1
from gx1.contracts.unified_exit_random_access_model_v1 import RANDOM_ACCESS_MODEL_SCHEMA_VERSION,RANDOM_ACCESS_MODEL_SCHEMA_SHA256
from tests.test_native_fresh_prefix_components import _bindings,_bind,_write
from tests.test_native_chronological_control_source import _control_plan
from tests.test_candidate_execution_staging import Harness,Rows,equal_tree,digest
from tests.test_candidate_training_session import _recipe_source_provenance


def _prepared(tmp_path):
    value,files,design,norm,labels,prep,composite=_bindings(tmp_path)
    directory=tmp_path/'control';directory.mkdir()
    control_binding,frame,_,_,control_design=_control_plan(directory,population=5000)
    frame['parent_entry_row_index']=frame.entry_row_index
    frame.loc[:4499,'entry_time_ns']=pd.date_range('2025-06-01T00:00Z',periods=4500,freq='5min').asi8
    frame['first_state_time_ns']=frame.entry_time_ns+300_000_000_000
    ip=Path(control_design['calendar']['index']['path']);frame.to_parquet(ip,index=False)
    control_rows=np.arange(4700,4956,dtype='int64')
    rp=Path(control_design['calendar']['bindings']['CONTROL256_PARENT_ROWS']['path']);np.save(rp,control_rows)
    control_design['calendar']['index']=_bind(ip)
    control_design['calendar']['bindings']['CONTROL256_PARENT_ROWS']=_bind(rp)
    design['scope'].update(control_design['scope']);design['calendar']=control_design['calendar']
    design['selection']=control_design['selection'];design['budget'].update(control_design['budget'])
    value['design']=_write(Path(value['design']['path']),design)
    prep['frozen_design']=value['design'];pb=_write(Path(norm['prefix_preparation']['path']),prep)
    norm['prefix_preparation']=pb;norm['frozen_design']=value['design']
    labels['prefix_preparation']=pb;labels['labels']['CONTROL256']['row_binding']=_bind(rp)
    value['normalization_result']=_write(Path(value['normalization_result']['path']),norm)
    value['labels_result']=_write(Path(value['labels_result']['path']),labels)
    cohort=build_chronological_control_cohort(value['design'])
    class Adapter:
        _random_access_train={'reference_cutoff_time_ns':norm['fit_cutoff_time_ns'],
                              'reference_policy':design['targets']['reference_policy'],
                              'normalization_artifact':composite['lifetime_summary_normalization']}
        _native_random_access_index=frame
        def set_full_population_epoch_index(self,epoch):
            assert epoch==0
            return {'every_entry_pair_exactly_once':True,'entry_pair_count':5000,'epoch_index':epoch}
        def random_access_selected_entry_rows_v1(self):return tuple(range(5000))[::-1]
    class Dataset(Rows):
        def __init__(self,role):
            super().__init__(5000);self.parquet_path=files['entry_train_parquet']
            self._policy_dependent_auxiliary_binding={'result':value['labels_result'],'design':value['design'],
                'prefix_preparation':pb,'role':role,'labels':labels['labels'][role]}
            bound=np.arange(4500,dtype='int64') if role=='TRAIN' else control_rows
            self._policy_dependent_auxiliary_bound_rows=np.zeros(5000,dtype=bool)
            self._policy_dependent_auxiliary_bound_rows[bound]=True
            if role=='TRAIN':
                self._unified_exit_lifecycle_v2=Adapter()
                self._random_access_child_index_by_parent=dict(zip(range(5000),range(5000)))
        def __getitem__(self,index):
            assert self._policy_dependent_auxiliary_bound_rows[index]
            return super().__getitem__(index)
    factory=RandomAccessValStateFactoryV1.__new__(RandomAccessValStateFactoryV1)
    factory.source_split='train'
    factory.artifact_file_sha256={'random_access_index':_bind(ip)['sha256']}
    factory.factory_receipt={'split':'train','entry_pair_count':5000,
                             'composite_normalization_sha256':norm['composite_normalization_sha256']}
    context={'frame':frame.iloc[cohort['entry_row_indices']].copy(),'state_factory':factory,
             'parent_coordinate_evidence':{'synthetic':True},'val_sequence_audit':files['sequence_source_audit'],
             'max_model_forwards':100000,'max_state_views':100000,'max_wall_seconds':10800,
             'progress_interval_forwards':64,'policy_batch_size':256,'cpu_pipeline_workers':8,
             'evaluation_cohort':cohort}
    return SimpleNamespace(value=value,files=files,design=design,norm=norm,prep=prep,composite=composite,
                           train=Dataset('TRAIN'),control=Dataset('CONTROL256'),context=context)


def _binding_args(data):
    torch.manual_seed(20260911)
    return dict(chronological_prefix=data.value,train_ds=data.train,val_ds=data.control,
                train_parquet=data.files['entry_train_parquet'],val_parquet=data.files['entry_train_parquet'],
                input_normalization=data.composite['base_feature_normalization']['artifact']['contract'],
                model=torch.nn.Linear(3,2),seed=20260911,batch_size=16,learning_rate=.0001,weight_decay=.0001)


def test_prefix_binding_covers_exact_roles_rows_cutoff_normalization_and_fresh_identity(tmp_path):
    data=_prepared(tmp_path);args=_binding_args(data)
    binding,parents,order=trainer._prefix_candidate_training_binding(**args)
    assert parents.tolist()==list(range(4500)) and order.tolist()==list(range(4500))[::-1]
    assert binding['reference_cutoff_time_ns']==data.norm['fit_cutoff_time_ns']
    assert binding['maximum_optimizer_steps']==256 and binding['maximum_completed_epochs']==0
    assert binding['initial_model_state_sha256']==trainer._model_state_sha256(args['model'])
    assert binding["model_functions"] == trainer._PREFIX_MODEL_FUNCTIONS
    context=trainer._native_candidate_val_context_binding(data.context)
    assert context['report_only'] is True and context['factory_receipt']['split']=='train'
    assert context['schema_version']=='gx1_candidate_native_bounded_control_context_v1'
    assert context['evaluation_cohort']['plan']==data.value['design']
    with pytest.raises(RuntimeError,match='BOUNDED_CONTROL_NOT_EPOCH_VALIDATION'):
        trainer._native_candidate_epoch_validation(session=None,model=None,target_model=None,weight_ema=None,
            val_ds=None,device=torch.device('cpu'),batch_size=16,epoch_index=0,context=data.context)


@pytest.mark.parametrize('fault',['cutoff','policy','summary','base','role','mask','order','source','old_labels'])
def test_prefix_binding_rejects_changed_supervision_or_population(tmp_path,fault):
    data=_prepared(tmp_path);args=_binding_args(data)
    if fault=='cutoff':data.train._unified_exit_lifecycle_v2._random_access_train['reference_cutoff_time_ns']+=1
    elif fault=='policy':data.train._unified_exit_lifecycle_v2._random_access_train['reference_policy']=None
    elif fault=='summary':data.train._unified_exit_lifecycle_v2._random_access_train['normalization_artifact']={}
    elif fault=='base':args['input_normalization']={'contract_sha256':'0'*64}
    elif fault=='role':data.control._policy_dependent_auxiliary_binding['role']='TRAIN'
    elif fault=='mask':data.train._policy_dependent_auxiliary_bound_rows[4700]=True
    elif fault=='order':Path(data.prep['bindings']['TRAIN_NATIVE_EPOCH0_ORDER']['path']).write_bytes(b'changed')
    elif fault=='source':args['val_parquet']=Path('/old/june.parquet')
    elif fault=='old_labels':data.train._policy_dependent_auxiliary_binding['result']={'path':'old','sha256':'0'*64}
    with pytest.raises(RuntimeError):trainer._prefix_candidate_training_binding(**args)


@pytest.mark.parametrize('fault',['source','parent','population'])
def test_bounded_context_rejects_mismatched_factory_or_coordinates(tmp_path,fault):
    data=_prepared(tmp_path)
    if fault=='source':data.context['state_factory'].source_split='val'
    elif fault=='parent':data.context['frame'].iloc[0,data.context['frame'].columns.get_loc('parent_entry_row_index')]+=1
    else:data.context['state_factory'].factory_receipt['entry_pair_count']=5001
    with pytest.raises(RuntimeError,match='CONTROL_SOURCE_INVALID'):trainer._native_candidate_val_context_binding(data.context)


class PrefixHarness(Harness):
    def __init__(self,root,data):
        super().__init__(root);self.data=data;self.last_kwargs=None;self.teacher_hashes=[]
        self.ns["_copy_frozen_prefix_reference_model"] = copy.deepcopy
    def train(self,model,teacher,loader,optimizer,device,**kwargs):
        self.teacher_hashes.append(trainer._model_state_sha256(teacher))
        return super().train(model,teacher,loader,optimizer,device,**kwargs)
    def run_prefix(self,output,steps,expected_pointer=None,**overrides):
        d=self.data;torch.manual_seed(20260911)
        model=torch.nn.Linear(3,2)
        model.unified_exit_random_access_architecture_version=RANDOM_ACCESS_MODEL_SCHEMA_VERSION
        model.register_buffer('unified_exit_random_access_architecture_sha256',torch.tensor(list(bytes.fromhex(RANDOM_ACCESS_MODEL_SCHEMA_SHA256)),dtype=torch.uint8))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
        ema=trainer._WeightEma(model,.5)
        budget={'stop_after_optimizer_steps':steps,'stop_after_completed_val_epochs':None,
                'max_invocation_seconds':12000,'expected_active_pointer_sha256':expected_pointer}
        kwargs=dict(model=model,optimizer=optimizer,weight_ema=ema,
            lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30),device=torch.device('cpu'),
            train_ds=d.train,val_ds=d.control,effective_train_rows=4500,batch_size=16,num_workers=0,
            pin_memory=False,persistent_workers=False,prefetch_factor=None,epochs=30,
            early_stopping_patience=5,early_stopping_min_delta=0.,minimum_epochs_before_stop=1,save_top_k=1,
            out_bundle_dir=output,gx1_data_override='',run_id='V46_20260825T170935Z_CANDIDATE',dataset_run_id='V46_20260825T170935Z',
            train_parquet=d.files['entry_train_parquet'],val_parquet=d.files['entry_train_parquet'],
            m5_prebuilt_path=d.files['m5_prebuilt'],unified_exit_lifecycle_manifest_path=d.files['random_access_root'],
            input_normalization=d.composite['base_feature_normalization']['artifact']['contract'],
            seed=20260911,grad_accum_steps=1,grad_clip_norm=1.,weight_decay=.0001,lr=.0001,dropout=0.,seq_len=96,
            per_tf_seq_lens={'M5':16,'M15':64,'H1':96,'H4':96,'D1':252},multi_tf_num_layers=2,specialist_num_layers=1,
            multi_tf_scale=.5,specialist_fusion_scale=.25,cross_family_fusion_scale=.25,
            unified_exit_lifecycle_evidence={'splits':{'train':{'lifecycle_manifest_sha256':'a'*64}},'root_manifest_sha256':'b'*64},
            recipe_source_provenance=_recipe_source_provenance(source_commit='1'*40),precision_policy=trainer.DETERMINISTIC_FP32,
            execution_budget=budget,execution_budget_sha256='e'*64,invocation_started_monotonic=self.clock,
            checkpoint_monitor=MARKED_NET_CHECKPOINT_MONITOR,native_val_context=d.context,chronological_prefix=d.value)
        kwargs.update(overrides);self.last_kwargs=kwargs
        with pytest.raises(self.ns['_CandidateExecutionPaused']) as pause:self.ns['_run_resumable_candidate_training'](**kwargs)
        return model,pause.value.evidence


def test_native_prefix_resume_preserves_teacher_optimizer_rng_and_original_order(tmp_path):
    artifacts=tmp_path/'artifacts';artifacts.mkdir();data=_prepared(artifacts)
    h=PrefixHarness(tmp_path/'run',data)
    direct=h.root/'DIRECT';split=h.root/'SPLIT'
    _,pause=h.run_prefix(direct,4);expected=h.state(direct);batches=list(h.batches)
    h.batches.clear()
    _,first=h.run_prefix(split,2)
    _,last=h.run_prefix(split,4,expected_pointer=digest(h.pointer(split)))
    actual=h.state(split)
    assert first['global_optimizer_steps']==2 and last['global_optimizer_steps']==4
    assert last['epoch_index']==0 and last['completed_val_epochs']==0 and last['phase']=='train'
    for key in ['model_state','target_model_state','optimizer_state','weight_ema_state','lr_scheduler_state',
                'rng_state','epoch_order','training_progress']:equal_tree(expected[key],actual[key])
    assert h.batches==batches and [i for batch in batches for i in batch]==list(range(4500))[::-1][:64]
    assert h.validation_batches==0 and len(set(h.teacher_hashes))==1
    contract=json.loads((h.pointer(split).parent/trainer._CANDIDATE_TRAINING_CONTRACT_FILENAME).read_text())
    assert contract['training']['reference_cutoff_time_ns']==data.norm['fit_cutoff_time_ns']
    assert contract['chronological_prefix']['artifacts']==data.value
    assert contract['chronological_prefix']['initial_model_state_sha256']==h.teacher_hashes[0]
    assert 'native_full_val' not in contract and contract['native_bounded_control']['report_only'] is True
    # An unbound warm start cannot masquerade as resume of the fresh session.
    kwargs=dict(h.last_kwargs);kwargs['model']=copy.deepcopy(kwargs['model'])
    with torch.no_grad():kwargs['model'].weight.add_(1.)
    kwargs['execution_budget']={**kwargs['execution_budget'],'expected_active_pointer_sha256':digest(h.pointer(split))}
    before=h.pointer(split).read_bytes()
    with pytest.raises(RuntimeError,match='CONTRACT'):h.ns['_run_resumable_candidate_training'](**kwargs)
    assert h.pointer(split).read_bytes()==before


@pytest.mark.parametrize('fault',['extension','unlimited','old_origin','full_val','legacy_probe'])
def test_prefix_fixed_scope_rejects_expansion_before_checkpoint_or_updates(tmp_path,fault):
    artifacts=tmp_path/'artifacts';artifacts.mkdir();data=_prepared(artifacts)
    h=PrefixHarness(tmp_path/'run',data);output=h.root/'CANDIDATE'
    budget={'stop_after_optimizer_steps':256,'stop_after_completed_val_epochs':None,
            'max_invocation_seconds':12000,'expected_active_pointer_sha256':None}
    overrides={}
    if fault=='extension':budget['stop_after_optimizer_steps']=257
    elif fault=='unlimited':budget['stop_after_optimizer_steps']=None
    elif fault=='old_origin':overrides['candidate_resume_origin']={'schema_version':'old'}
    elif fault=='full_val':budget['stop_after_completed_val_epochs']=1
    elif fault=='legacy_probe':budget['resume_probe_val_rows']=32
    with pytest.raises(RuntimeError,match='PREFIX_FIXED_BUDGET_REQUIRED'):
        h.run_prefix(output,256,execution_budget=budget,**overrides)
    assert not h.pointer(output).exists() and h.batches==[]
