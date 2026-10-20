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
    def __init__(self,normalization,entry_training_phase=runner.trainer.JOINT_FITTED_Q_TRAINING_PHASE,
                 dynamic_m5_query_minutes=None, entry_execution_profile=None):
        from gx1.contracts.entry_model_native_joint_task_weighting_v1 import joint_task_names_for_phase
        super().__init__();self.backbone=torch.nn.Linear(3,3);self.head=torch.nn.Linear(3,2)
        self.entry_training_phase=entry_training_phase
        self.task_log_variances=torch.nn.ParameterDict({name:torch.nn.Parameter(torch.zeros(()))
            for name in joint_task_names_for_phase(entry_training_phase)})
        self.cfg=SimpleNamespace(entry_training_phase=entry_training_phase,dynamic_m5_query_minutes=dynamic_m5_query_minutes)
        if entry_execution_profile is not None:
            self.entry_execution_profile = dict(entry_execution_profile)
            self.cfg.entry_execution_profile = dict(entry_execution_profile)
        if dynamic_m5_query_minutes is not None:
            self.register_buffer('dynamic_m5_query_horizons',torch.tensor(dynamic_m5_query_minutes,dtype=torch.int64))
        self.register_buffer('normalization_identity',torch.tensor(list(bytes.fromhex(normalization['contract_sha256'])),dtype=torch.uint8))
    def forward(self,*args,**kwargs):raise AssertionError('No forward belongs in component construction')


def test_fresh_model_is_seeded_whole_constructor_and_neutral_tasks(tmp_path,monkeypatch):
    _,_,_,_,_,_,composite=_bindings(tmp_path)
    norm=composite['base_feature_normalization']['artifact']['contract'];calls=[]
    def construct(meta,normalization,device,**model_kwargs):
        calls.append((meta,normalization,device,model_kwargs));return TinyModel(normalization,**model_kwargs)
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
    def biased(*args,**model_kwargs):
        m=TinyModel(norm,**model_kwargs);next(iter(m.task_log_variances.values())).data.fill_(1);return m
    monkeypatch.setattr(runner.val,'_model',biased)
    with pytest.raises(RuntimeError,match='FRESH_TASK_WEIGHTS_INVALID'):runner._fresh_prefix_model(**kwargs)


@pytest.fixture
def component_chain(tmp_path,monkeypatch):
    return _component_chain(tmp_path, monkeypatch)


def _component_chain(tmp_path, monkeypatch, physical=None):
    if physical is None:
        value,files,design,norm,labels,prep,composite=_bindings(tmp_path)
        population, val_population = 5000, 5000
    else:
        value,files,design = physical["value"],dict(physical["files"]),physical["design"]
        for name in set(runner._DATA_FILES) - set(files):
            files[name] = tmp_path/(name+".json");files[name].write_text("{}")
        composite=json.loads(files["child_composite_normalization"].read_text())
        population=physical["sources"]["train"]["physical_rows"]
        val_population=physical["sources"]["val"]["physical_rows"]
    prefix=_check(value,files)
    monkeypatch.setattr(runner,'_require_prefix_component_bindings',lambda *a,**kw:prefix)
    seen={}

    def index_frame(count):
        return pd.DataFrame({'entry_row_index':np.arange(count),'parent_entry_row_index':np.arange(count),
                             'lifecycle_state_count':np.full(count,8)})
    frame=index_frame(population)
    ip=tmp_path/'index.parquet';frame.to_parquet(ip,index=False)
    mp=tmp_path/'index.json';mp.write_text('{}')
    root={'schema_version':'fixture','root_sha256':'1'*64,'splits':{'train':{
          'index_parquet_path':str(ip),'index_parquet_sha256':file_sha256(ip),
          'manifest_path':str(mp),'manifest_sha256':'2'*64}}}
    source_split="train" if physical is None else "val"
    if physical is not None:
        vp=tmp_path/'val-index.parquet';index_frame(val_population).to_parquet(vp,index=False)
        vm=tmp_path/'val-index.json';vm.write_text("{}")
        root["splits"]["val"]={"index_parquet_path":str(vp),"index_parquet_sha256":file_sha256(vp),
                              "manifest_path":str(vm),"manifest_sha256":"4"*64}
        meta=json.loads(files["source_bundle_metadata"].read_text())
    else:
        vp,vm=ip,mp
        meta={'seq_len':96,'multi_tf':{tf+'_seq_len':7 for tf in ['m5','m15','h1','h4','d1']}}
    raw_read=runner.val._read
    def read(path):
        if physical is not None:
            for split in ("train","val"):
                if path==files[f"entry_{split}_manifest"]:
                    return {"splits":{split:physical["sources"][split]["declared_window"]}}
        elif path==files['entry_train_manifest']:
            return {'splits':{'train':{'start':'2021-06-01T00:00Z','end':'2026-06-01T00:00Z'}}}
        if path==files['source_bundle_metadata']:return meta
        if path==files['random_access_root']:return root
        return raw_read(path)
    monkeypatch.setattr(runner.val,'_read',read)
    monkeypatch.setattr(runner.val,'_bind_multi_tf_cache_from_source_bundle_metadata',lambda m:seen.setdefault('meta',m))
    class Dataset:
        def __init__(self,path,**kw):
            seen.setdefault('dataset_constructors',[]).append((path,kw))
            self.parquet_path=path;self.role=None;self.storage=object();self.df=pd.DataFrame({'feature':[1.,2.]})
        def __len__(self):return population if self.parquet_path==files["entry_train_parquet"] else val_population
        def bind_policy_dependent_auxiliary_targets(self,**kw):
            assert self.role is None;self.role=kw['role'];self.label_binding=kw
        def bind_unified_exit_lifecycle_v2(self,adapter):self._unified_exit_lifecycle_v2=adapter
        def bind_random_access_entry_coordinate_mapping_v1(self,**kw):
            self._random_access_child_index_by_parent=dict(zip(kw['parent_entry_row_indices'],kw['child_entry_row_indices']))
        def _get_exit_multi_tf_episode_histories(self,*args):raise AssertionError('No model input materialization')
    monkeypatch.setattr(runner.val,'EntryV10CtxDataset',Dataset)
    class Corpus:
        @classmethod
        def from_random_access_index(cls, **kw):
            assert physical is not None
            assert kw["root_manifest_path"] == files["random_access_root"]
            return cls(**kw)
        def __init__(self,**kw):
            expected=("train",) if physical is None else ("train","val")
            assert kw['splits']==expected and set(kw['entry_parquets'])==set(expected)
            seen['corpus']=kw;self.splits={key:object() for key in expected}
            self.evidence={'root_manifest_sha256':'3'*64};seen["feature_sources"]=self.splits
    monkeypatch.setattr(runner.val,'UnifiedExitLifecycleCorpus',Corpus)
    monkeypatch.setattr(runner.val,'require_random_access_index_root',lambda r:r)
    class Adapter:
        _random_access_train={};_native_random_access_index=frame
        def set_full_population_epoch_index(self,epoch):
            assert epoch==0
            return {'every_entry_pair_exactly_once':True,'entry_pair_count':population,'epoch_index':epoch}
        def random_access_selected_entry_rows_v1(self):
            return tuple(range(population))[::-1] if physical is None else tuple(physical["order"].tolist())
        def random_access_training_bindings_v1(self):
            return {"sampler_contract":physical["selected"]["selected_sampler_contract"],
                    **physical["selected"]["reference_workload"]}
    def factory(**kw):
        seen['adapter_binding']=kw
        def build(budget):seen["sampler_budget"]=budget;return Adapter()
        return build
    monkeypatch.setattr(runner,'build_random_access_train_adapter_factory_v1',factory)
    def manifest(data,**kw):
        assert kw['expected_split']==source_split
        return {'manifest_sha256':root["splits"][source_split]["manifest_sha256"]}
    monkeypatch.setattr(runner.val,'require_random_access_index_manifest',manifest)
    from gx1.contracts import unified_exit_bounded_val_cohort_v1 as cohort_owner
    control_rows=list(range(4700,4956)) if physical is None else np.load(
        design["calendar"]["bindings"]["CONTROL256_PARENT_ROWS"]["path"]).tolist()
    cohort={'entry_row_indices':control_rows,'parent_entry_row_indices':control_rows,
            'source_split':source_split,'population_rows':val_population,'source_index':_bind(vp)}
    def build_cohort(*a,**kw):seen["cohort_arguments"]=(a,kw);return cohort
    monkeypatch.setattr(cohort_owner,'build_chronological_control_cohort',build_cohort)
    monkeypatch.setattr(runner.val,'require_parent_entry_coordinate_equivalence',lambda **kw:seen.setdefault('parent_equivalence',kw))
    monkeypatch.setattr(runner.val,'_load_val_economics_readiness',lambda path:{'economics_objective_contract':{}})
    monkeypatch.setattr(runner.val,'_build_provider',lambda **kw:SimpleNamespace(economic_exit_step_manifest={}))
    monkeypatch.setattr(runner.val,'_source_path',lambda manifest,name:tmp_path/name)
    def val_factory(**kw):
        role = "train" if physical is not None and kw["source_split"] == "train" else "control"
        seen[role + "_factory"] = kw
        return SimpleNamespace(source_split=kw["source_split"])

    monkeypatch.setattr(runner.val.RandomAccessValStateFactoryV1,'from_artifacts',val_factory)
    def model(meta,normalization,device,**model_kwargs):
        expected_context = "train_context" if "frozen_train_policy_scope" in args else "control_context"
        assert expected_context in seen
        seen["model_constructed"]=True
        seen["model_kwargs"]=model_kwargs
        return TinyModel(normalization,**model_kwargs)
    monkeypatch.setattr(runner.val,'_model',model)
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
    assert c['weight_ema']._phase_inactive_parameter_names == frozenset()
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

# Current physical preprocessing and frozen row identity use real contract
# builders on synthetic sources. These are not native model/market evidence.
@pytest.fixture(scope="module")
def physical_component_templates():
    from tests.test_native_prefix_recipe import _physical_normalization_templates_for_rows
    return _physical_normalization_templates_for_rows(32768)


@pytest.fixture
def physical_component_case(tmp_path, physical_component_templates):
    from tests.test_native_prefix_recipe import physical_recipe
    from tests.test_unified_exit_selected_sampler_v1 import _direct_fixture, _direct_build
    from gx1.contracts.entry_model_native_training_run_lineage_v1 import deterministic_uniform_subsample_indices
    from gx1.contracts.unified_exit_random_access_sampler_v1 import schedule_random_access_entry_anchors
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import native_sha256
    c = physical_recipe.__wrapped__(tmp_path, physical_component_templates)
    design = c["design"]
    actual = json.loads(Path("configs/research/NATIVE_V38_LEARNING_DESIGN_20261001.json").read_text())
    design["initialization"]["target"] = actual["initialization"]["target"]
    design["budget"].update(train_only_probe_entries=256, train_batch_size=16)
    design["selection"]["seed"] = 20260911
    c["seal"]()
    value = c["recipe"]["chronological_prefix"]
    sampler_dir = tmp_path / "sampler"; sampler_dir.mkdir()
    sample_case = _direct_fixture(sampler_dir, population=32768, design_path=Path(value["design"]["path"]))
    selected = _direct_build(sample_case)
    selected_binding = _write(sampler_dir / "SELECTED.json", selected)
    order = np.array([
        row["entry_row_index"] for row in schedule_random_access_entry_anchors(
            sampler_contract=selected["selected_sampler_contract"], epoch_index=0)
    ], dtype=np.int64)
    assert len(order) == 32768
    planned = order[:4096]
    probe = planned[deterministic_uniform_subsample_indices(
        population_rows=4096, requested_rows=256, seed=20260911, split_salt=0)]
    bindings = {"TRAIN_ELIGIBLE_PARENT_ROWS": design["calendar"]["bindings"]["TRAIN_CALENDAR_PARENT_ROWS"]}
    for name, rows in (("TRAIN_NATIVE_EPOCH0_ORDER", order), ("TRAIN_NATIVE4096_PARENT_ROWS", planned),
                       ("TRAIN256_PROBE_PARENT_ROWS", probe)):
        path = tmp_path / (name + ".npy"); np.save(path, rows); bindings[name] = _bind(path)
    coordinates = {
        "schema_version": "gx1_physical_native_training_coordinates_v1",
        "decision": "TRAIN_COORDINATES_FROZEN_NO_MODEL", "design": value["design"],
        "train_source": c["sources"]["train"]["parquet"], "control_source": c["sources"]["val"]["parquet"],
        "control_parent_rows": design["calendar"]["bindings"]["CONTROL256_PARENT_ROWS"],
        "selected_sampler": selected_binding, "bindings": bindings,
        "selection_uses_outcome_values": False, "test_data_used": False,
        "model_forwards": 0, "optimizer_steps": 0, "fits": 0,
    }
    def seal_coordinates():
        coordinates.pop("coordinates_sha256", None)
        coordinates["coordinates_sha256"] = native_sha256(coordinates)
        value["native_coordinates"] = _write(tmp_path / "native-coordinates.json", coordinates)
    seal_coordinates()
    files = {
        **{f"entry_{split}_{kind}": Path(c["sources"][split][kind]["path"])
           for split in ("train", "val") for kind in ("parquet", "manifest")},
        "selected_sampler": Path(selected_binding["path"]),
        "random_access_root": sample_case["root_path"], "sampler_candidate_set": sample_case["candidate_path"],
        "child_composite_normalization": Path(c["normalization"]["artifacts"]["COMPOSITE_NORMALIZATION.json"]["path"]),
        "source_bundle_metadata": sampler_dir / "SOURCE_BUNDLE.json",
        "val_sequence_source_audit": tmp_path / "val-audit.json",
    }
    files["val_sequence_source_audit"].write_text("{}")
    return {**c, "value": value, "files": files, "selected": selected, "coordinates": coordinates,
            "seal_coordinates": seal_coordinates, "order": order, "probe": probe, "sample_case": sample_case}


def test_physical_component_identity_reuses_sources_normalization_and_actual_sampler_rows(physical_component_case):
    c = physical_component_case
    result = _check(c["value"], c["files"])
    assert result["model_functions"] == runner.trainer._PREFIX_CURRENT_MODEL_FUNCTIONS
    assert result["physical_preprocessing"]["physical_sources"] == c["sources"]
    assert np.array_equal(result["epoch0_parent_order"], c["order"])
    assert np.array_equal(result["physical_coordinates"]["probe_parent_rows"], c["probe"])
    assert result["eligible_parent_rows"].tolist() == list(range(32768))
    assert np.intersect1d(result["eligible_parent_rows"], np.arange(256)).size == 256
    assert result["normalization"] == c["normalization"]
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_chronological_prefix_recipe
    recipe = require_chronological_prefix_recipe(c["recipe"])
    assert np.array_equal(recipe["physical_coordinates"]["epoch0_parent_order"], c["order"])


@pytest.mark.parametrize("fault", [
    "missing_coordinates", "train_as_control", "wrong_source", "wrong_selected_path",
    "missing_val_audit", "reordered_plan", "different_probe", "changed_probe_bytes",
    "outcome_selection", "prior_optimizer", "wrong_design", "wrong_row_dtype",
])
def test_physical_component_rejects_unbound_or_reselected_coordinates(physical_component_case, fault):
    c = physical_component_case
    value, files, coords = c["value"], c["files"], c["coordinates"]
    if fault == "missing_coordinates": value.pop("native_coordinates")
    if fault == "train_as_control": files["entry_val_parquet"] = files["entry_train_parquet"]
    if fault == "wrong_source": coords["train_source"] = coords["control_source"]
    if fault == "wrong_selected_path":
        p = files["selected_sampler"].with_name("COPY.json");p.write_bytes(files["selected_sampler"].read_bytes())
        files["selected_sampler"] = p
    if fault == "missing_val_audit": files.pop("val_sequence_source_audit")
    if fault in ("reordered_plan", "different_probe", "wrong_row_dtype", "changed_probe_bytes"):
        key = "TRAIN_NATIVE4096_PARENT_ROWS" if fault == "reordered_plan" else "TRAIN256_PROBE_PARENT_ROWS"
        p = Path(coords["bindings"][key]["path"]); rows = np.load(p)
        if fault in ("different_probe", "changed_probe_bytes"): rows = rows[::-1].copy()
        if fault == "reordered_plan": rows = rows[::-1].copy()
        if fault == "wrong_row_dtype": rows = rows.astype(np.int32)
        np.save(p, rows)
        if fault != "changed_probe_bytes": coords["bindings"][key] = _bind(p)
    if fault == "outcome_selection": coords["selection_uses_outcome_values"] = True
    if fault == "prior_optimizer": coords["optimizer_steps"] = 1
    if fault == "wrong_design": coords["design"] = {**coords["design"], "sha256":"f"*64}
    if fault != "missing_coordinates": c["seal_coordinates"]()
    with pytest.raises(RuntimeError):
        _check(value, files)


def test_changed_native_order_is_rejected_before_model_or_optimizer(component_chain, monkeypatch):
    args, seen, prefix, _ = component_chain
    prefix["epoch0_parent_order"] = prefix["epoch0_parent_order"][::-1].copy()
    def forbidden(*a, **kw):
        pytest.fail("Model/optimizer constructed before frozen order validation")
    monkeypatch.setattr(runner, "_fresh_prefix_model", forbidden)
    monkeypatch.setattr(runner.torch.optim, "AdamW", forbidden)
    with pytest.raises(RuntimeError, match="NATIVE_ORDER_MISMATCH"):
        runner._build_bound_full_train_components(**args)


def test_current_component_owner_routes_distinct_train_and_val_before_initialization(
    physical_component_case, tmp_path, monkeypatch,
):
    case=physical_component_case
    args,seen,prefix,composite=_component_chain(tmp_path,monkeypatch,case)
    components=runner._build_bound_full_train_components(**args)
    train,control,probe=(components[k] for k in ("train_ds","val_ds","train_probe_ds"))
    assert train.parquet_path==args["files"]["entry_train_parquet"]
    assert control.parquet_path==args["files"]["entry_val_parquet"]
    assert train.storage is not control.storage
    assert train.role=="TRAIN" and control.role=="CONTROL256"
    assert probe.storage is train.storage and probe.df is train.df
    assert not hasattr(probe,"_unified_exit_lifecycle_v2")
    assert not hasattr(control,"_unified_exit_lifecycle_v2")
    assert [p for p,_ in seen["dataset_constructors"]]==[train.parquet_path,control.parquet_path]
    assert seen["dataset_constructors"][0][1]["sequence_source_audit_json"]==args["files"]["sequence_source_audit"]
    assert seen["dataset_constructors"][1][1]["sequence_source_audit_json"]==args["files"]["val_sequence_source_audit"]
    assert seen["corpus"]["splits"]==("train","val")
    assert seen["adapter_binding"]["train_feature_source_owner"] is seen["feature_sources"]["train"]
    assert seen["control_factory"]["source_owner"] is seen["feature_sources"]["val"]
    assert seen["sampler_budget"]==case["selected"]["transition_budget_per_epoch"]
    assert seen["control_factory"]["source_split"]=="val"
    assert seen["control_factory"]["mtf_materializer"].__self__ is control
    assert seen["train_factory"]["source_owner"] is seen["feature_sources"]["train"]
    assert seen["train_factory"]["mtf_materializer"].__self__ is components["train_probe_ds"]
    assert seen["train_factory"]["source_split"] == "train"
    assert "evaluation_cohort" not in seen["train_factory"]
    assert components["measurement_state_factories"]["control"] is components["native_val_context"]["state_factory"]
    assert components["measurement_state_factories"]["train"].source_split == "train"

    assert seen["control_factory"]["evaluation_cohort"]==components["native_val_context"]["evaluation_cohort"]
    assert seen["cohort_arguments"][1]["source_index_manifest_binding"]==_bind(tmp_path/"val-index.json")
    assert seen["parent_equivalence"]["expected_split"]=="val"
    assert seen["parent_equivalence"]["parent_parquet"]==case["sources"]["val"]["parquet"]
    assert components["native_val_context"]["val_sequence_audit"]==args["files"]["val_sequence_source_audit"]
    assert len(components["native_val_context"]["frame"])==256
    assert components["effective_train_rows"]==32768
    assert np.array_equal(components["prefix_epoch0_parent_order"],case["order"])
    assert components["seed_binding"]["checkpoint_loaded"] is False
    assert components["optimizer"].state_dict()["state"]=={} and components["weight_ema"].steps==0
    assert seen["model_constructed"] is True


@pytest.mark.parametrize("fault", ["source_bytes","calendar","order"])
def test_current_component_owner_rejects_mismatch_before_initialization(
    physical_component_case, tmp_path, monkeypatch, fault,
):
    case=physical_component_case
    args,seen,prefix,_=_component_chain(tmp_path,monkeypatch,case)
    if fault=="source_bytes":args["files"]["entry_val_parquet"].write_text("changed")
    elif fault=="calendar":
        # Prefix metadata was already validated; the independently read manifest
        # now reports another period.
        original=runner.val._read
        def read(path):
            data=copy.deepcopy(original(path))
            if path==args["files"]["entry_val_manifest"]:
                data["splits"]["val"]["start"]="2021-06-02T00:00Z"
            return data
        monkeypatch.setattr(runner.val,"_read",read)
    else:prefix["epoch0_parent_order"]=prefix["epoch0_parent_order"][::-1].copy()
    def forbidden(*a,**kw):pytest.fail("Model/optimizer reached before physical identity checks")
    monkeypatch.setattr(runner,"_fresh_prefix_model",forbidden)
    monkeypatch.setattr(runner.torch.optim,"AdamW",forbidden)
    with pytest.raises(RuntimeError,match={
        "source_bytes":"SOURCE_BYTES_MISMATCH","calendar":"WINDOW_MISMATCH","order":"NATIVE_ORDER_MISMATCH",
    }[fault]):
        runner._build_bound_full_train_components(**args)


def test_physical_coordinate_identity_rejects_index_of_different_parent_source(physical_component_case):
    from tests.test_unified_exit_selected_sampler_v1 import _seal, _direct_build, _file_binding
    case=physical_component_case
    sampler=case["sample_case"]
    manifest=sampler["manifest"]
    manifest["source_bindings"]["parent_entry_parquet"]={**manifest["source_bindings"]["parent_entry_parquet"],
                                                       "path":str(Path(manifest["source_bindings"]["parent_entry_parquet"]["path"]).with_name("other-train.parquet"))}
    _seal(sampler["manifest_path"],manifest,"manifest_sha256")
    root=sampler["root"]
    root["splits"]["train"]["manifest_sha256"]=manifest["manifest_sha256"]
    root["full_train_population"]["train_manifest_sha256"]=manifest["manifest_sha256"]
    _seal(sampler["root_path"],root,"root_sha256")
    sampler["receipt"]["run_bindings"]["files"]["root_manifest"]=_file_binding(sampler["root_path"])
    _seal(sampler["receipt_path"],sampler["receipt"],"receipt_sha256")
    selected=_direct_build(sampler)
    case["coordinates"]["selected_sampler"]=_write(case["files"]["selected_sampler"],selected)
    case["seal_coordinates"]()
    with pytest.raises(RuntimeError,match="INDEX_PARENT_SOURCE_MISMATCH"):
        _check(case["value"],case["files"])


def test_native_component_ema_uses_actual_observed_phase_and_exactly_preserves_inactive_weights(physical_component_case, tmp_path, monkeypatch):
    from gx1.contracts import entry_observed_market_horizons_v1 as hz
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_M5_PATH_TRAINING_PHASE
    args, seen, prefix, composite = _component_chain(tmp_path, monkeypatch, physical=physical_component_case)
    def bind(dataset, scope):
        dataset._entry_observed_market_binding = {'identity': {'training_phase': scope['training_phase']}}
    monkeypatch.setattr(hz, 'bind_observed_entry_dataset', bind)
    def model(meta, normalization, device, **model_kwargs):
        result = TinyModel(normalization, **model_kwargs)
        result.head_dip = torch.nn.Linear(3, 2)
        result.head_exit_action = torch.nn.Linear(3, 2)
        return result
    monkeypatch.setattr(runner.val, '_model', model)
    result = runner._build_bound_full_train_components(**args,
        observed_entry_scope={'training_phase': ENTRY_ONLY_M5_PATH_TRAINING_PHASE})
    model, ema = result['model'], result['weight_ema']
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    assert {'head_dip.weight', 'head_exit_action.weight'} <= ema._phase_inactive_parameter_names
    optimized = {id(parameter) for group in result['optimizer'].param_groups for parameter in group['params']}
    assert all(not parameter.requires_grad and id(parameter) not in optimized
        for name, parameter in model.named_parameters() if name in ema._phase_inactive_parameter_names)
    for _ in range(9):
        with torch.no_grad():
            model.backbone.weight.add_(1.)
        ema.update(model)
    shadow = ema.state_dict_clone()
    for name in ema._phase_inactive_parameter_names:
        assert torch.equal(shadow[name], initial[name])
    assert not torch.equal(shadow['backbone.weight'], model.backbone.weight)


@pytest.mark.parametrize('fault', ['missing_dataset_binding', 'scope_phase_differs'])
def test_native_component_ema_rejects_unbound_or_mismatched_observed_phase(physical_component_case, tmp_path, monkeypatch, fault):
    from gx1.contracts import entry_observed_market_horizons_v1 as hz
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_M5_PATH_TRAINING_PHASE
    args, seen, prefix, composite = _component_chain(tmp_path, monkeypatch, physical=physical_component_case)
    def bind(dataset, scope):
        if fault == 'scope_phase_differs':
            dataset._entry_observed_market_binding = {'identity': {'training_phase': 'entry_only'}}
    monkeypatch.setattr(hz, 'bind_observed_entry_dataset', bind)
    with pytest.raises(RuntimeError, match='EMA_PHASE_BINDING_MISMATCH'):
        runner._build_bound_full_train_components(**args,
            observed_entry_scope={'training_phase': ENTRY_ONLY_M5_PATH_TRAINING_PHASE})


@pytest.fixture
def dynamic_constructor_case(physical_component_case, tmp_path, monkeypatch):
    """Constructor/optimizer UNIT boundary, not a complete dynamic native route.

    The shared scaffold stubs completed data owners. Its pre-prefix order call
    is separately blocked in production; reuse the scaffold's frozen order
    here without fabricating or binding dynamic epoch-order authority.
    """
    from gx1.contracts import entry_observed_market_horizons_v1 as hz
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    args, seen, prefix, _ = _component_chain(tmp_path, monkeypatch, physical=physical_component_case)
    phase = ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    def completed_order_boundary(dataset, *, epoch_index, parent_population):
        assert epoch_index == 0
        assert dataset._entry_observed_market_binding["identity"]["training_phase"] == phase
        assert parent_population.dtype == torch.int64
        assert torch.equal(parent_population, torch.as_tensor(prefix["eligible_parent_rows"]))
        seen.setdefault("constructor_order_boundary_calls", []).append(epoch_index)
        return torch.as_tensor(prefix["epoch0_parent_order"].copy(), dtype=torch.int64)
    monkeypatch.setattr(runner.trainer, "_candidate_training_epoch_order", completed_order_boundary)
    queries = runner.trainer._dynamic_m5_model_kwargs(phase)["dynamic_m5_query_minutes"]
    scope = {"training_phase": phase, "query_minutes": queries}
    def bind(dataset, bound_scope):
        dataset._entry_observed_market_binding = {"identity": {
            "training_phase": bound_scope["training_phase"], "query_minutes": list(bound_scope["query_minutes"])}}
    monkeypatch.setattr(hz, "bind_observed_entry_dataset", bind)
    def model(meta, normalization, device, **model_kwargs):
        seen["model_kwargs"] = model_kwargs
        result = TinyModel(normalization, **model_kwargs)
        result.head_exit_action = torch.nn.Linear(3, 2)
        result.head_entry_action_q = torch.nn.Linear(3, 3)
        result.head_dip = torch.nn.Linear(3, 2)
        seen["constructed_model"] = result
        return result
    monkeypatch.setattr(runner.val, "_model", model)
    return args, seen, scope


def test_native_dynamic_constructor_uses_bound_phase_queries_and_freezes_before_optimizer(dynamic_constructor_case):
    from gx1.contracts.entry_model_native_training_objective_v1 import active_joint_task_names
    from gx1.contracts.entry_observed_market_v1 import is_inactive_phase_parameter
    args, seen, scope = dynamic_constructor_case
    result = runner._build_bound_full_train_components(**args, observed_entry_scope=scope)
    model, optimizer, ema = result["model"], result["optimizer"], result["weight_ema"]
    assert seen["constructor_order_boundary_calls"] == [0]
    assert not hasattr(result["train_ds"], "_entry_dynamic_m5_epoch_order_binding")
    assert seen["model_kwargs"] == runner.trainer._dynamic_m5_model_kwargs(scope["training_phase"])
    assert model.entry_training_phase == scope["training_phase"]
    assert tuple(model.dynamic_m5_query_horizons.tolist()) == scope["query_minutes"] == model.cfg.dynamic_m5_query_minutes
    active_tasks = {name for name, value in model.task_log_variances.items() if value.requires_grad}
    assert active_tasks == set(active_joint_task_names(scope["training_phase"]))
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    inactive = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
        if is_inactive_phase_parameter(name, scope["training_phase"])}
    assert inactive and set(inactive) == set(ema._phase_inactive_parameter_names)
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad is (name not in inactive)
        if name in inactive: assert id(parameter) not in optimized and parameter.grad is None
    optimizer.zero_grad(set_to_none=True)
    model.backbone.weight.sum().backward()
    optimizer.step()
    ema.update(model)
    for name, parameter in model.named_parameters():
        if name in inactive:
            assert parameter.grad is None and torch.equal(parameter, inactive[name])
            assert torch.equal(ema.state_dict_clone()[name], inactive[name])


@pytest.mark.parametrize("fault", ["scope_queries", "binding_queries", "model_phase", "model_config_phase", "model_tasks", "model_queries", "model_config"])
def test_native_dynamic_constructor_rejects_phase_task_or_query_drift_before_optimizer(dynamic_constructor_case, monkeypatch, fault):
    from gx1.contracts import entry_observed_market_horizons_v1 as hz
    args, seen, scope = dynamic_constructor_case
    if fault == "scope_queries": scope["query_minutes"] = scope["query_minutes"][:-1]
    if fault == "binding_queries":
        original = hz.bind_observed_entry_dataset
        def bind(dataset, bound_scope):
            original(dataset, bound_scope)
            dataset._entry_observed_market_binding["identity"]["query_minutes"].reverse()
        monkeypatch.setattr(hz, "bind_observed_entry_dataset", bind)
    original_model = runner.val._model
    def model(*args, **kwargs):
        result = original_model(*args, **kwargs)
        if fault == "model_phase": result.entry_training_phase = runner.trainer.JOINT_FITTED_Q_TRAINING_PHASE
        elif fault == "model_config_phase": result.cfg.entry_training_phase = runner.trainer.JOINT_FITTED_Q_TRAINING_PHASE
        elif fault == "model_tasks": result.task_log_variances.pop(next(iter(result.task_log_variances)))
        elif fault == "model_queries": result.dynamic_m5_query_horizons[0] += 1
        elif fault == "model_config": result.cfg.dynamic_m5_query_minutes = result.cfg.dynamic_m5_query_minutes[:-1]
        return result
    monkeypatch.setattr(runner.val, "_model", model)
    def forbidden(*args, **kwargs): raise AssertionError("Invalid component reached optimizer or EMA")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(runner.trainer, "_WeightEma", forbidden)
    with pytest.raises(RuntimeError, match="NATIVE_COMPONENT_.*MISMATCH"):
        runner._build_bound_full_train_components(**args, observed_entry_scope=scope)
    if fault in {"scope_queries", "binding_queries"}: assert "constructed_model" not in seen


@pytest.fixture
def dynamic_entry_component_chain(tmp_path, monkeypatch):
    """Component join unit: strict data/cohort/session owners are separately qualified."""
    from gx1.contracts import entry_observed_market_horizons_v1 as hz
    from gx1.contracts import unified_exit_bounded_val_cohort_v1 as cohorts
    from gx1.contracts.entry_exit_production_architecture_v1 import PRODUCTION_MTF_PER_TF_WINDOW_BARS
    phase = runner.trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    queries = runner.trainer._dynamic_m5_model_kwargs(phase)["dynamic_m5_query_minutes"]
    files = {key: tmp_path / (key + ".json") for key in runner._DYNAMIC_M5_DATA_FILES}
    for path in files.values(): path.write_text("{}")
    base = {"contract_sha256": "a" * 64, "lineage": {"dataset_run_id": "UNIT_DYNAMIC_M5"}}
    _write(files["entry_only_base_normalization"], {"contract": base})
    meta = {"seq_len": 96, "multi_tf": {tf.lower() + "_seq_len": count
            for tf, count in PRODUCTION_MTF_PER_TF_WINDOW_BARS}}
    _write(files["source_bundle_metadata"], meta)
    design = {"scope": {"entry_only_training_phase": phase}}
    bindings = {"design": _write(tmp_path / "design-dynamic.json", design),
        "normalization_result": _bind(files["entry_only_base_normalization"]),
        "labels_result": _write(tmp_path / "labels-dynamic.json", {})}
    for split in ("train", "val"):
        _write(files[f"entry_{split}_manifest"], {"extra": {"entry_run_id": "UNIT_DYNAMIC_M5"}})
    source = {split: {"parquet": _bind(files[f"entry_{split}_parquet"]),
        "manifest": _bind(files[f"entry_{split}_manifest"]), "physical_rows": count}
        for split, count in (("train", 17), ("val", 16))}
    single = {"binding": _write(tmp_path / "witness-dynamic.json", {}), "train_rows": 17,
        "last_batch_rows": 1, "optimizer_steps": 2,
        "witness": {"training_phase": phase, "native_sampler_contract": {"unit": "frozen Entry order"}}}
    physical = {"entry_training_phase": phase, "physical_sources": source,
        "entry_only_base_normalization": _bind(files["entry_only_base_normalization"]),
        "base_contract_sha256": base["contract_sha256"],
        "sequence_source_audits": {"train": _bind(files["sequence_source_audit"]),
                                   "val": _bind(files["val_sequence_source_audit"])}}
    prefix = {"bindings": bindings, "design": design, "physical_preprocessing": physical,
        "entry_m5_path_single_epoch": single, "model_functions": runner.trainer._PREFIX_CURRENT_MODEL_FUNCTIONS,
        "eligible_parent_rows": np.arange(17, dtype=np.int64),
        "epoch0_parent_order": np.arange(17, dtype=np.int64)[::-1].copy()}
    from gx1.contracts import entry_dynamic_m5_loss_scale_v1 as loss_owner
    from tests.test_entry_dynamic_m5_loss_scale import _synthetic_loaded_loss_scales
    loss_scales = _synthetic_loaded_loss_scales(tmp_path / "component-loss-units")
    loss_identity = loss_owner.dynamic_m5_loss_scale_identity(loss_scales)
    prefix["dynamic_m5_loss_scales"] = loss_identity
    # This component unit already isolates completed physical/quote ownership.
    # The artifact itself still goes through the real strict loader before model
    # or Dataset construction; its three synthetic rows claim no physical fit.
    def completed_loss_lineage(scope, population):
        assert scope["training_phase"] == phase and population is single
        return copy.deepcopy(loss_scales["fit_lineage"])
    monkeypatch.setattr(runner, "dynamic_m5_loss_fit_lineage", completed_loss_lineage)
    # The fresh artifact publisher re-enters this completed target-lineage owner
    # through campaign, which imports the contract owner rather than runner's alias.
    monkeypatch.setattr(loss_owner, "dynamic_m5_loss_fit_lineage", completed_loss_lineage)
    events = []
    monkeypatch.setattr(runner, "_require_prefix_component_bindings", lambda *a, **kw: prefix)
    monkeypatch.setattr(runner.val, "_bind_multi_tf_cache_from_source_bundle_metadata", lambda m: None)
    class Dataset:
        def __init__(self, path, **kwargs):
            self.parquet_path = path; self.count = 17 if path == files["entry_train_parquet"] else 16
            events.append(("dataset", path, kwargs))
        def __len__(self): return self.count
        def bind_policy_dependent_auxiliary_targets(self, **kwargs):
            self.role = kwargs["role"]; events.append(("labels", self.role))
    monkeypatch.setattr(runner.val, "EntryV10CtxDataset", Dataset)
    identity = {"training_phase": phase, "query_minutes": list(queries)}
    def bind(dataset, scope):
        dataset._entry_observed_market_binding = {"identity": copy.deepcopy(identity), "role": dataset.role}
    monkeypatch.setattr(hz, "bind_observed_entry_dataset", bind)
    def model(metadata, normalization, device, **kwargs):
        events.append(("model", kwargs)); value = TinyModel(normalization, **kwargs)
        value.head_exit_action = torch.nn.Linear(3, 2)
        return value
    monkeypatch.setattr(runner.val, "_model", model)
    def session(**kwargs):
        assert kwargs["train_ds"].role == "TRAIN" and kwargs["val_ds"].role == "CONTROL256"
        assert not kwargs["model"].head_exit_action.weight.requires_grad
        kwargs["train_ds"].verified = True; events.append(("session_binding",))
        assert kwargs["train_ds"]._entry_dynamic_m5_loss_scale_identity == loss_identity
        assert kwargs["val_ds"]._entry_dynamic_m5_loss_scale_identity == loss_identity
        return {"unit_verified": True, "dynamic_m5_loss_scales": loss_identity}, torch.arange(17), torch.arange(17).flip(0)
    monkeypatch.setattr(runner.trainer, "_prefix_candidate_training_binding", session)
    def order(dataset, **kwargs):
        assert dataset.verified; events.append(("epoch_order",)); return torch.arange(17).flip(0)
    monkeypatch.setattr(runner.trainer, "_candidate_training_epoch_order", order)
    cohort = {"measurement_only": True, "training_phase": phase, "source_split": "val"}
    monkeypatch.setattr(cohorts, "build_chronological_measurement_cohort", lambda *a, **kw: copy.deepcopy(cohort))
    monkeypatch.setattr(runner.trainer, "_native_candidate_val_context_binding",
                        lambda context: events.append(("context", copy.deepcopy(context))))
    def forbidden(*a, **kw): pytest.fail("Dynamic Entry construction entered an Exit owner")
    monkeypatch.setattr(runner, "_require_component_sampler", forbidden)
    monkeypatch.setattr(runner, "build_random_access_train_adapter_factory_v1", forbidden)
    monkeypatch.setattr(runner.val, "UnifiedExitLifecycleCorpus", forbidden)
    original = runner.torch.optim.AdamW
    def optimizer(*a, **kw):
        assert any(event[0] == "session_binding" for event in events)
        assert any(event[0] == "context" for event in events)
        events.append(("optimizer",)); return original(*a, **kw)
    monkeypatch.setattr(runner.torch.optim, "AdamW", optimizer)
    args = dict(files=files, dataset_run_id="UNIT_DYNAMIC_M5", seed_launch_path=None, seed_authority_path=None,
        seed_authority_file_sha256=None, device=torch.device("cpu"), batch_size=16, epochs=1,
        seed=20260911, learning_rate=.0001, weight_decay=.0001,
        val_limits={"max_model_forwards": 100, "max_state_views": 256, "max_wall_seconds": 600,
                    "progress_interval_forwards": 16}, chronological_prefix=bindings,
        observed_entry_scope={"training_phase": phase, "query_minutes": queries},
        dynamic_m5_loss_scales=loss_scales["binding"])
    return args, events, prefix, identity


def test_dynamic_entry_component_uses_no_exit_and_validates_order_before_optimizer(dynamic_entry_component_chain):
    args, events, prefix, identity = dynamic_entry_component_chain
    result = runner._build_bound_full_train_components(**args)
    assert result["entry_observed_market"] == identity
    assert result["dynamic_m5_loss_scales"] == prefix["dynamic_m5_loss_scales"]
    assert result["train_ds"]._entry_dynamic_m5_loss_scale_identity == prefix["dynamic_m5_loss_scales"]
    assert result["val_ds"]._entry_dynamic_m5_loss_scale_identity == prefix["dynamic_m5_loss_scales"]
    assert result["effective_train_rows"] == 17
    assert result["entry_m5_path_single_epoch"]["last_batch_rows"] == 1
    assert result["train_probe_ds"] is not result["train_ds"]
    assert result["native_val_context"]["val_sequence_audit"] == prefix["physical_preprocessing"]["sequence_source_audits"]["val"]
    assert "state_factory" not in result["native_val_context"] and "measurement_state_factories" not in result
    assert result["unified_exit_lifecycle_evidence"]["exit_lifecycle_bound"] is False
    names = [event[0] for event in events]
    assert names.index("session_binding") < names.index("epoch_order") < names.index("optimizer")
    assert result["optimizer"].state_dict()["state"] == {} and result["weight_ema"].steps == 0
    optimized = {id(parameter) for group in result["optimizer"].param_groups for parameter in group["params"]}
    assert id(result["model"].head_exit_action.weight) not in optimized
    assert not result["model"].head_exit_action.weight.requires_grad
    dataset_calls = [event for event in events if event[0] == "dataset"]
    assert dataset_calls[0][2]["sequence_source_audit_json"] == args["files"]["sequence_source_audit"]
    assert dataset_calls[1][2]["sequence_source_audit_json"] == args["files"]["val_sequence_source_audit"]


@pytest.mark.parametrize("fault", ["missing", "changed"])
def test_dynamic_component_loss_artifact_rejected_before_dataset_model_or_optimizer(dynamic_entry_component_chain, monkeypatch, fault):
    args, events, _, _ = dynamic_entry_component_chain
    args["dynamic_m5_loss_scales"] = (None if fault == "missing" else
        {**args["dynamic_m5_loss_scales"], "sha256": "0" * 64})
    def forbidden(*a, **kw): pytest.fail("Unbound loss units reached model or optimizer")
    monkeypatch.setattr(runner.val, "_model", forbidden)
    monkeypatch.setattr(runner.torch.optim, "AdamW", forbidden)
    with pytest.raises((RuntimeError, ValueError), match="(binding invalid|SHA-256 mismatch)"):
        runner._build_bound_full_train_components(**args)
    assert events == []


@pytest.mark.parametrize("fault", ["exit_reference", "exit_file", "missing_scope", "geometry", "source", "order", "dataset_id"])
def test_dynamic_entry_component_rejects_mixed_identity_before_optimizer(dynamic_entry_component_chain, monkeypatch, fault):
    args, events, prefix, identity = dynamic_entry_component_chain
    if fault == "exit_reference": args["exit_reference_policy"] = {"legacy": True}
    elif fault == "exit_file": args["files"]["random_access_root"] = args["files"]["entry_measurement_coordinates"]
    elif fault == "missing_scope": args["observed_entry_scope"] = None
    elif fault == "geometry":
        path = args["files"]["source_bundle_metadata"]; meta = json.loads(path.read_text()); meta["multi_tf"]["m5_seq_len"] += 1; _write(path, meta)
    elif fault == "source": args["files"]["entry_train_parquet"].write_text("changed")
    elif fault == "order": prefix["epoch0_parent_order"] = prefix["epoch0_parent_order"][::-1].copy()
    elif fault == "dataset_id": args["dataset_run_id"] = "OTHER"
    def forbidden(*a, **kw): pytest.fail("Invalid dynamic components reached optimizer")
    monkeypatch.setattr(runner.torch.optim, "AdamW", forbidden)
    with pytest.raises(RuntimeError, match="NATIVE_(DYNAMIC_M5|PREFIX_NATIVE_ORDER_MISMATCH)"):
        runner._build_bound_full_train_components(**args)
    assert not any(event[0] == "optimizer" for event in events)


def test_dynamic_native_wrapper_binds_fixed_online_without_exit_economics(dynamic_entry_component_chain, tmp_path, monkeypatch):
    from gx1.contracts.entry_candidate_checkpoint_policy_v1 import DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR
    args, events, prefix, identity = dynamic_entry_component_chain
    components = runner._build_bound_full_train_components(**args)
    meta = components["metadata"]
    meta.update(dropout=.1, specialist_fusion={"num_layers": 1, "fusion_scale": 1., "cross_family_fusion_scale": 1.})
    meta["multi_tf"].update(multi_tf_num_layers=1, multi_tf_scale=1.)
    monkeypatch.setattr(runner.trainer, "_require_cuda_trainer_guard_execution", lambda **kw: None)
    monkeypatch.setattr(runner.trainer, "_guard_no_rl", lambda: None)
    monkeypatch.setattr(runner.trainer, "_GRAD_CLIP_NORM", runner.trainer._GRAD_CLIP_NORM)
    monkeypatch.setattr(runner.trainer, "_WEIGHT_DECAY", runner.trainer._WEIGHT_DECAY)
    seen = {}
    monkeypatch.setattr(runner.trainer, "_run_resumable_candidate_training", lambda **kw: seen.update(kw) or {"unit": True})
    def forbidden(*a, **kw): pytest.fail("Dynamic wrapper loaded Exit/economic readiness")
    monkeypatch.setattr(runner.val, "_load_val_economics_readiness", forbidden)
    result = runner._run_bound_full_train_candidate(components=components, files=args["files"],
        device=torch.device("cuda"), run_id="UNIT_DYNAMIC_M5", dataset_run_id="UNIT_DYNAMIC_M5", output=tmp_path / "offline",
        gx1_data=str(tmp_path), seed=args["seed"], learning_rate=args["learning_rate"], weight_decay=args["weight_decay"],
        grad_clip_norm=1., recipe_source_provenance={"unit": True}, execution_budget={"unit": True},
        execution_budget_sha256="b" * 64, invocation_started_monotonic=0.)
    assert result == {"unit": True}
    assert seen["checkpoint_monitor"] == DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR
    assert seen["unified_exit_lifecycle_manifest_path"] is None and seen["epochs"] == 1
    assert seen["save_top_k"] == 0
    assert seen["chronological_prefix"] == args["chronological_prefix"]
    # CUDA guard/session are spies: this is a wrapper join unit, no GPU admission.


@pytest.fixture
def dynamic_fresh_preparation(dynamic_entry_component_chain, tmp_path, monkeypatch):
    """Small model state proof; physical/source admission is an isolated owner boundary."""
    from torch.optim.adamw import AdamW
    from gx1.contracts import gx1_capped_execution_v1 as cap
    args, events, prefix, _ = dynamic_entry_component_chain
    # The model-only producer must not enter the Dataset/session chain used by
    # the component fixture. Its optimizer is the same real owner without that spy.
    monkeypatch.setattr(torch.optim, "AdamW", AdamW)
    def forbidden(*args, **kwargs):
        pytest.fail("Fresh preparation attempted Dataset, forward, optimizer step or EMA update")
    monkeypatch.setattr(runner.val, "EntryV10CtxDataset", forbidden)
    monkeypatch.setattr(AdamW, "step", forbidden)
    monkeypatch.setattr(runner.trainer._WeightEma, "update", forbidden)
    checks = []
    monkeypatch.setattr(cap, "require_capped_cpu_audit_execution", lambda: checks.append("audit"))
    monkeypatch.setattr(cap, "require_capped_lock_ancestry", lambda: checks.append("lock"))
    bindings = args["chronological_prefix"]
    bindings["native_coordinates"] = _write(tmp_path / "preparation-native-coordinates.json", {"unit": "frozen"})
    model_path = Path(runner.__file__).resolve().parents[2] / "gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py"
    source = {"python:model": runner.launch_owner.artifact_binding(model_path)}
    policy = {"training_enabled": False, "full_epoch_training_allowed": False}
    policy_binding = _write(tmp_path / "preparation-policy.json", policy)
    from gx1.contracts import entry_observed_market_horizons_v1 as horizons
    monkeypatch.setattr(horizons, "require_observed_entry_scope", lambda recipe: args["observed_entry_scope"])
    recipe = {"source_repo": str(Path(runner.__file__).resolve().parents[2]),
        "dynamic_m5_loss_scales": args["dynamic_m5_loss_scales"],
        "files": {key: _bind(path) for key, path in args["files"].items()},
        "chronological_prefix": bindings, "next_run_policy": policy_binding,
        "source_bindings": source, "source_bindings_sha256": runner.launch_owner.canonical_json_sha256(source),
        "trainer_cli": {"seed": args["seed"], "batch_size": args["batch_size"], "epochs": args["epochs"],
            "learning_rate": args["learning_rate"], "weight_decay": args["weight_decay"]}}
    def require_preparation(path, digest):
        value = runner.val._read(runner._bound_artifact({"path": str(path), "sha256": digest}))
        return value, {key: Path(binding["path"]) for key, binding in value["files"].items()}, {}
    monkeypatch.setattr(runner, "_require_native_full_train_recipe", require_preparation)
    def seal():
        return _write(tmp_path / "PREPARATION_RECIPE.json", recipe)
    return {"args": args, "events": events, "prefix": prefix, "checks": checks,
        "recipe": recipe, "policy": policy, "policy_path": Path(policy_binding["path"]), "seal": seal,
        "output": tmp_path / "initial-components"}


def test_dynamic_fresh_preparation_publishes_exact_owner_state_without_dataset(dynamic_fresh_preparation, monkeypatch):
    case = dynamic_fresh_preparation
    original_restore = runner._restore_prefix_initial_measurement_state
    restored = []
    def restore(**kwargs):
        binding = kwargs["scope"]["initialization"]["initial_state"]
        path = Path(binding["path"])
        assert path.is_file() and path.parent != case["output"] and not case["output"].exists()
        restored.append(dict(binding))
        return original_restore(**kwargs)
    monkeypatch.setattr(runner, "_restore_prefix_initial_measurement_state", restore)
    preparation = case["seal"]()
    artifacts = runner.publish_dynamic_m5_fresh_initialization(recipe_path=Path(preparation["path"]),
        recipe_file_sha256=preparation["sha256"], output_dir=case["output"])
    assert set(artifacts) == {"initialization_result", "measurement_binding_result"}
    assert {path.name for path in case["output"].iterdir()} == {
        "INITIAL_STATE.pt", "INITIALIZATION.json", "MEASUREMENT_BINDING.json"}
    initial = runner.val._read(runner._bound_artifact(artifacts["initialization_result"]))
    measurement = runner.val._read(runner._bound_artifact(artifacts["measurement_binding_result"]))
    state = torch.load(runner._bound_artifact(initial["initial_state"]), map_location="cpu", weights_only=False)
    assert case["checks"] == ["audit", "lock"] and len(restored) == 1
    assert restored[0]["sha256"] == initial["initial_state"]["sha256"]
    assert initial["initial_state"]["path"] == str(case["output"] / "INITIAL_STATE.pt")
    assert [event[0] for event in case["events"]] == ["model"]
    assert state["optimizer_state"]["state"] == {} and state["weight_ema_state"]["steps"] == 0
    assert initial["target_frozen"] is True and initial["inherited_checkpoint_weights"] is False
    assert initial["model_forwards"] == initial["optimizer_steps"] == initial["ema_steps"] == initial["normalization_refits"] == 0
    assert measurement["actual_model_forwards"] == measurement["optimizer_steps"] == 0
    assert initial["dynamic_m5_loss_scales"] == measurement["dynamic_m5_loss_scales"] == state["dynamic_m5_loss_scales"]
    assert initial["dynamic_m5_loss_scales"] == case["prefix"]["dynamic_m5_loss_scales"]
    assert measurement["coordinate_result"] == case["recipe"]["files"]["entry_measurement_coordinates"]
    assert initial["native_recipe_source_bindings"] == case["recipe"]["source_bindings"]
    assert all("recipe_sha256" not in record for record in (initial, measurement, state))
    # Compare against the same genuine model/optimizer owners used by components,
    # rather than constructing a second data loader or inventing expected weights.
    args = case["args"]
    metadata = runner.val._read(args["files"]["source_bundle_metadata"])
    normalization = runner.val._read(args["files"]["entry_only_base_normalization"])["contract"]
    model, seed = runner._fresh_prefix_model(metadata=metadata, normalization=normalization,
        device=torch.device("cpu"), seed=args["seed"], entry_training_phase=runner.trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    runner._configure_component_training_phase(model, runner.trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    optimizer, ema, scheduler, derivation = runner._component_optimizer_state(model=model,
        entry_phase=runner.trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        train_rows=case["prefix"]["entry_m5_path_single_epoch"]["train_rows"], batch_size=args["batch_size"],
        epochs=args["epochs"], learning_rate=args["learning_rate"], weight_decay=args["weight_decay"])
    expected = runner.trainer._model_state_sha256(model)
    assert expected == seed["model_state_sha256"] == initial["online_model_state_sha256"]
    for surface in (state["model_state"], state["target_model_state"], state["weight_ema_state"]["shadow"]):
        assert runner.val.canonical_model_state_sha256(surface) == expected
        assert all(not tensor.requires_grad and tensor.device.type == "cpu" for tensor in surface.values())
    assert runner.val.canonical_sha256(state["optimizer_state"]) == initial["optimizer_state_sha256"] == runner.val.canonical_sha256(optimizer.state_dict())
    assert runner.val.canonical_sha256(state["lr_scheduler_state"]) == initial["lr_scheduler_state_sha256"] == runner.val.canonical_sha256(None if scheduler is None else scheduler.state_dict())
    assert initial["ema_derivation"] == derivation
    rng = runner.trainer._attended_session_rng_state(device=torch.device("cpu"))
    assert set(rng) == set(state["rng_state"])
    for key, value in rng.items():
        if isinstance(value, torch.Tensor): assert torch.equal(value, state["rng_state"][key])
        else: assert value == state["rng_state"][key]


@pytest.mark.parametrize("fault", ["existing_output", "initial_scope", "learning_scope", "training", "phase", "row_count"])
def test_dynamic_fresh_preparation_rejects_scope_before_model_creation(dynamic_fresh_preparation, fault):
    case = dynamic_fresh_preparation
    if fault == "existing_output": case["output"].mkdir()
    elif fault == "initial_scope": case["recipe"]["chronological_initial_measurement"] = {}
    elif fault == "learning_scope": case["recipe"]["chronological_learning_measurement"] = {}
    elif fault == "training":
        case["policy"]["training_enabled"] = True
        case["recipe"]["next_run_policy"] = _write(case["policy_path"], case["policy"])
    elif fault == "phase": case["prefix"]["physical_preprocessing"]["entry_training_phase"] = runner.trainer.JOINT_FITTED_Q_TRAINING_PHASE
    else: case["prefix"]["entry_m5_path_single_epoch"]["train_rows"] += 1
    binding = case["seal"]()
    with pytest.raises(RuntimeError, match="NATIVE_DYNAMIC_M5_FRESH_PREPARATION_"):
        runner.publish_dynamic_m5_fresh_initialization(recipe_path=Path(binding["path"]),
            recipe_file_sha256=binding["sha256"], output_dir=case["output"])
    assert not case["events"]


def test_dynamic_fresh_preparation_strictloads_staged_optimizer_before_publication(dynamic_fresh_preparation, monkeypatch):
    case = dynamic_fresh_preparation
    original_save = torch.save
    def corrupt(value, path):
        value = copy.deepcopy(value)
        value["optimizer_state"]["param_groups"][0]["lr"] *= 2.0
        original_save(value, path)
    monkeypatch.setattr(torch, "save", corrupt)
    binding = case["seal"]()
    with pytest.raises(RuntimeError, match="INITIAL_OPTIMIZER_STATE_MISMATCH"):
        runner.publish_dynamic_m5_fresh_initialization(recipe_path=Path(binding["path"]),
            recipe_file_sha256=binding["sha256"], output_dir=case["output"])
    assert not case["output"].exists()
    assert len(list(case["output"].parent.glob(".initial-components.staging.*/INITIAL_STATE.pt"))) == 1


def test_fresh_compact_model_binds_explicit_canonical_profile_without_checkpoint_or_forward(tmp_path, monkeypatch):
    from gx1.features.entry_specialist_feature_groups_v1 import compile_compact_entry_execution_profile
    _,_,_,_,_,_,composite = _bindings(tmp_path)
    norm = composite["base_feature_normalization"]["artifact"]["contract"]
    compiled = compile_compact_entry_execution_profile()
    reference = {key:compiled[key] for key in ("name", "contract_sha256")}
    calls = []
    def construct(metadata, normalization, device, **kwargs):
        calls.append(kwargs)
        return TinyModel(normalization, **kwargs)
    monkeypatch.setattr(runner.val, "_model", construct)
    model, binding = runner._fresh_prefix_model(metadata={"entry_execution_profile":reference},
        normalization=norm, device=torch.device("cpu"), seed=20260911,
        entry_training_phase=runner.trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        entry_execution_profile=reference)
    assert calls == [{**runner.trainer._dynamic_m5_model_kwargs(runner.trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE),
        "entry_execution_profile":reference}]
    assert model.entry_execution_profile == model.cfg.entry_execution_profile == binding["entry_execution_profile"] == reference
    assert binding["checkpoint_loaded"] is False


@pytest.mark.parametrize("fault", ["missing_argument", "metadata_reference", "phase", "constructed_profile"])
def test_fresh_compact_model_rejects_profile_mismatch_before_optimizer(tmp_path, monkeypatch, fault):
    from gx1.features.entry_specialist_feature_groups_v1 import compile_compact_entry_execution_profile
    _,_,_,_,_,_,composite = _bindings(tmp_path)
    compiled = compile_compact_entry_execution_profile()
    reference = {key:compiled[key] for key in ("name", "contract_sha256")}
    args = {"metadata":{"entry_execution_profile":dict(reference)},
        "normalization":composite["base_feature_normalization"]["artifact"]["contract"],
        "device":torch.device("cpu"), "seed":20260911,
        "entry_training_phase":runner.trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        "entry_execution_profile":reference}
    calls = []
    def construct(metadata, normalization, device, **kwargs):
        calls.append(kwargs)
        if fault == "constructed_profile": kwargs.pop("entry_execution_profile")
        return TinyModel(normalization, **kwargs)
    monkeypatch.setattr(runner.val, "_model", construct)
    if fault == "missing_argument": args.pop("entry_execution_profile")
    elif fault == "metadata_reference": args["metadata"]["entry_execution_profile"]["contract_sha256"] = "0"*64
    elif fault == "phase": args["entry_training_phase"] = runner.trainer.JOINT_FITTED_Q_TRAINING_PHASE
    with pytest.raises(RuntimeError): runner._fresh_prefix_model(**args)
    assert len(calls) == (1 if fault == "constructed_profile" else 0)


def test_dynamic_fresh_metadata_hash_scopes_exclude_model_save_and_restore(
    dynamic_fresh_preparation, monkeypatch
):
    """Exercise the public producer, genuine streams and staged CPU roundtrip."""
    from gx1.contracts import local_random_access_campaign_v2 as common
    from gx1.contracts import unified_exit_native_candidate_campaign_v1 as campaign
    case = dynamic_fresh_preparation
    binding = case["recipe"]["files"]["entry_train_parquet"]
    path = Path(binding["path"])
    reads, common_pt_reads, raw_pt_reads = [], [], []
    original_hash = common._file_sha256_uncached
    original_raw_hash = runner.val.file_sha256
    def traced_hash(source, *, identity, scoped):
        if source == path:
            reads.append(scoped)
        if source.suffix == ".pt":
            common_pt_reads.append((source, scoped))
        return original_hash(source, identity=identity, scoped=scoped)
    def traced_raw_hash(source):
        if Path(source).suffix == ".pt":
            raw_pt_reads.append(Path(source))
        return original_raw_hash(source)
    monkeypatch.setattr(common, "_file_sha256_uncached", traced_hash)
    monkeypatch.setattr(runner.val, "file_sha256", traced_raw_hash)

    def metadata_reads():
        assert runner._bound_artifact(binding) == path
        assert common.require_binding(binding, label="same immutable input") == binding
        assert runner._bound_artifact(binding) == path
    require_recipe = runner._require_native_full_train_recipe
    require_prefix = runner._require_prefix_component_bindings
    make_model = runner._fresh_prefix_model
    save = torch.save
    restore = runner._restore_prefix_initial_measurement_state
    require_initial = campaign._require_dynamic_initial_artifacts
    def recipe_boundary(*args, **kwargs):
        metadata_reads()
        return require_recipe(*args, **kwargs)
    def prefix_boundary(*args, **kwargs):
        metadata_reads()
        return require_prefix(*args, **kwargs)
    def model_boundary(*args, **kwargs):
        assert runner._bound_artifact(binding) == path
        return make_model(*args, **kwargs)
    def save_boundary(*args, **kwargs):
        assert runner._bound_artifact(binding) == path
        return save(*args, **kwargs)
    def restore_boundary(*args, **kwargs):
        assert runner._bound_artifact(binding) == path
        return restore(*args, **kwargs)
    def post_state_boundary(*args, **kwargs):
        metadata_reads()
        result = require_initial(*args, **kwargs)
        metadata_reads()
        return result
    monkeypatch.setattr(runner, "_require_native_full_train_recipe", recipe_boundary)
    monkeypatch.setattr(runner, "_require_prefix_component_bindings", prefix_boundary)
    monkeypatch.setattr(runner, "_fresh_prefix_model", model_boundary)
    monkeypatch.setattr(torch, "save", save_boundary)
    monkeypatch.setattr(runner, "_restore_prefix_initial_measurement_state", restore_boundary)
    monkeypatch.setattr(campaign, "_require_dynamic_initial_artifacts", post_state_boundary)
    preparation = case["seal"]()
    for operation in (1, 2):
        output = case["output"].with_name("initial-components-" + str(operation))
        artifacts = runner.publish_dynamic_m5_fresh_initialization(recipe_path=Path(preparation["path"]),
            recipe_file_sha256=preparation["sha256"], output_dir=output)
        assert set(artifacts) == {"initialization_result", "measurement_binding_result"}
        # One pre-model stream, three independent model/save/restore reads,
        # then one fresh post-state metadata stream. A new producer invocation
        # must perform both admissions again rather than keep a lifetime memo.
        assert reads == [True, False, False, False, True, False] * (operation - 1) + [True, False, False, False, True]
        assert runner._bound_artifact(binding) == path
        assert reads == [True, False, False, False, True, False] * operation
        # The real staged archive is physically hashed by the unchanged raw PT
        # owner, then again by strict restore outside either metadata scope.
        assert len(raw_pt_reads) == len(common_pt_reads) == operation
        assert [item[0] for item in common_pt_reads] == raw_pt_reads
        assert all(scoped is False for _, scoped in common_pt_reads)
        assert raw_pt_reads[-1].name == "INITIAL_STATE.pt"
        assert raw_pt_reads[-1].parent != output


@pytest.mark.parametrize("failure_stage", ["pre_model", "post_state"])
def test_dynamic_fresh_metadata_exceptions_clear_scope_without_publishing(
    dynamic_fresh_preparation, monkeypatch, failure_stage
):
    from gx1.contracts import local_random_access_campaign_v2 as common
    from gx1.contracts import unified_exit_native_candidate_campaign_v1 as campaign
    case = dynamic_fresh_preparation
    binding = case["recipe"]["files"]["entry_train_parquet"]
    path = Path(binding["path"])
    reads = []
    original_hash = common._file_sha256_uncached
    def traced(source, *, identity, scoped):
        if source == path:
            reads.append(scoped)
        return original_hash(source, identity=identity, scoped=scoped)
    monkeypatch.setattr(common, "_file_sha256_uncached", traced)
    require_recipe = runner._require_native_full_train_recipe
    def recipe_boundary(*args, **kwargs):
        assert runner._bound_artifact(binding) == path
        assert common.require_binding(binding, label="repeat metadata admission") == binding
        if failure_stage == "pre_model":
            raise RuntimeError("bounded fresh metadata pre_model failure")
        return require_recipe(*args, **kwargs)
    def post_state_boundary(*args, **kwargs):
        assert runner._bound_artifact(binding) == path
        assert common.require_binding(binding, label="repeat post-state admission") == binding
        raise RuntimeError("bounded fresh metadata post_state failure")
    monkeypatch.setattr(runner, "_require_native_full_train_recipe", recipe_boundary)
    monkeypatch.setattr(campaign, "_require_dynamic_initial_artifacts", post_state_boundary)
    preparation = case["seal"]()
    with pytest.raises(RuntimeError, match="bounded fresh metadata " + failure_stage + " failure"):
        runner.publish_dynamic_m5_fresh_initialization(recipe_path=Path(preparation["path"]),
            recipe_file_sha256=preparation["sha256"], output_dir=case["output"])
    assert reads == ([True] if failure_stage == "pre_model" else [True, True])
    assert runner._bound_artifact(binding) == path
    assert reads[-1] is False
    assert not case["output"].exists()
    assert [event[0] for event in case["events"]] == ([] if failure_stage == "pre_model" else ["model"])
