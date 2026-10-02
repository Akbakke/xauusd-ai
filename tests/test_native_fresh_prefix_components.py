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
    def val_factory(**kw):seen['control_factory']=kw;return object()
    monkeypatch.setattr(runner.val.RandomAccessValStateFactoryV1,'from_artifacts',val_factory)
    def model(meta,normalization,device):
        assert "control_context" in seen
        seen["model_constructed"]=True
        return TinyModel(normalization)
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
