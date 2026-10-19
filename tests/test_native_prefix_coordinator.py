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
from tests.test_native_fresh_prefix_components import (
    _bindings, _bind, _write, physical_component_templates, physical_component_case,
)
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
                train_parquet=data.files['entry_train_parquet'],val_parquet=data.files['entry_val_parquet'],
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
        def teacher_copy(model, *, model_functions):
            assert model_functions == trainer._prefix_reference_model_functions(data.design)
            return copy.deepcopy(model)
        self.ns["_copy_frozen_prefix_reference_model"] = teacher_copy
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
            train_ds=d.train,val_ds=d.control,effective_train_rows=int(d.train._policy_dependent_auxiliary_bound_rows.sum()),batch_size=16,num_workers=0,
            pin_memory=False,persistent_workers=False,prefetch_factor=None,epochs=30,
            early_stopping_patience=5,early_stopping_min_delta=0.,minimum_epochs_before_stop=1,save_top_k=1,
            out_bundle_dir=output,gx1_data_override='',run_id='V46_20260825T170935Z_CANDIDATE',dataset_run_id='V46_20260825T170935Z',
            train_parquet=d.files['entry_train_parquet'],val_parquet=d.files['entry_val_parquet'],
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



def test_current_design_cannot_relabel_legacy_same_source_control(tmp_path):
    data = _prepared(tmp_path)
    args = _binding_args(data)
    current = json.loads((Path(__file__).resolve().parents[1] / "configs/research/NATIVE_V38_LEARNING_DESIGN_20261001.json").read_text())
    data.design["calendar"].update({
        "physical_source_splits": current["calendar"]["physical_source_splits"],
        "physical_coordinate_namespaces_are_separate": True,
    })
    data.design["initialization"].update(current["initialization"])
    data.value["design"] = _write(Path(data.value["design"]["path"]), data.design)
    with pytest.raises(RuntimeError, match="PHYSICAL_SPLIT_BINDING_REQUIRED"):
        trainer._prefix_candidate_training_binding(**args)
    from gx1.scripts import run_unified_exit_random_access_full_train_v1 as runner
    with pytest.raises(RuntimeError, match="PHYSICAL_SPLIT_BINDING_REQUIRED"):
        runner._require_prefix_component_bindings(
            data.value, files=data.files, seed=20260911, batch_size=16,
            learning_rate=.0001, weight_decay=.0001,
        )


@pytest.fixture
def physical_prepared(physical_component_case, tmp_path):
    """Real contract/coordinate bytes with synthetic datasets and state owners."""
    from tests.test_native_chronological_control_source import _physical_control_plan
    from gx1.contracts.unified_exit_random_access_index_v1 import canonical_sha256
    case=physical_component_case
    value,files,design=case["value"],dict(case["files"]),case["design"]
    for key in ("m5_prebuilt",):
        files[key]=tmp_path/(key+".json");files[key].write_text("{}")
    composite=case["composite"]
    count=case["sources"]["train"]["physical_rows"]
    class Adapter:
        def __init__(self):
            self._random_access_train={
                **copy.deepcopy(case["selected"]["reference_workload"]),
                "sampler_contract":copy.deepcopy(case["selected"]["selected_sampler_contract"]),
                "normalization_artifact":composite["lifetime_summary_normalization"],
            }
            self._native_random_access_index=pd.DataFrame({
                "entry_row_index":np.arange(count),"parent_entry_row_index":np.arange(count)})
            self.order=case["order"].copy()
        def random_access_training_bindings_v1(self):return self._random_access_train.copy()
        def set_full_population_epoch_index(self,epoch):
            assert epoch==0
            return {"every_entry_pair_exactly_once":True,"entry_pair_count":count,"epoch_index":epoch}
        def random_access_selected_entry_rows_v1(self):return tuple(self.order.tolist())
    class Dataset(Rows):
        def __init__(self,split,role):
            source=case["sources"][split]
            super().__init__(source["physical_rows"]);self.parquet_path=files[f"entry_{split}_parquet"]
            row_binding=design["calendar"]["bindings"]["TRAIN_CALENDAR_PARENT_ROWS" if role=="TRAIN" else "CONTROL256_PARENT_ROWS"]
            rows=np.load(row_binding["path"])
            self._policy_dependent_auxiliary_bound_rows=np.zeros(len(self),dtype=np.bool_)
            self._policy_dependent_auxiliary_bound_rows[rows]=True
            proof=case["labels"]["splits"][split]
            self._policy_dependent_auxiliary_binding={
                "result":value["labels_result"],"design":value["design"],"role":role,
                "mode":"original_physical_split_targets","source_split":split,
                "parent_entry_parquet":source["parquet"],"parent_entry_manifest":source["manifest"],
                "row_binding":row_binding,"rows":len(rows),
                "active_target_columns":list(trainer._MODEL_NATIVE_ACTIVE_TARGET_COLS),
                "policies":{"direction_policy":{"policy_sha256":proof["direction_policy_sha256"]},
                            "position_size_policy":{"policy_sha256":proof["position_size_policy_sha256"]}},
                "targets_rewritten":False,"inactive_diagnostics_refreshed":False,
            }
            if split=="train":
                self._unified_exit_lifecycle_v2=Adapter()
                self._random_access_child_index_by_parent=dict(zip(range(count),range(count)))
        def __getitem__(self,index):
            assert self._policy_dependent_auxiliary_bound_rows[index]
            return super().__getitem__(index)
    directory=tmp_path/"physical-control";directory.mkdir()
    _,ib,mb,_,manifest,frame,_=_physical_control_plan(
        directory,population=case["sources"]["val"]["physical_rows"],
        entry_start=case["sources"]["val"]["declared_window"]["start"])
    for kind in ("parquet","manifest"):
        manifest["source_bindings"][f"entry_{kind}"]=case["sources"]["val"][kind]
        manifest["source_bindings"][f"parent_entry_{kind}"]=case["sources"]["val"][kind]
    manifest["parent_entry_source_sha256"]=case["sources"]["val"]["parquet"]["sha256"]
    manifest["parent_entry_manifest_sha256"]=case["sources"]["val"]["manifest"]["sha256"]
    manifest["manifest_sha256"]=canonical_sha256({k:v for k,v in manifest.items() if k!="manifest_sha256"})
    mb=_write(Path(mb["path"]),manifest)
    cohort=build_chronological_control_cohort(value["design"],source_index_binding=ib,source_index_manifest_binding=mb)
    factory=RandomAccessValStateFactoryV1.__new__(RandomAccessValStateFactoryV1)
    factory.source_split="val"
    factory.artifact_file_sha256={"random_access_index":ib["sha256"]}
    factory.factory_receipt={"split":"val","entry_pair_count":len(frame),
        "composite_normalization_sha256":composite["composite_normalization_sha256"],
        "physical_control_cohort_sha256":cohort["cohort_sha256"]}
    context={"frame":frame.iloc[cohort["entry_row_indices"]].copy(),"state_factory":factory,
        "parent_coordinate_evidence":{"synthetic":True},"val_sequence_audit":files["val_sequence_source_audit"],
        "max_model_forwards":100000,"max_state_views":100000,"max_wall_seconds":10800,
        "progress_interval_forwards":64,"policy_batch_size":256,"cpu_pipeline_workers":8,
        "evaluation_cohort":cohort}
    return SimpleNamespace(value=value,files=files,design=design,norm=case["normalization"],
        composite=composite,train=Dataset("train","TRAIN"),control=Dataset("val","CONTROL256"),
        context=context,case=case)


def test_physical_training_binding_allows_shared_integers_only_in_separate_sources(physical_prepared):
    data=physical_prepared
    binding,parents,order=trainer._prefix_candidate_training_binding(**_binding_args(data))
    assert binding["physical_sources"]==data.case["sources"]
    assert binding["native_coordinates"]==data.value["native_coordinates"]
    assert binding["selected_sampler"]==data.case["coordinates"]["selected_sampler"]
    assert binding["model_functions"]==trainer._PREFIX_CURRENT_MODEL_FUNCTIONS
    assert "prefix_preparation" not in binding
    assert np.array_equal(parents.numpy(),np.arange(len(data.train)))
    assert np.array_equal(order.numpy(),data.case["order"])
    control=np.flatnonzero(data.control._policy_dependent_auxiliary_bound_rows)
    assert len(np.intersect1d(parents.numpy(),control))==256
    assert trainer._native_candidate_val_context_binding(data.context)["factory_receipt"]["split"]=="val"


@pytest.mark.parametrize("fault",[
    "same_file","wrong_dataset","manifest_bytes","control_as_train","source_parent",
    "partial_train","control_reselected","changed_policy","rewritten_targets",
    "wrong_normalization","wrong_sampler","missing_coordinates",
])
def test_physical_training_binding_rejects_mixed_sources_or_supervision(physical_prepared,fault):
    data=physical_prepared;args=_binding_args(data)
    control=data.control._policy_dependent_auxiliary_binding
    training=data.train._unified_exit_lifecycle_v2._random_access_train
    if fault=="same_file":args["val_parquet"]=args["train_parquet"]
    elif fault=="wrong_dataset":data.control.parquet_path=data.train.parquet_path
    elif fault=="manifest_bytes":data.files["entry_val_manifest"].write_text("changed")
    elif fault=="control_as_train":control["source_split"]="train"
    elif fault=="source_parent":control["parent_entry_parquet"]=data.case["sources"]["train"]["parquet"]
    elif fault=="partial_train":data.train._policy_dependent_auxiliary_bound_rows[0]=False
    elif fault=="control_reselected":
        data.control._policy_dependent_auxiliary_bound_rows[0]=False
        data.control._policy_dependent_auxiliary_bound_rows[256]=True
    elif fault=="changed_policy":control["policies"]["position_size_policy"]["policy_sha256"]="f"*64
    elif fault=="rewritten_targets":control["targets_rewritten"]=True
    elif fault=="wrong_normalization":training["normalization_artifact"]={}
    elif fault=="wrong_sampler":training["sampler_contract"]["transition_budget_per_epoch"]=65536
    elif fault=="missing_coordinates":data.value.pop("native_coordinates")
    with pytest.raises(RuntimeError):trainer._prefix_candidate_training_binding(**args)


def test_physical_prefix_resume_preserves_exact_sources_teacher_order_and_optimizer(physical_prepared,tmp_path):
    data=physical_prepared;h=PrefixHarness(tmp_path/"physical-run",data)
    direct,split=h.root/"DIRECT",h.root/"SPLIT"
    h.run_prefix(direct,4);expected=h.state(direct);batches=list(h.batches)
    h.batches.clear();h.run_prefix(split,2)
    _,last=h.run_prefix(split,4,expected_pointer=digest(h.pointer(split)))
    actual=h.state(split)
    for key in ["model_state","target_model_state","optimizer_state","weight_ema_state",
                "lr_scheduler_state","rng_state","epoch_order","training_progress"]:
        equal_tree(expected[key],actual[key])
    assert h.batches==batches
    assert [i for batch in batches for i in batch]==data.case["order"][:64].tolist()
    assert h.validation_batches==0 and len(set(h.teacher_hashes))==1
    assert last["global_optimizer_steps"]==4 and last["epoch_index"]==0 and last["phase"]=="train"
    contract=json.loads((h.pointer(split).parent/trainer._CANDIDATE_TRAINING_CONTRACT_FILENAME).read_text())
    prefix=contract["chronological_prefix"]
    assert prefix["artifacts"]==data.value and prefix["physical_sources"]==data.case["sources"]
    assert prefix["initial_model_state_sha256"]==h.teacher_hashes[0]
    assert contract["artifacts"]["train_parquet"]==data.case["sources"]["train"]["parquet"]
    assert contract["artifacts"]["val_parquet"]==data.case["sources"]["val"]["parquet"]
    assert contract["native_bounded_control"]["evaluation_cohort"]["source_split"]=="val"
    assert prefix["maximum_optimizer_steps"]==256 and prefix["maximum_completed_epochs"]==0


@pytest.mark.parametrize("fault",["source_bytes","native_order","coordinate_authority"])
def test_physical_resume_rejects_changed_sources_or_coordinate_authority_without_updates(physical_prepared,tmp_path,fault):
    data=physical_prepared;h=PrefixHarness(tmp_path/"physical-resume",data)
    output=h.root/"SPLIT";h.run_prefix(output,2)
    pointer_before=h.pointer(output).read_bytes();batches_before=list(h.batches)
    if fault=="source_bytes":data.files["entry_val_parquet"].write_bytes(b"changed physical VAL")
    elif fault=="native_order":
        adapter=data.train._unified_exit_lifecycle_v2;adapter.order=adapter.order[::-1].copy()
    else:
        coordinates=data.case["coordinates"];source=Path(coordinates["selected_sampler"]["path"])
        target=source.with_name("COPIED_SELECTED.json");target.write_bytes(source.read_bytes())
        coordinates["selected_sampler"]=_bind(target);data.case["seal_coordinates"]()
    with pytest.raises(RuntimeError,match={
        "source_bytes":"PHYSICAL_SESSION_SOURCE_MISMATCH",
        "native_order":"RESUME_ORDER_MISMATCH","coordinate_authority":"CONTRACT",
    }[fault]):
        h.run_prefix(output,4,expected_pointer=digest(h.pointer(output)))
    assert h.pointer(output).read_bytes()==pointer_before and h.batches==batches_before


def test_physical_current_design_cannot_inherit_historical_512_step_extension(physical_prepared,tmp_path):
    h=PrefixHarness(tmp_path/"physical-extension",physical_prepared);output=h.root/"FORBIDDEN"
    with pytest.raises(RuntimeError,match="PHYSICAL_PREFIX_CONTINUATION_FORBIDDEN"):
        h.run_prefix(output,257,chronological_continuation={
            "plan":{"from_optimizer_steps":256,"stop_after_optimizer_steps":512},
            "plan_binding":{"path":"/historical/continuation.json","sha256":"1"*64}})
    assert not h.pointer(output).exists() and h.batches==[]


def _horizon_eligible_case(data, tmp_path, *, eligible=None, declared_rows=None):
    """Coordinates v2 over an eligible subset of the label-bound TRAIN rows (operator decision 12.10.2026)."""
    from tests.test_native_fresh_prefix_components import _bind, _write
    from gx1.contracts.entry_model_native_training_run_lineage_v1 import deterministic_uniform_subsample_indices
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import HORIZON_ELIGIBILITY_RULE
    case = data.case
    full = np.load(data.design["calendar"]["bindings"]["TRAIN_CALENDAR_PARENT_ROWS"]["path"])
    eligible = np.setdiff1d(full, full[::97]) if eligible is None else eligible
    order = case["order"][np.isin(case["order"], eligible)]
    planned = order[:4096]
    probe = planned[deterministic_uniform_subsample_indices(population_rows=4096, requested_rows=256, seed=20260911, split_salt=0)]
    design_doc = _write(tmp_path / "horizons-design.json", {"schema_version": "fixture"})
    bindings = {}
    for name, rows in (("TRAIN_ELIGIBLE_PARENT_ROWS", eligible), ("TRAIN_NATIVE_EPOCH0_ORDER", order),
                       ("TRAIN_NATIVE4096_PARENT_ROWS", planned), ("TRAIN256_PROBE_PARENT_ROWS", probe)):
        path = tmp_path / ("v2-" + name + ".npy")
        np.save(path, rows)
        bindings[name] = _bind(path)
    coordinates = case["coordinates"]
    coordinates.update(schema_version="gx1_physical_native_training_coordinates_v2", bindings=bindings,
                       eligibility={"rule": HORIZON_ELIGIBILITY_RULE, "horizons_design": design_doc,
                                    "entry_eligibility_cutoff_utc": "2021-05-02T19:20:00+00:00",
                                    "eligible_rows": int(len(eligible) if declared_rows is None else declared_rows),
                                    "censored_rows_excluded": int(len(full) - len(eligible)),
                                    "complete_at_all_observation_horizons": True})
    case["seal_coordinates"]()
    data.train._unified_exit_lifecycle_v2.order = order.copy()
    return eligible, order


def test_physical_training_binding_trains_on_the_horizon_eligible_subset_of_label_bound_rows(physical_prepared, tmp_path):
    data = physical_prepared
    eligible, order = _horizon_eligible_case(data, tmp_path)
    full = np.flatnonzero(data.train._policy_dependent_auxiliary_bound_rows)
    assert len(eligible) < len(full) and data.train._policy_dependent_auxiliary_binding["rows"] == len(full)
    binding, parents, epoch_order = trainer._prefix_candidate_training_binding(**_binding_args(data))
    assert np.array_equal(parents.numpy(), eligible) and np.array_equal(epoch_order.numpy(), order)
    assert binding["train_rows"] == len(eligible)
    assert binding["train_parent_rows"] == data.case["coordinates"]["bindings"]["TRAIN_ELIGIBLE_PARENT_ROWS"]
    assert binding["label_parent_rows"] == data.design["calendar"]["bindings"]["TRAIN_CALENDAR_PARENT_ROWS"]
    assert binding["horizon_eligibility"]["eligible_rows"] == len(eligible)


@pytest.mark.parametrize("fault", ["outside_label_rows", "declared_count", "label_rows_shrunk"])
def test_physical_training_binding_rejects_eligible_rows_outside_the_label_binding(physical_prepared, tmp_path, fault):
    data = physical_prepared
    full = np.flatnonzero(data.train._policy_dependent_auxiliary_bound_rows)
    if fault == "outside_label_rows":
        _horizon_eligible_case(data, tmp_path, eligible=np.append(np.setdiff1d(full, full[::97]), len(full) + 5))
    elif fault == "declared_count":
        _horizon_eligible_case(data, tmp_path, declared_rows=7)
    else:
        _horizon_eligible_case(data, tmp_path)
        data.train._policy_dependent_auxiliary_bound_rows[full[3]] = False
    with pytest.raises(RuntimeError):
        trainer._prefix_candidate_training_binding(**_binding_args(data))


def _declare_synthetic_single_epoch(data, tmp_path, monkeypatch):
    """Real binding/coordinate validators over a named synthetic primary mask."""
    from tests.test_entry_observed_market_horizons_learning import _single_epoch_design
    from tests.test_unified_exit_selected_sampler_v1 import _direct_build
    from gx1.contracts import unified_exit_native_candidate_campaign_v1 as campaign
    from gx1.contracts.entry_observed_market_v1 import entry_observed_m5_path_contract, ENTRY_ONLY_M5_PATH_TRAINING_PHASE
    from gx1.scripts import materialize_unified_exit_chronological_coordinates_v1 as producer
    case = data.case
    part, witness = _single_epoch_design(tmp_path, rows=273,
        source=case["sources"]["train"]["parquet"], sampler=case["selected"]["selected_sampler_contract"],
        order=case["order"], cutoff=data.design["calendar"]["train_control_cutoff"])
    data.design["scope"].update(part["scope"])
    data.design["budget"].update(part["budget"])
    case["recipe"]["trainer_cli"]["epochs"] = 1
    case["recipe"]["entry_m5_path_single_epoch"] = part["scope"]["entry_m5_path_single_epoch"]
    case["seal"]()
    # A fresh synthetic CPU measurement receipt binds this exact design. The
    # sampler contract/order itself remains the original one-way input.
    receipt = case["sample_case"]["receipt"]
    receipt["run_bindings"]["files"]["chronological_design"] = data.value["design"]
    receipt["receipt_sha256"] = campaign.native_sha256({k:v for k,v in receipt.items() if k != "receipt_sha256"})
    _write(case["sample_case"]["receipt_path"], receipt)
    selected = _direct_build(case["sample_case"])
    selected_binding = _write(data.files["selected_sampler"], selected)
    rows = np.load(witness["eligible_parent_rows"]["path"])
    order = np.load(witness["epoch0_parent_order"]["path"])
    coords = case["coordinates"]
    coords.update(schema_version=campaign.PHYSICAL_COORDINATES_SCHEMA_M5_PATH,
        design=data.value["design"], selected_sampler=selected_binding,
        eligibility={"rule":campaign.M5_PATH_ELIGIBILITY_RULE, "training_phase":ENTRY_ONLY_M5_PATH_TRAINING_PHASE,
            "target_contract":entry_observed_m5_path_contract(),
            "entry_eligibility_cutoff_utc":data.design["calendar"]["train_control_cutoff"],
            "eligible_rows":len(rows), "censored_rows_excluded":len(data.train)-len(rows),
            "long_horizons_gate_primary_eligibility":False})
    from gx1.contracts.entry_model_native_training_run_lineage_v1 import deterministic_uniform_subsample_indices
    probe_positions = deterministic_uniform_subsample_indices(population_rows=len(rows), requested_rows=256,
        seed=data.design["selection"]["seed"], split_salt=0)
    for name, values in (("TRAIN_ELIGIBLE_PARENT_ROWS",rows), ("TRAIN_NATIVE_EPOCH0_ORDER",order),
            ("TRAIN_NATIVE4096_PARENT_ROWS",order), ("TRAIN256_PROBE_PARENT_ROWS",order[probe_positions])):
        path = tmp_path / (name + ".npy"); np.save(path, values, allow_pickle=False)
        coords["bindings"][name] = _bind(path)
    case["seal_coordinates"]()
    monkeypatch.setattr(producer, "m5_path_eligible_rows", lambda path, *, cutoff_utc: {
        "rows":rows, "physical_rows":len(data.train), "censored_rows_excluded":len(data.train)-len(rows)})
    for dataset, role, split in ((data.train,"TRAIN","train"),(data.control,"CONTROL256","val")):
        dataset._policy_dependent_auxiliary_binding.update(result=data.value["labels_result"], design=data.value["design"])
        dataset._entry_observed_market_binding = {"role":role,
            "physical_source":case["sources"][split], "identity":{"training_phase":ENTRY_ONLY_M5_PATH_TRAINING_PHASE}}
    previous_cohort = data.context["evaluation_cohort"]
    cohort = build_chronological_control_cohort(data.value["design"],
        source_index_binding=previous_cohort["source_index"],
        source_index_manifest_binding=previous_cohort["source_index_manifest"])
    data.context["evaluation_cohort"] = cohort
    data.context["state_factory"].factory_receipt["physical_control_cohort_sha256"] = cohort["cohort_sha256"]
    return order, part["budget"]["planned_optimizer_steps"]


def test_m5_single_epoch_exact_partial_batch_and_resume_stop_before_val(physical_prepared, tmp_path, monkeypatch):
    data = physical_prepared
    order, steps = _declare_synthetic_single_epoch(data, tmp_path, monkeypatch)
    h = PrefixHarness(tmp_path / "m5-epoch", data)
    # The component owner still knows the whole physical source; the coordinate
    # owner determines the independent primary-valid training population.
    direct, resumed = h.root/"DIRECT", h.root/"RESUMED"
    overrides = {"epochs":1, "effective_train_rows":len(order)}
    _, end = h.run_prefix(direct, steps, **overrides)
    expected = h.state(direct); batches = list(h.batches)
    contract = json.loads((h.pointer(direct).parent/trainer._CANDIDATE_TRAINING_CONTRACT_FILENAME).read_text())
    checkpoint_policy = contract["training"]["checkpoint_policy"]
    assert contract["training"]["epochs"] == checkpoint_policy["max_epochs"] == 1
    assert checkpoint_policy == expected["training_progress"]["checkpoint_selection"]["checkpoint_policy"]
    assert end["phase"] == "train" and end["epoch_index"] == 0 and not end["complete"]
    assert end["next_batch_offset"] == steps and end["global_optimizer_steps"] == steps
    assert [row for batch in batches for row in batch] == order.tolist()
    assert len({row for batch in batches for row in batch}) == len(order)
    assert len(batches[-1]) == 1 and h.validation_batches == 0
    h.batches.clear()
    h.run_prefix(resumed, steps-1, **overrides)
    h.run_prefix(resumed, steps, expected_pointer=digest(h.pointer(resumed)), **overrides)
    actual = h.state(resumed)
    for key in ("model_state","target_model_state","optimizer_state","weight_ema_state",
            "lr_scheduler_state","rng_state","epoch_order","training_progress"):
        equal_tree(expected[key], actual[key])
    assert h.batches == batches and h.validation_batches == 0
    h.batches.clear()
    h.run_prefix(resumed, steps, expected_pointer=digest(h.pointer(resumed)), **overrides)
    assert h.batches == [] and h.validation_batches == 0
    with pytest.raises(RuntimeError, match="PREFIX_FIXED_BUDGET_REQUIRED"):
        h.run_prefix(h.root/"EXTRA", steps+1, **overrides)
    with pytest.raises(RuntimeError, match="M5_SINGLE_EPOCH_SCOPE_INVALID"):
        h.run_prefix(h.root/"TWO_EPOCHS", steps, **{**overrides,"epochs":2})


@pytest.fixture
def dynamic_prefix_binding_unit(physical_prepared, tmp_path, monkeypatch):
    """Prefix join only: stub completed preprocessing/sampler owners, not admission.

    The dynamic witness reader and actual scalar/npy/hash joins remain real.
    No quote reconstruction, normalization fit or measured sampler is claimed.
    """
    import hashlib
    from tests.test_native_prefix_recipe import _dynamic_single_epoch_design
    from tests.test_native_fresh_prefix_components import TinyModel
    from gx1.contracts import entry_observed_market_horizons_v1 as horizons
    from gx1.contracts import unified_exit_native_candidate_campaign_v1 as campaign
    from gx1.contracts.entry_observed_market_v1 import DynamicM5TargetProvider, entry_observed_dynamic_m5_path_contract
    from gx1.time.session_detector import m5_decision_availability
    data = physical_prepared
    directory = tmp_path / "dynamic-prefix-unit"; directory.mkdir()
    part, witness = _dynamic_single_epoch_design(directory, rows=len(data.train))
    queries = trainer._dynamic_m5_model_kwargs(witness["training_phase"])["dynamic_m5_query_minutes"]
    target = entry_observed_dynamic_m5_path_contract()
    control_binding = data.design["calendar"]["bindings"]["CONTROL256_PARENT_ROWS"]
    sources, frames, proofs = {}, {}, {}
    for split, count in (("train", len(data.train)), ("val", len(data.control))):
        if split == "train":
            path = Path(witness["train_parquet"]["path"])
            frame = pd.read_parquet(path)
            manifest = json.loads(Path(witness["train_manifest"]["path"]).read_text())
        else:
            path = directory / "dynamic-val.parquet"
            frame = pd.DataFrame({"time": pd.date_range(part["calendar"]["train_control_cutoff"], periods=count, freq="5min")})
            manifest = {"output_data_path":str(path),"extra":{"rows":count}}
        frame[trainer.DYNAMIC_M5_POINTER_COLUMNS[0]] = np.arange(count, dtype=np.int64) + (0 if split == "train" else len(data.train))
        frame[trainer.DYNAMIC_M5_POINTER_COLUMNS[1]] = True
        frame.to_parquet(path, index=False)
        source = {"parquet":_bind(path), "manifest":_write(path.with_suffix(".manifest.json"), manifest),
            "physical_rows":count,
            "clock_sha256":hashlib.sha256(np.asarray(pd.DatetimeIndex(frame.time).as_unit("ns").asi8,dtype="<i8").tobytes()).hexdigest()}
        frames[split],sources[split] = frame,source
        proofs[split] = {"entry_m1_pointer_sha256":hashlib.sha256(frame.iloc[:,1].to_numpy(dtype="<i8").tobytes()).hexdigest(),
            "dynamic_m5_supervision":{"source_quote_binding":{"qualified_owner_stub":True},
                "index_binding":{"qualified_owner_stub":split}}}
    witness.update(train_parquet=sources["train"]["parquet"],train_manifest=sources["train"]["manifest"])
    part["scope"][horizons.M5_SINGLE_EPOCH_SCOPE_KEY] = _write(directory / "dynamic-population.json", witness)
    data.design["scope"].update(part["scope"]);data.design["budget"].update(part["budget"])
    data.design["calendar"].update(part["calendar"],source_bindings=sources)
    data.design.pop("targets",None)  # Dynamic binding must not manufacture an Exit policy.
    data.value["design"] = _write(Path(data.value["design"]["path"]),data.design)
    registry = _write(directory / "registry-clock.json",{
        "train_decision_rows":len(data.train),
        "train_decision_clock_sha256":hashlib.sha256(np.asarray(m5_decision_availability(pd.DatetimeIndex(frames["train"].time)).as_unit("ns").asi8,dtype="<i8").tobytes()).hexdigest()})
    receipt = {"schema_version":trainer.M5_DYNAMIC_PHYSICAL_TARGET_BINDING_SCHEMA,
        "design":data.value["design"],"registry":registry,"target_contract":target,"splits":proofs}
    data.value["labels_result"] = _write(directory / "receipt-boundary.json",receipt)
    full_rows = data.design["calendar"]["bindings"]["TRAIN_CALENDAR_PARENT_ROWS"]
    order = np.load(witness["epoch0_parent_order"]["path"],allow_pickle=False)
    coordinates = {"bindings":{"TRAIN_ELIGIBLE_PARENT_ROWS":witness["eligible_parent_rows"],
        "TRAIN_NATIVE_EPOCH0_ORDER":witness["epoch0_parent_order"]},
        "eligibility":{"rule":campaign.DYNAMIC_M5_PATH_ELIGIBILITY_RULE,
            "entry_eligibility_cutoff_utc":witness["entry_eligibility_cutoff_utc"],
            "eligible_rows":len(order),"censored_rows_excluded":0}}
    data.value["native_coordinates"] = _write(directory / "coordinate-boundary.json",coordinates)
    physical = {"physical_sources":sources,"train_parent_rows":full_rows,
        "base_contract_sha256":data.composite["base_feature_normalization"]["contract_sha256"],
        "entry_only_base_normalization":data.value["normalization_result"],
        "sequence_source_audits":{split:_write(directory / (split + "-audit-boundary.json"),{"split":split})
                                  for split in ("train","val")}}
    seen = []
    def preprocessing(artifacts, **kwargs):
        assert "targets" not in kwargs["design"]
        assert artifacts["labels_result"] == data.value["labels_result"]
        seen.append("qualified_preprocessing_boundary")
        return physical
    def coordinate_owner(artifacts, **kwargs):
        assert kwargs["physical"] is physical
        seen.append("qualified_coordinate_boundary")
        return {"artifact":coordinates,"binding":data.value["native_coordinates"],
            "population_witness":part["scope"][horizons.M5_SINGLE_EPOCH_SCOPE_KEY],
            "native_sampler_contract":witness["native_sampler_contract"]}
    monkeypatch.setattr(campaign,"require_physical_chronological_preprocessing",preprocessing)
    monkeypatch.setattr(campaign,"require_physical_native_training_coordinates",coordinate_owner)
    registry_clock = {"rows":len(data.train),
        "train_decision_clock_sha256":json.loads(Path(registry["path"]).read_text())["train_decision_clock_sha256"],"exact_match":True}
    class Dataset(Rows):
        pass
    identity = {"design":data.value["design"],"target_contract":target,"training_phase":witness["training_phase"],
        "query_minutes":list(queries),"economics":{"qualified_owner_stub":True}}
    expected_columns = list(dict.fromkeys([*trainer.DYNAMIC_M5_POINTER_COLUMNS,*trainer._FORECAST_TARGET_COLS,
        *trainer.MODEL_NATIVE_FORECAST_VALID_COLUMNS,*horizons.HORIZON_LABEL_COLUMNS]))
    for split,role in (("train","TRAIN"),("val","CONTROL256")):
        dataset = Dataset(sources[split]["physical_rows"])
        dataset.df = frames[split];dataset.parquet_path = Path(sources[split]["parquet"]["path"])
        row_binding = full_rows if split == "train" else control_binding
        rows = np.load(row_binding["path"],allow_pickle=False)
        mask = np.zeros(len(dataset),dtype=np.bool_);mask[rows] = True
        dataset._policy_dependent_auxiliary_bound_rows = mask
        dataset._entry_observed_market_admitted_rows = mask.copy()
        proof = proofs[split]
        dataset._policy_dependent_auxiliary_binding = {"result":data.value["labels_result"],"design":data.value["design"],
            "role":role,"mode":trainer.DYNAMIC_M5_PHYSICAL_BINDING_MODE,"source_split":split,
            "emission_scope":trainer.M5_DYNAMIC_OBSERVED_EMISSION_SCOPE,"training_phase":witness["training_phase"],
            "target_contract":target,"parent_entry_parquet":sources[split]["parquet"],"parent_entry_manifest":sources[split]["manifest"],
            "row_binding":row_binding,"rows":len(rows),"active_target_columns":expected_columns,
            "targets_rewritten":False,"inactive_diagnostics_refreshed":False,"registry":registry,
            "entry_m1_pointer_sha256":proof["entry_m1_pointer_sha256"],"whole_train_registry_clock":dict(registry_clock)}
        provider = DynamicM5TargetProvider.__new__(DynamicM5TargetProvider)
        provider.index = SimpleNamespace(binding=lambda value=proof["dynamic_m5_supervision"]["index_binding"]:value)
        provider.fill_surface = pd.DataFrame({"entry_m1_row":dataset.df.iloc[:,1].to_numpy().copy()})
        provider.admitted_rows = mask.copy();provider.query_minutes = queries;provider.economics = identity["economics"]
        def forbidden_query(*args,**kwargs):raise AssertionError("Prefix binding must not query future outcomes")
        provider.targets_for_rows = forbidden_query
        dataset._entry_dynamic_m5_target_provider = provider
        dataset._entry_observed_market_binding = {"identity":copy.deepcopy(identity),"physical_source":sources[split],"role":role,
            **proof["dynamic_m5_supervision"],"dense_future_paths_materialized":False,"admitted_rows":int(mask.sum()),
            "admitted_row_indices_sha256":hashlib.sha256(np.asarray(rows,dtype="<i8").tobytes()).hexdigest(),
            "first_prefix_mask_sha256":hashlib.sha256(dataset.df.iloc[:,2].to_numpy().tobytes()).hexdigest()}
        setattr(data,"train" if split == "train" else "control",dataset)
        data.files["entry_"+split+"_parquet"] = dataset.parquet_path
    args = _binding_args(data)
    args["model"] = TinyModel(args["input_normalization"],**trainer._dynamic_m5_model_kwargs(witness["training_phase"]))
    return SimpleNamespace(data=data,args=args,coordinates=coordinates,witness=witness,seen=seen)


def test_dynamic_prefix_binding_unit_joins_real_witness_clock_pointer_and_roles_without_exit(dynamic_prefix_binding_unit):
    case = dynamic_prefix_binding_unit
    binding,parents,order = trainer._prefix_candidate_training_binding(**case.args)
    assert case.seen == ["qualified_preprocessing_boundary","qualified_coordinate_boundary"]
    assert not hasattr(case.data.train,"_unified_exit_lifecycle_v2") and "targets" not in case.data.design
    assert np.array_equal(parents.numpy(),np.load(case.witness["eligible_parent_rows"]["path"]))
    assert np.array_equal(order.numpy(),np.load(case.witness["epoch0_parent_order"]["path"]))
    assert binding["train_rows"] == case.witness["train_rows"]
    assert binding["maximum_optimizer_steps"] == case.witness["optimizer_steps"]
    assert binding["entry_training_phase"] == case.witness["training_phase"]
    assert binding["candidate_quality_native_authority_admitted"] is False
    assert "selected_sampler" not in binding and "composite_normalization_sha256" not in binding
    assert binding["entry_only_base_normalization"] == case.data.value["normalization_result"]
    assert set(binding["sequence_source_audits"]) == {"train", "val"}
    assert binding["entry_epoch_order_authority"] == {
        "population_witness":case.data.design["scope"]["entry_m5_path_single_epoch"],
        "native_sampler_contract":case.witness["native_sampler_contract"]}
    assert "dynamic_m5_eligibility" in binding and "horizon_eligibility" not in binding
    assert binding["active_joint_task_names"] == list(trainer._model_joint_task_names(case.args["model"]))


@pytest.mark.parametrize("fault",["legacy_mode","pointer","clock","provider_mask","row_binding","registry_clock","phase","tasks","model_queries","order"])
def test_dynamic_prefix_binding_unit_rejects_role_pointer_clock_model_or_order_drift(dynamic_prefix_binding_unit,fault):
    case = dynamic_prefix_binding_unit;dataset = case.data.train
    if fault == "legacy_mode": dataset._policy_dependent_auxiliary_binding["mode"] = "original_physical_split_targets"
    elif fault == "pointer": dataset.df.iloc[0,1] += 1
    elif fault == "clock": dataset.df.loc[0,"time"] -= pd.Timedelta(minutes=5)
    elif fault == "provider_mask": dataset._entry_dynamic_m5_target_provider.admitted_rows[0] = False
    elif fault == "row_binding": dataset._policy_dependent_auxiliary_binding["row_binding"] = case.data.design["calendar"]["bindings"]["CONTROL256_PARENT_ROWS"]
    elif fault == "registry_clock": dataset._policy_dependent_auxiliary_binding["whole_train_registry_clock"]["rows"] -= 1
    elif fault == "phase": case.args["model"].entry_training_phase = trainer.JOINT_FITTED_Q_TRAINING_PHASE
    elif fault == "tasks": case.args["model"].task_log_variances.pop(next(iter(case.args["model"].task_log_variances)))
    elif fault == "model_queries": case.args["model"].dynamic_m5_query_horizons[0] += 1
    else:
        path = Path(case.coordinates["bindings"]["TRAIN_NATIVE_EPOCH0_ORDER"]["path"]).with_name("changed-order.npy")
        order = np.load(case.witness["epoch0_parent_order"]["path"])[::-1].copy();np.save(path,order,allow_pickle=False)
        case.coordinates["bindings"]["TRAIN_NATIVE_EPOCH0_ORDER"] = _bind(path)
    with pytest.raises(RuntimeError):trainer._prefix_candidate_training_binding(**case.args)



def test_dynamic_epoch0_reuses_verified_prefix_order_without_exit_or_random_fallback(dynamic_prefix_binding_unit,monkeypatch):
    case = dynamic_prefix_binding_unit
    binding,parents,frozen = trainer._prefix_candidate_training_binding(**case.args)
    def forbidden(*args,**kwargs):raise AssertionError("Dynamic native order must not sample a new order")
    monkeypatch.setattr(torch,"randperm",forbidden)
    actual = trainer._candidate_training_epoch_order(case.data.train,epoch_index=0,parent_population=parents)
    assert torch.equal(actual,frozen)
    assert torch.equal(trainer._candidate_training_epoch_order(case.data.train,epoch_index=0,parent_population=parents),frozen)
    cached = case.data.train._entry_dynamic_m5_epoch_order_binding
    assert type(cached["prefix_epoch0_order"]) is tuple
    assert cached["entry_epoch_order_authority"] == binding["entry_epoch_order_authority"]
    assert not hasattr(case.data.train,"_unified_exit_lifecycle_v2")


@pytest.mark.parametrize("fault",["unbound","epoch_one","missing_parents","wrong_parents","role","source","exit_adapter", "cache_order","cache_witness","clock","mask","mask_dtype","phase","witness_bytes"])
def test_dynamic_epoch0_rejects_wrong_phase_role_source_witness_or_parent_order(dynamic_prefix_binding_unit,fault):
    case = dynamic_prefix_binding_unit;dataset = case.data.train
    if fault == "unbound": parents = torch.arange(len(dataset),dtype=torch.int64)
    else: _,parents,_ = trainer._prefix_candidate_training_binding(**case.args)
    epoch = 0
    if fault == "epoch_one":epoch = 1
    elif fault == "missing_parents":parents = None
    elif fault == "wrong_parents":parents = parents[:-1]
    elif fault == "role":dataset._policy_dependent_auxiliary_binding["role"] = "CONTROL256"
    elif fault == "source":dataset.parquet_path = case.data.control.parquet_path
    elif fault == "exit_adapter":dataset._unified_exit_lifecycle_v2 = object()
    elif fault == "cache_order":
        cached = dataset._entry_dynamic_m5_epoch_order_binding
        cached["prefix_epoch0_order"] = cached["prefix_epoch0_order"][::-1]
    elif fault == "cache_witness":dataset._entry_dynamic_m5_epoch_order_binding["entry_epoch_order_authority"]["population_witness"]["sha256"] = "f"*64
    elif fault == "clock":dataset.df.loc[0,"time"] -= pd.Timedelta(minutes=5)
    elif fault == "mask":dataset._entry_observed_market_admitted_rows[0] = False
    elif fault == "mask_dtype":dataset._policy_dependent_auxiliary_bound_rows = dataset._policy_dependent_auxiliary_bound_rows.astype(np.int64)
    elif fault == "phase":dataset._entry_observed_market_binding["identity"]["training_phase"] = trainer.JOINT_FITTED_Q_TRAINING_PHASE
    elif fault == "witness_bytes":Path(dataset._entry_dynamic_m5_epoch_order_binding["entry_epoch_order_authority"]["population_witness"]["path"]).write_bytes(b"changed witness")
    with pytest.raises(RuntimeError):
        trainer._candidate_training_epoch_order(dataset,epoch_index=epoch,parent_population=parents)


@pytest.mark.parametrize("fault",["order","design","extra"])
def test_dynamic_prefix_epoch_order_proof_allows_exact_rebind_and_rejects_tamper(dynamic_prefix_binding_unit,fault):
    case = dynamic_prefix_binding_unit
    trainer._prefix_candidate_training_binding(**case.args)
    trainer._prefix_candidate_training_binding(**case.args)
    cached = case.data.train._entry_dynamic_m5_epoch_order_binding
    if fault == "order":cached["prefix_epoch0_order"] = cached["prefix_epoch0_order"][::-1]
    elif fault == "design":cached["design"]["sha256"] = "f"*64
    else:cached["unbound_field"] = True
    before = copy.deepcopy(cached)
    with pytest.raises(RuntimeError,match="CANDIDATE_DYNAMIC_EPOCH_ORDER_REBIND_FORBIDDEN"):
        trainer._prefix_candidate_training_binding(**case.args)
    assert case.data.train._entry_dynamic_m5_epoch_order_binding == before


@pytest.fixture(scope="module")
def dynamic_context_normalization_contract():
    from tests.model_native_input_normalization_support import input_normalization_fixture
    return input_normalization_fixture(signal_names=["ordinary"], mtf_names=["ordinary_mtf"], rows=8,
        entry_training_phase=trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)


@pytest.fixture
def dynamic_context_unit(tmp_path, monkeypatch, dynamic_context_normalization_contract):
    """Real normalization contract; the separately tested cohort owner is a unit boundary."""
    from gx1.contracts import unified_exit_bounded_val_cohort_v1 as cohorts
    from gx1.contracts.entry_observed_market_v1 import entry_observed_dynamic_m5_path_contract
    norm = copy.deepcopy(dynamic_context_normalization_contract)
    lineage = norm["lineage"]
    sources = {"train":{"parquet":{"path":lineage["train_parquet_path"],"sha256":lineage["train_parquet_sha256"]},
        "manifest":{"path":lineage["train_manifest_path"],"sha256":lineage["train_manifest_sha256"]},
        "physical_rows":lineage["entry_train_decision_row_count"],"clock_sha256":"4"*64},
        "val":{"parquet":{"path":"/fixture/val.parquet","sha256":"5"*64},
        "manifest":{"path":"/fixture/val.manifest.json","sha256":"6"*64},"physical_rows":8,"clock_sha256":"7"*64}}
    plan = _write(tmp_path / "design-unit.json", {"unit_boundary":True})
    audit = _write(tmp_path / "train-audit.json", {"unit_boundary":True,"split":"train"})
    base = _write(tmp_path / "base.json", {"schema_version":"gx1_entry_only_dynamic_m5_base_normalization_v1",
        "decision":"PASS","entry_training_phase":trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        "native_training_admitted":False,"contract":norm,"contract_sha256":norm["contract_sha256"],"sequence_source_audit":audit})
    identity = {"training_phase":trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,"design":plan,
        "query_minutes":list(trainer._dynamic_m5_model_kwargs(trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)["dynamic_m5_query_minutes"]),
        "target_contract":entry_observed_dynamic_m5_path_contract()}
    cohort = {"training_phase":trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,"source_split":"train","plan":plan,
        "parent_entry_parquet":sources["train"]["parquet"],"parent_entry_manifest":sources["train"]["manifest"],
        "population_rows":sources["train"]["physical_rows"],"source_clock_sha256":sources["train"]["clock_sha256"],"exit_evaluation":False}
    calls=[]
    def checked(value):calls.append(value);return value
    monkeypatch.setattr(cohorts,"require_bounded_val_cohort",checked)
    context={"entry_training_phase":trainer.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        "entry_supervision_identity":identity,"physical_sources":sources,
        "entry_only_base_normalization":base,"base_contract_sha256":norm["contract_sha256"],"evaluation_cohort":cohort,
        "val_sequence_audit":audit,"max_model_forwards":16,"max_state_views":256,"max_wall_seconds":120,"progress_interval_forwards":16}
    return SimpleNamespace(context=context,calls=calls)


def test_dynamic_native_context_binds_entry_inputs_without_exit_factory(dynamic_context_unit):
    case=dynamic_context_unit
    bound=trainer._native_candidate_val_context_binding(case.context)
    assert case.calls == [case.context["evaluation_cohort"]]
    assert bound["entry_only_base_normalization"] == case.context["entry_only_base_normalization"]
    assert bound["base_contract_sha256"] == case.context["base_contract_sha256"]
    assert bound["sequence_source_audit"] == case.context["val_sequence_audit"]
    assert bound["physical_sources"] == case.context["physical_sources"]
    assert bound["exit_evaluation"] is bound["native_launch_authorized"] is False
    assert "factory_receipt" not in bound and "entry_q_target_semantics" not in bound


@pytest.mark.parametrize("fault", ["legacy_factory","clock","source","phase","target","query","normhash","audit","limit"])
def test_dynamic_native_context_rejects_source_phase_normalization_or_legacy_mixing(dynamic_context_unit,fault):
    context=copy.deepcopy(dynamic_context_unit.context)
    if fault=="legacy_factory":context["state_factory"]=object()
    elif fault=="clock":context["evaluation_cohort"]["source_clock_sha256"]="9"*64
    elif fault=="source":context["evaluation_cohort"]["parent_entry_parquet"]=context["physical_sources"]["val"]["parquet"]
    elif fault=="phase":context["entry_supervision_identity"]["training_phase"]="joint_fitted_q"
    elif fault=="target":context["entry_supervision_identity"]["target_contract"]={}
    elif fault=="query":context["entry_supervision_identity"]["query_minutes"]=[1]
    elif fault=="normhash":context["base_contract_sha256"]="9"*64
    elif fault=="audit":context["val_sequence_audit"]["sha256"]="9"*64
    else:context["max_wall_seconds"]=True
    with pytest.raises(RuntimeError):trainer._native_candidate_val_context_binding(context)


def test_dynamic_session_contract_seals_base_audits_and_witness_without_exit_manifest(tmp_path,dynamic_prefix_binding_unit):
    from tests.test_candidate_training_session import _dynamic_session_contract_arguments
    case=dynamic_prefix_binding_unit
    prefix,_,_=trainer._prefix_candidate_training_binding(**case.args)
    args=_dynamic_session_contract_arguments(tmp_path,prefix=prefix,input_normalization=case.args["input_normalization"])
    contract=trainer._candidate_training_session_contract(**args)
    assert "unified_exit_lifecycle_manifest" not in contract["artifacts"]
    assert contract["artifacts"]["entry_only_base_normalization"] == prefix["entry_only_base_normalization"]
    for split in ("train","val"):
        assert contract["artifacts"][split+"_sequence_source_audit"] == prefix["sequence_source_audits"][split]
    assert "exit_reference_policy" not in contract["training"]
    assert contract["training"]["reference_cutoff_time_ns"] == prefix["reference_cutoff_time_ns"]
    assert contract["training"]["checkpoint_policy"]["checkpoint_monitor"] == trainer.DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR
    assert contract["authority"]["bundle"] is contract["authority"]["test"] is False


@pytest.mark.parametrize("fault",["exitpolicy","exitmanifest","backup","missing_audit","changed_base","legacy_without_exitpolicy","economic_monitor"])
def test_dynamic_session_contract_rejects_exit_mixing_or_changed_source_seals(tmp_path,dynamic_prefix_binding_unit,fault):
    from tests.test_candidate_training_session import _dynamic_session_contract_arguments
    case=dynamic_prefix_binding_unit
    prefix,_,_=trainer._prefix_candidate_training_binding(**case.args)
    args=_dynamic_session_contract_arguments(tmp_path,prefix=prefix,input_normalization=case.args["input_normalization"])
    if fault=="exitpolicy":args["exit_reference_policy"]={"fake":True}
    elif fault=="exitmanifest":args["lifecycle_manifest_path"]=args["train_parquet"]
    elif fault=="backup":args["exit_backup_steps"]=5
    elif fault=="missing_audit":prefix["sequence_source_audits"].pop("val")
    elif fault=="changed_base":Path(prefix["entry_only_base_normalization"]["path"]).write_bytes(b"changed")
    elif fault=="economic_monitor":args["checkpoint_monitor"]=MARKED_NET_CHECKPOINT_MONITOR
    else:prefix["entry_training_phase"]="joint_fitted_q"
    with pytest.raises(RuntimeError):trainer._candidate_training_session_contract(**args)


def test_dynamic_progress_preserves_empty_fixed_boundary_selection(dynamic_prefix_binding_unit):
    case=dynamic_prefix_binding_unit
    prefix,_,_=trainer._prefix_candidate_training_binding(**case.args)
    kwargs={"checkpoint_monitor":trainer.DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR,"prefix_training_binding":prefix}
    progress=trainer._new_candidate_training_progress(**kwargs)
    selected=progress["checkpoint_selection"]
    assert selected["checkpoint_policy"]["save_top_k"] == 0
    assert selected["best_val"] is selected["best_policy_pnl"] is selected["best_unique_target_action_agreement"] is None
    assert selected["top_k_checkpoints"] == [] and progress["validation_snapshot"] is None
    assert trainer._require_candidate_training_progress(progress,**kwargs) == progress
    # Genuine task gradient/observation evidence may advance independently of selection.
    for flags in ("joint_task_supervision_observed","joint_task_gradient_observed"):
        progress[flags]={name:True for name in progress[flags]}
    assert trainer._require_candidate_training_progress(progress,**kwargs) == progress


@pytest.mark.parametrize("fault",["score","topk","best_model","earlystop","valstate","valepoch"])
def test_dynamic_progress_rejects_any_adaptive_or_validation_selection(dynamic_prefix_binding_unit,fault):
    case=dynamic_prefix_binding_unit
    prefix,_,_=trainer._prefix_candidate_training_binding(**case.args)
    kwargs={"checkpoint_monitor":trainer.DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR,"prefix_training_binding":prefix}
    progress=trainer._new_candidate_training_progress(**kwargs)
    selection=progress["checkpoint_selection"]
    if fault=="score":selection["best_val"]=0.
    elif fault=="topk":selection["top_k_checkpoints"]=[{}]
    elif fault=="best_model":selection["best_state"]={}
    elif fault=="earlystop":selection["early_stopped"]=True
    elif fault=="valepoch":selection["last_epoch"]=1
    else:progress["validation_snapshot"]={}
    with pytest.raises(RuntimeError,match="SELECTION_STATE_INVALID"):
        trainer._require_candidate_training_progress(progress,**kwargs)
