"""Bounded synthetic recipe/campaign integration, not model learning evidence."""
import copy
import json
from pathlib import Path
import pytest
import torch
from gx1.contracts import unified_exit_native_candidate_campaign_v1 as native
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.scripts import run_unified_exit_random_access_full_train_v1 as runner
from gx1.scripts import materialize_local_random_access_campaign_v2 as materializer
from gx1.scripts import run_unified_exit_native_candidate_window_v1 as window
from gx1.contracts.entry_candidate_checkpoint_policy_v1 import native_checkpoint_monitor
from tests.test_native_learning_calibration_scope import scope
from tests.test_native_fresh_prefix_components import _bindings,_bind,_write
from tests.test_local_random_access_campaign_v2 import _boot
from tests.test_unified_exit_random_access_state_view_v1 import _objective


@pytest.fixture
def prefix_scope(scope,tmp_path,monkeypatch):
    policy,recipe,write,save=scope
    directory=tmp_path/'artifacts';directory.mkdir()
    value,files,design,norm,labels,prep,_=_bindings(directory)
    design['scope']['one_experiment']=True
    design['budget'].update(epochs_completed=0,repeat_or_automatic_extension=False)
    value['design']=_write(Path(value['design']['path']),design)
    prep['frozen_design']=value['design'];pb=_write(Path(norm['prefix_preparation']['path']),prep)
    norm['frozen_design']=value['design'];norm['prefix_preparation']=pb
    labels['prefix_preparation']=pb
    value['normalization_result']=_write(Path(value['normalization_result']['path']),norm)
    value['labels_result']=_write(Path(value['labels_result']['path']),labels)
    files['economics_readiness']=Path(recipe['files']['economics_readiness']['path'])
    _write(files['economics_readiness'],{'economics_objective_contract':_objective('liquidation_advantage_v1')})
    controls={'epochs':30,'batch_size':16,'num_workers':0,'grad_accum_steps':1,
              'early_stopping_patience':5,'early_stopping_min_delta':0.,'minimum_epochs_before_stop':1,'save_top_k':1,
              'precision_policy':'deterministic_fp32','seed':20260911,'learning_rate':.0001,'weight_decay':.0001,
              'grad_clip_norm':float(runner.trainer._GRAD_CLIP_NORM),
              'checkpoint_monitor':native_checkpoint_monitor(json.loads(files['economics_readiness'].read_text())['economics_objective_contract'])}
    policy.pop('native_learning_calibration');recipe.pop('candidate_resume_origin')
    recipe.update(schema_version=runner.NATIVE_FULL_TRAIN_RECIPE_SCHEMA,profile='candidate',source_repo=str(tmp_path),
        source_commit='c'*40,source_bindings={},source_bindings_sha256=runner.val.canonical_sha256({}),
        run_id='PREFIX_ONCE',dataset_run_id='PREFIX_DATA',gx1_data_root=str(tmp_path/'data'),
        out_bundle_dir=str(tmp_path/'out'/'MODEL'),files={k:_bind(p) for k,p in files.items()},
        seed_launch=None,seed_authority=None,smoke_full_val=None,trainer_cli=controls,recipe_env={},
        initialization=design['initialization']['mode'],test_data_used=False,chronological_prefix=value,
        exit_reference_policy=design['targets']['reference_policy'])
    recipe['val_limits']={**recipe['val_limits'],'max_model_forwards':100000,'max_state_views':1000000}
    manifest={'entry_row_count':5000,'parent_entry_source_rows':5000}
    manifest['manifest_sha256']=native.native_sha256(manifest)
    mb=write('index-manifest.json',manifest)
    root={'decision':'PASS','allowed_splits':['train','val'],'test_accessed':False,
          'splits':{'train':{'manifest_path':mb['path'],'manifest_sha256':manifest['manifest_sha256']}}}
    root['root_sha256']=native.native_sha256(root)
    recipe['files']['random_access_root']=_write(files['random_access_root'],root)
    policy['chronological_learning_run']={'chronological_prefix':value,'run_id':recipe['run_id'],
        'out_bundle_dir':recipe['out_bundle_dir'],'source_bindings_sha256':recipe['source_bindings_sha256'],
        'optimizer_steps':256,'maximum_trained_entry_rows':4096,'max_invocations':3,'final_model_variant':'ONLINE',
        'teacher_refresh_allowed':False,'full_epoch_training_allowed':False,'full_val_allowed':False,'test_data_used':False}
    def seal():
        save();recipe['recipe_sha256']=runner.val.canonical_sha256({k:v for k,v in recipe.items() if k!='recipe_sha256'})
        return write('recipe.json',recipe)
    seal()
    return policy,recipe,files,seal


def test_prefix_scope_is_finite_fixed_and_separate_from_full_training(prefix_scope):
    policy,recipe,files,seal=prefix_scope
    rb=seal()
    checked,count=native.require_native_recipe_metadata(rb,source_repo=Path(recipe['source_repo']),source_commit=recipe['source_commit'])
    assert checked==recipe and count==4500
    for n in (1,2,3):assert native.require_native_run_scope(recipe,invocation_number=n)==256
    assert native.native_completed_val_ceiling(recipe) is None and policy['training_enabled'] is False
    for n in (0,4,True):
        with pytest.raises(RuntimeError,match='PREFIX_INVOCATION'):native.require_native_run_scope(recipe,invocation_number=n)
    budget={'stop_after_optimizer_steps':256,'stop_after_completed_val_epochs':None,'max_invocation_seconds':12000}
    assert native.require_native_run_scope(recipe,execution_budget=budget)==256
    for key,bad in [('stop_after_optimizer_steps',255),('stop_after_optimizer_steps',257),
                    ('stop_after_optimizer_steps',256.),('stop_after_completed_val_epochs',1),
                    ('max_invocation_seconds',12001),('resume_probe_val_rows',32)]:
        with pytest.raises(RuntimeError,match='PREFIX_BUDGET'):native.require_native_run_scope(recipe,execution_budget={**budget,key:bad})


@pytest.mark.parametrize('fault',['missing_authority','full_training','full_val','refresh','extra_steps','output','source',
                                  'old_seed','old_smoke','old_origin','readout','backup','changed_artifact'])
def test_prefix_admission_rejects_mixed_initialization_and_scope(prefix_scope,fault):
    policy,recipe,files,seal=prefix_scope
    if fault=='missing_authority':policy.pop('chronological_learning_run')
    elif fault=='full_training':policy['training_enabled']=True
    elif fault=='full_val':policy['chronological_learning_run']['full_val_allowed']=True
    elif fault=='refresh':policy['chronological_learning_run']['teacher_refresh_allowed']=True
    elif fault=='extra_steps':policy['chronological_learning_run']['optimizer_steps']=257
    elif fault=='output':recipe['out_bundle_dir']+='changed'
    elif fault=='source':recipe['source_bindings_sha256']='a'*64
    elif fault=='old_seed':recipe['seed_launch']={'path':'old','sha256':'a'*64}
    elif fault=='old_smoke':recipe['smoke_full_val']={'path':'old','sha256':'a'*64}
    elif fault=='old_origin':recipe['candidate_resume_origin']={}
    elif fault=='readout':recipe['frozen_readout_evaluation']={}
    elif fault=='backup':recipe['exit_backup_steps']=5
    elif fault=='changed_artifact':Path(recipe['chronological_prefix']['normalization_result']['path']).write_text('{}')
    seal()
    with pytest.raises(RuntimeError):native.require_native_run_scope(recipe,invocation_number=1)


def test_existing_full_recipe_validator_uses_prefix_artifacts_without_historical_loaders(prefix_scope,monkeypatch):
    _,recipe,files,seal=prefix_scope;rb=seal();root=Path(recipe['source_repo'])
    monkeypatch.setattr(runner,'__file__',str(root/'gx1/scripts/runner.py'))
    monkeypatch.setattr(runner,'_native_recipe_source_bindings',lambda repo:{})
    clean=[];monkeypatch.setattr(runner.val,'_assert_clean_source',lambda r:clean.append(r['source_commit']))
    monkeypatch.setattr(runner.trainer,'require_model_native_recipe_env',lambda v:v)
    monkeypatch.setattr(runner.trainer,'_resolve_train_out_bundle_dir',lambda p,d:p)
    monkeypatch.setattr(runner.launch_owner,'require_training_recipe_source_provenance_metadata',lambda d,**kw:d)
    def forbidden(*a,**kw):raise AssertionError('historical seed/smoke loader reached')
    monkeypatch.setattr(runner.val,'require_launch_manifest',forbidden)
    monkeypatch.setattr(runner.val,'_load_final_authority',forbidden)
    monkeypatch.setattr(runner.val,'_load_val_economics_readiness',lambda p:json.loads(Path(p).read_text()))
    checked,bound,provenance=runner._require_native_full_train_recipe(Path(rb['path']),rb['sha256'])
    assert checked==recipe and bound==files and clean==['c'*40]
    assert provenance['recipe_audit_sha256']==rb['sha256']
    recipe['trainer_cli']['epochs']=31;rb=seal()
    with pytest.raises(RuntimeError,match='CONTROLS_MISMATCH'):runner._require_native_full_train_recipe(Path(rb['path']),rb['sha256'])


def test_guarded_dispatch_passes_fresh_prefix_to_existing_coordinator(prefix_scope,monkeypatch,tmp_path):
    _,recipe,files,seal=prefix_scope;rb=seal();seen={}
    budget={'stop_after_optimizer_steps':256,'stop_after_completed_val_epochs':None,'max_invocation_seconds':12000}
    monkeypatch.setattr(runner.trainer,'_require_cuda_trainer_guard_execution',lambda **kw:seen.update(guard=kw))
    monkeypatch.setattr(runner.trainer,'_resolve_device',lambda _:torch.device('cpu'))
    monkeypatch.setattr(runner,'_require_native_full_train_recipe',lambda *a:(recipe,files,{}))
    monkeypatch.setattr(runner.launch_owner,'require_candidate_execution_budget',lambda *a,**kw:budget)
    def components(**kw):
        assert kw['seed_launch_path'] is kw['seed_authority_path'] is kw['seed_authority_file_sha256'] is None
        assert kw['chronological_prefix']==recipe['chronological_prefix'];seen['components']=kw
        return {'chronological_prefix':kw['chronological_prefix']}
    monkeypatch.setattr(runner,'_build_bound_full_train_components',components)
    def train(**kw):
        assert kw['components']['chronological_prefix']==recipe['chronological_prefix']
        assert kw['candidate_resume_origin'] is None and kw['execution_budget']['stop_after_optimizer_steps']==256
        seen['train']=kw
        raise runner.trainer._CandidateExecutionPaused({'reason':'optimizer_step_ceiling','global_optimizer_steps':256})
    monkeypatch.setattr(runner,'_run_bound_full_train_candidate',train)
    receipt=tmp_path/'pause.json';receipt.write_text('{}')
    monkeypatch.setattr(runner.trainer,'_write_candidate_execution_pause_receipt',lambda *a,**kw:receipt)
    monkeypatch.setattr(runner,'_native_resume_state',lambda **kw:{'fixture':True})
    def forbidden(*a,**kw):raise AssertionError('historical smoke or VAL reached')
    monkeypatch.setattr(runner.val,'_read',forbidden)
    monkeypatch.setattr(runner,'_run_native_calibration_validation',forbidden)
    result=runner.run_guarded_native_candidate_invocation(recipe_path=Path(rb['path']),recipe_file_sha256=rb['sha256'],
        execution_budget_path=tmp_path/'budget.json',execution_budget_file_sha256='b'*64)
    assert result['decision']=='PAUSED_RESUMABLE' and result['bundle_written'] is False
    assert seen['guard']=={'execution_tier':'canonical'} and 'train' in seen


def _cursor(tmp_path,recipe_binding,steps):
    d=tmp_path/'synthetic-session';d.mkdir(exist_ok=True)
    state=d/'candidate_training_state_slot_0.pt';state.write_bytes(b'synthetic-no-model')
    raw={'schema_version':'gx1_candidate_training_session_v1','slot':0,'state_sha256':native.file_sha256(state),
         'session_contract_sha256':'a'*64,'phase':'train','epoch_index':0,'next_batch_offset':steps,
         'global_optimizer_steps':steps,'complete':False}
    pb=_write(d/'CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json',raw)
    position={k:raw[k] for k in ('session_contract_sha256','phase','epoch_index','next_batch_offset','global_optimizer_steps','complete')}
    position.update(training_pointer=pb,training_state=_bind(state),active_val_cursor=None,
                    active_val_model_forwards=0,epoch_schedule_sha256='b'*64)
    cursor=native.build_native_cursor(recipe=recipe_binding,resume_state=position,outcome='RESUMABLE')
    return position,_write(tmp_path/'cursor.json',cursor)


def test_native_campaign_materialization_window_and_final_review_stop(prefix_scope,monkeypatch,tmp_path):
    _,recipe,files,seal=prefix_scope;rb=seal();repo=Path(recipe['source_repo'])
    monkeypatch.setattr(materializer,'_source_commit',lambda _:recipe['source_commit'])
    guards={};controllers={}
    for names,target in [(('runner','guard','query','certificate'),guards),(('controller','observer','campaign_cli'),controllers)]:
        for name in names:
            path=repo/'sources'/name;path.parent.mkdir(exist_ok=True);path.write_text(name);target[name]=_bind(path)
    (repo/'scripts').mkdir();(repo/'scripts/gx1_capped_run.sh').write_text('synthetic')
    (repo/'.venv/bin').mkdir(parents=True);(repo/'.venv/bin/python').write_text('synthetic')
    monkeypatch.setattr(materializer,'_sources',lambda *a:(guards,controllers))
    boot=tmp_path/'boot.json';_write(boot,_boot(100,0))
    selection={'selected_batch_size':16,'artifact_sha256':'e'*64,'entry_pairs_per_epoch':16384,
               'transition_budget_per_epoch':65536,'total_batches_per_epoch':1024,'test_data_used':False}
    sb=_write(tmp_path/'selection.json',selection)
    prior={'fixture':'historical-hardware-only','phase':'full_val','selection_receipt':sb,'source_commit':'d'*40}
    pb=_write(tmp_path/'prior.json',prior)
    real_require=campaign.require_plan
    def plans(value,**kw):return prior if value.get('fixture')==prior['fixture'] else real_require(value,**kw)
    monkeypatch.setattr(campaign,'require_plan',plans);monkeypatch.setattr(materializer,'require_plan',plans)
    from gx1.contracts import unified_exit_gpu_batch_selection_v1 as gpu
    monkeypatch.setattr(gpu,'require_selection',lambda value,**kw:value)
    monkeypatch.setattr(materializer,'require_selection',lambda value,**kw:value)
    def forbidden(*a,**kw):raise AssertionError('historical model/smoke authority reached')
    monkeypatch.setattr(native,'require_native_completed_smoke',forbidden)
    result=materializer.materialize_native_candidate_campaign(repo=repo,output=tmp_path/'campaign',runtime=tmp_path/'runtime',
        gpu_uuid='GPU-fixture',prepared_boot_path=boot,prepared_boot_file_sha256=native.file_sha256(boot),
        certificate_path=Path(guards['certificate']['path']),prior_campaign_path=Path(pb['path']),prior_campaign_file_sha256=pb['sha256'],
        selection_path=Path(sb['path']),selection_file_sha256=sb['sha256'],recipe_path=Path(rb['path']),recipe_file_sha256=rb['sha256'],window_count=3)
    plan=result['plan'];assert plan['final_train_checkpoint_authority'] is None and plan['entry_pairs_per_epoch']==4500
    assert len(plan['checked_invocations'])==3 and plan['policy']['maximum_memory_junction_temperature_c']==80
    assert all(i['launcher_argv'][:3]==[str(repo/'scripts/gx1_capped_run.sh'),'--class','trainer'] for i in plan['checked_invocations'])
    invocation=plan['checked_invocations'][0];wp=invocation['native_window_policy']
    position,cursor=_cursor(tmp_path,rb,256)
    monkeypatch.setattr(window.native.trainer,'_require_cuda_trainer_guard_execution',lambda **kw:None)
    monkeypatch.setattr(window,'_context',lambda *a:(wp,plan,invocation,{}))
    monkeypatch.setattr(window,'_expected_training_pointer',lambda **kw:None)
    def run(**kw):
        budget=json.loads(Path(kw['execution_budget_path']).read_text())
        assert budget['stop_after_optimizer_steps']==256 and budget['stop_after_completed_val_epochs'] is None
        return {'decision':'PAUSED_RESUMABLE','resume_state':position}
    monkeypatch.setattr(window.native,'run_guarded_native_candidate_invocation',run)
    progress=window.run_window(policy_path=tmp_path/'unused-window',policy_file_sha256='a'*64,progress_path=Path(wp['progress_path']))
    raw=json.loads(Path(progress['progress']['path']).read_text());assert raw['total_units']==raw['completed_units']==256
    receipt={'outcome':'RESUMABLE','checkpoint_pointer_snapshot':_bind(Path(wp['campaign_cursor_path']))}
    monkeypatch.setattr(campaign,'require_receipt_chain',lambda plan,receipts,**kw:receipts)
    assert campaign.next_action(plan,[receipt],current_boot=_boot(101,1))=={'decision':'BLOCKED_NATIVE_LEARNING_REVIEW_REQUIRED'}
    # Remaining scheduled windows cannot turn a finished experiment into more training.
    bad=copy.deepcopy(plan);bad['final_train_checkpoint_authority']=sb
    bad.pop('checked_invocations');bad.pop('selection_artifact_sha256');bad.pop('plan_sha256')
    bad['plan_sha256']=campaign.canonical_sha256(bad)
    with pytest.raises(RuntimeError,match='historical seed'):real_require(bad,verify_files=True)
