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
    initial='chronological_initial_measurement' in recipe
    steps,windows=(0,1) if initial else (256,3)
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
        selection_path=Path(sb['path']),selection_file_sha256=sb['sha256'],recipe_path=Path(rb['path']),recipe_file_sha256=rb['sha256'],window_count=windows)
    plan=result['plan'];assert plan['final_train_checkpoint_authority'] is None and plan['entry_pairs_per_epoch']==4500
    assert len(plan['checked_invocations'])==windows and plan['policy']['maximum_memory_junction_temperature_c']==80
    assert all(i['launcher_argv'][:3]==[str(repo/'scripts/gx1_capped_run.sh'),'--class','trainer'] for i in plan['checked_invocations'])
    invocation=plan['checked_invocations'][0];wp=invocation['native_window_policy']
    position,cursor=_cursor(tmp_path,rb,steps)
    monkeypatch.setattr(window.native.trainer,'_require_cuda_trainer_guard_execution',lambda **kw:None)
    monkeypatch.setattr(window,'_context',lambda *a:(wp,plan,invocation,{}))
    monkeypatch.setattr(window,'_expected_training_pointer',lambda **kw:None)
    def run(**kw):
        budget=json.loads(Path(kw['execution_budget_path']).read_text())
        assert budget['stop_after_optimizer_steps']==steps and budget['stop_after_completed_val_epochs'] is None
        return {'decision':'PAUSED_RESUMABLE','resume_state':position}
    monkeypatch.setattr(window.native,'run_guarded_native_candidate_invocation',run)
    progress=window.run_window(policy_path=tmp_path/'unused-window',policy_file_sha256='a'*64,progress_path=Path(wp['progress_path']))
    raw=json.loads(Path(progress['progress']['path']).read_text())
    assert raw['total_units']==max(1,steps) and raw['completed_units']==steps
    receipt={'outcome':'RESUMABLE','checkpoint_pointer_snapshot':_bind(Path(wp['campaign_cursor_path']))}
    monkeypatch.setattr(campaign,'require_receipt_chain',lambda plan,receipts,**kw:receipts)
    assert campaign.next_action(plan,[receipt],current_boot=_boot(101,1))=={'decision':'BLOCKED_NATIVE_LEARNING_REVIEW_REQUIRED'}
    # Remaining scheduled windows cannot turn a finished experiment into more training.
    bad=copy.deepcopy(plan);bad['final_train_checkpoint_authority']=sb
    bad.pop('checked_invocations');bad.pop('selection_artifact_sha256');bad.pop('plan_sha256')
    bad['plan_sha256']=campaign.canonical_sha256(bad)
    with pytest.raises(RuntimeError,match='historical seed'):real_require(bad,verify_files=True)


@pytest.fixture(scope="module")
def physical_normalization_templates():
    return _physical_normalization_templates_for_rows(4500)


def _physical_normalization_templates_for_rows(train_rows):
    import numpy as np
    import pandas as pd
    from tests.model_native_input_normalization_support import input_normalization_fixture
    from gx1.contracts.unified_exit_pilot_normalization_v1 import (
        build_physical_summary_sample_authority, fit_lifetime_summary_normalization,
    )
    from gx1.contracts.unified_exit_pilot_final_bindings_v1 import build_split_sequence_binding
    base = input_normalization_fixture(signal_names=["a", "b"], mtf_names=["m", "n"], rows=train_rows+1)
    authority = build_physical_summary_sample_authority(
        successor_transition_count_by_entry=[1]*train_rows, source_lineage_sha256="1"*64)
    values = np.arange(authority["fit_row_count"]*7, dtype=np.float64).reshape(-1, 7)
    lifetime = fit_lifetime_summary_normalization(values=values, sample_authority=authority)
    sequences = {}
    for split, first, count in (("train", "2021-01-01T00:00Z", train_rows), ("val", "2021-06-01T00:00Z", 320)):
        entry = pd.date_range(first, periods=count, freq="5min")
        m1 = pd.date_range(first, periods=count*5+10, freq="min")
        sequences[split] = build_split_sequence_binding(
            split=split, entry_times=entry, m1_times=m1, successor_transition_counts=[1]*count,
            **{key:"a"*64 for key in (
                "child_admission_file_sha256", "child_admission_witness_sha256", "child_parquet_sha256",
                "child_manifest_file_sha256", "child_manifest_contract_sha256", "m1_source_sha256",
                "m1_manifest_file_sha256", "closure_authority_file_sha256", "closure_authority_sha256")})
    return base, authority, lifetime, sequences


@pytest.fixture
def physical_recipe(tmp_path, physical_normalization_templates):
    import numpy as np
    from gx1.contracts.unified_exit_pilot_final_bindings_v1 import build_composite_normalization_binding
    from gx1.contracts.unified_exit_reference_policy_v1 import reference_policy_contract
    base, authority, lifetime, sequences = copy.deepcopy(physical_normalization_templates)
    train_rows = base["lineage"]["entry_train_decision_row_count"]
    sources = {}
    for split, count, start, end in (
        ("train",train_rows,"2021-01-01T00:00Z","2021-05-31T23:59:59Z"),
        ("val",320,"2021-06-01T00:00Z","2021-06-30T23:59:59Z"),
    ):
        import pandas as pd
        parquet = tmp_path/(split+".parquet")
        pd.DataFrame({"time": pd.date_range(start, periods=count, freq="5min")}).to_parquet(parquet, index=False)
        sources[split] = {
            "parquet": _bind(parquet),
            "manifest": _write(tmp_path/(split+".manifest.json"), {"synthetic":split}),
            "physical_rows":count,"clock_sha256":sequences[split]["entry_clock_sha256"],
            "declared_window":{"start":start,"end":end},
        }
        sequences[split]["bindings"]["child_parquet"] = sources[split]["parquet"]["sha256"]
        sequences[split]["binding_sha256"] = native.native_sha256(
            {k:v for k,v in sequences[split].items() if k!="binding_sha256"})
    row_bindings={}
    for key,rows in (("TRAIN_CALENDAR_PARENT_ROWS",np.arange(train_rows,dtype=np.int64)),
                     ("CONTROL256_PARENT_ROWS",np.arange(256,dtype=np.int64))):
        p=tmp_path/(key+".npy");np.save(p,rows);row_bindings[key]=_bind(p)
    design={
        "schema_version":"gx1_frozen_chronological_learning_design_v1",
        "status":"DESIGN_AND_CONTROL_IDS_FROZEN_NOT_EXECUTABLE",
        "scope":{"test_sealed":True,"same_architecture":True,"one_experiment":True,
                 "existing_control_periods_are_reused_development":True,"native_launch_authorized":False},
        "initialization":{"mode":"fresh_existing_model_constructor_no_checkpoint_weights","seed":20260911},
        "budget":{"planned_optimizer_steps":256,"maximum_trained_entry_rows":4096,
                  "epochs_completed":0,"repeat_or_automatic_extension":False,"later_control_entries":256},
        "selection":{"control_entries":256},
        "targets":{"reference_policy":reference_policy_contract()},
        "calendar":{"source_bindings":sources,"physical_source_splits":{"train":"train","control":"val"},
            "physical_coordinate_namespaces_are_separate":True,
            "train_entry_start_inclusive":"2021-01-01T00:00Z","train_control_cutoff":"2021-06-01T00:00Z",
            "development_control_entry_end_exclusive":"2021-07-01T00:00Z","bindings":row_bindings},
    }
    db=_write(tmp_path/"physical-design.json",design)
    base["lineage"]["train_parquet_path"]=sources["train"]["parquet"]["path"]
    base["lineage"]["train_parquet_sha256"]=sources["train"]["parquet"]["sha256"]
    base["contract_sha256"]=native.native_sha256({k:v for k,v in base.items() if k!="contract_sha256"})
    base_artifact={"schema_version":"gx1_unified_exit_pilot_base_normalization_v1",
        "decision":"PASS","contract":base,"contract_sha256":base["contract_sha256"],
        "population_witness_sha256":"2"*64,"val_fit_rows":0,"test_fit_rows":0,"test_accessed":False}
    summary={"schema_version":"gx1_unified_exit_pilot_summary_fit_inputs_v1","decision":"PASS",
        "split":"train","entry_pair_population":train_rows,"child_parquet_sha256":sources["train"]["parquet"]["sha256"],
        "summary_sample_authority":authority,"lifetime_summary_normalization":lifetime,
        "successor_counts_sha256":sequences["train"]["successor_counts_sha256"],
        "successor_transition_total":sequences["train"]["successor_transition_total"],
        "val_fit_rows":0,"test_fit_rows":0,"test_accessed":False}
    base_binding=_write(tmp_path/"base.json",base_artifact)
    summary["manifest_sha256"]=native.native_sha256(summary)
    sb=_write(tmp_path/"summary.json",summary)
    composite=build_composite_normalization_binding(
        base_artifact=base_artifact,base_path=base_binding["path"],base_file_sha256=base_binding["sha256"],
        summary_normalization=lifetime,summary_manifest_path=sb["path"],summary_manifest_file_sha256=sb["sha256"],
        summary_manifest_sha256=summary["manifest_sha256"])
    norm={"decision":"PASS_COMPOSITE_AND_FIRST_STATE_BINDINGS_NOT_NATIVE_ADMISSION",
          "test_accessed":False,"forbidden_access_attempts":[],"normalization_fit":False,
          "model_forwards":0,"optimizer_steps":0,"composite_normalization_sha256":composite["composite_normalization_sha256"],
          "artifacts":{"COMPOSITE_NORMALIZATION.json":_write(tmp_path/"composite.json",composite)},"splits":{}}
    for split in ("train","val"):
        norm["artifacts"]["SPLIT_SEQUENCE_BINDING_"+split.upper()+".json"]=_write(tmp_path/(split+"-sequence.json"),sequences[split])
        norm["splits"][split]={"entry_rows":sources[split]["physical_rows"],
            "first_state_geometry_exact":True,"successor_geometry_exact":True,
            "sequence_binding_sha256":sequences[split]["binding_sha256"]}
    inputs={"design":db,"control256":row_bindings["CONTROL256_PARENT_ROWS"]}
    for split in sources:
        for kind in ("parquet","manifest"):inputs[split+"_"+kind]=sources[split][kind]
    plan={"schema_version":"gx1_native_v38_auxiliary_reuse_precheck_plan_v1","input_bindings":inputs}
    labels={"schema_version":"gx1_native_v38_auxiliary_reuse_precheck_v1",
        "decision":"PASS_FROZEN_TRAIN_POLICIES_AND_COMPLETE_PRETEST_TARGET_CLOCK_SUPPORT",
        "identical_frozen_policies_across_physical_splits":True,"producer_code_unchanged_for_five_target_owners":True,
        "test_accessed":False,"forbidden_access_attempts":[],"model_forwards":0,"optimizer_steps":0,
        "plan":_write(tmp_path/"precheck-plan.json",plan),"splits":{}}
    for split in sources:
        labels["splits"][split]={"rows":sources[split]["physical_rows"],
            "exact_m1_fill_and_contiguous_outcome_rows":sources[split]["physical_rows"],
            "entry_clock_sha256":sources[split]["clock_sha256"],
            "supervision_boundary_exclusive_utc":("2021-06-01T00:00Z" if split=="train" else "2021-07-01T00:00Z"),
            "direction_policy_sha256":"a"*64,"position_size_policy_sha256":"b"*64}
    labels["splits"]["val"]["control256"]={"rows":256,"physical_coordinate_source":"val","all_support_checks_passed":True}
    recipe={"initialization":design["initialization"]["mode"],
            "trainer_cli":{"seed":20260911,"batch_size":16,"learning_rate":.0001,"weight_decay":.0001,"grad_accum_steps":1},
            "exit_reference_policy":design["targets"]["reference_policy"],
            "chronological_prefix":{"design":db,"normalization_result":_write(tmp_path/"normalization-result.json",norm),
                                    "labels_result":_write(tmp_path/"labels-result.json",labels)}}
    def seal():
        recipe["chronological_prefix"]["design"]=_write(tmp_path/"physical-design.json",design)
        inputs["design"]=recipe["chronological_prefix"]["design"]
        labels["plan"]=_write(tmp_path/"precheck-plan.json",plan)
        recipe["chronological_prefix"]["normalization_result"]=_write(tmp_path/"normalization-result.json",norm)
        recipe["chronological_prefix"]["labels_result"]=_write(tmp_path/"labels-result.json",labels)
        return recipe
    return dict(recipe=recipe,design=design,normalization=norm,labels=labels,plan=plan,
                composite=composite,base=base_artifact,summary=summary,sources=sources,
                sequences=sequences,seal=seal)


def test_physical_recipe_joins_existing_full_train_normalization_without_market_reads(physical_recipe,monkeypatch):
    import numpy as np
    import pandas as pd
    c=physical_recipe;before=copy.deepcopy(c["recipe"]);reads=[]
    original=Path.resolve
    raw_paths={b["path"] for src in c["sources"].values() for b in (src["parquet"],src["manifest"])}
    def resolve(path,*a,**kw):
        assert str(path) not in raw_paths,"raw dataset pointer resolved"
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,"resolve",resolve)
    def forbidden(*a,**kw):raise AssertionError("No market materialization or model construction")
    monkeypatch.setattr(pd,"read_parquet",forbidden)
    monkeypatch.setattr(runner.val,"_model",forbidden)
    checked=native.require_chronological_prefix_recipe(c["recipe"])
    p=checked["physical_preprocessing"]
    assert c["recipe"]==before and checked["train_rows"]==4500
    assert p["native_coordinates_bound"] is False
    assert p["physical_sources"]==c["sources"]
    assert p["composite_normalization"]==c["normalization"]["artifacts"]["COMPOSITE_NORMALIZATION.json"]
    assert p["base_contract_sha256"]==c["base"]["contract_sha256"]
    assert p["summary_normalization_sha256"]==c["summary"]["lifetime_summary_normalization"]["normalization_sha256"]
    # Equal integer IDs are legal across the two physical source namespaces.
    assert np.intersect1d(np.load(p["train_parent_rows"]["path"]),np.load(p["control_parent_rows"]["path"])).size==256


@pytest.mark.parametrize("fault",[
    "partial_physical_design","collapsed_source","test_pointer","wrong_design",
    "label_failure","normalization_failure","test_accessed","refit","model_forward",
    "population","clock","geometry","sequence_clock","sequence_source","composite_identity",
    "base_file","summary_file","base_source","base_population","base_control_fit",
    "summary_source","summary_population","summary_control_fit","summary_authority","summary_counts","policy_lineage",
    "control_source","train_subset","control_outside","control_duplicate","manifest_bytes",
])
def test_physical_recipe_rejects_mixed_exposed_or_wrong_population_artifacts(physical_recipe,monkeypatch,fault):
    import numpy as np
    from gx1.contracts.unified_exit_pilot_final_bindings_v1 import build_composite_normalization_binding
    c=physical_recipe;d,n,l=c["design"],c["normalization"],c["labels"]
    if fault=="partial_physical_design":d["calendar"]["physical_coordinate_namespaces_are_separate"]=False
    elif fault=="collapsed_source":
        c["sources"]["val"]["parquet"]=c["sources"]["train"]["parquet"]
        c["plan"]["input_bindings"]["val_parquet"]=c["sources"]["train"]["parquet"]
    elif fault=="test_pointer":c["plan"]["input_bindings"]["train_parquet"]={"path":"/DO_NOT_RESOLVE_test.parquet","sha256":"f"*64}
    elif fault=="label_failure":l["decision"]="FAIL"
    elif fault=="normalization_failure":n["decision"]="FAIL"
    elif fault=="test_accessed":n["test_accessed"]=True
    elif fault=="refit":n["normalization_fit"]=True
    elif fault=="model_forward":l["model_forwards"]=1
    elif fault=="population":l["splits"]["train"]["rows"]-=1
    elif fault=="clock":l["splits"]["val"]["entry_clock_sha256"]="f"*64
    elif fault=="geometry":n["splits"]["train"]["first_state_geometry_exact"]=False
    elif fault in ("sequence_clock","sequence_source"):
        seq=c["sequences"]["train"]
        if fault=="sequence_clock":seq["entry_clock_sha256"]="f"*64
        else:seq["bindings"]["child_parquet"]="f"*64
        seq["binding_sha256"]=native.native_sha256({k:v for k,v in seq.items() if k!="binding_sha256"})
        key="SPLIT_SEQUENCE_BINDING_TRAIN.json"
        n["artifacts"][key]=_write(Path(n["artifacts"][key]["path"]),seq)
        n["splits"]["train"]["sequence_binding_sha256"]=seq["binding_sha256"]
    elif fault=="composite_identity":n["composite_normalization_sha256"]="f"*64
    elif fault=="base_file":Path(c["composite"]["base_feature_normalization"]["path"]).write_text("{}")
    elif fault=="summary_file":Path(c["composite"]["summary_fit_manifest"]["path"]).write_text("{}")
    elif fault.startswith("base_") or fault.startswith("summary_"):
        b,s=c["base"],c["summary"]
        if fault=="base_source":b["contract"]["lineage"]["train_parquet_path"]=c["sources"]["val"]["parquet"]["path"]
        elif fault=="base_population":b["contract"]["lineage"]["entry_train_decision_row_count"]-=1;b["contract"]["lineage"]["exit_train_decision_row_count"]+=1
        elif fault=="base_control_fit":b["contract"]["fit_end_utc"]=b["contract"]["lineage"]["train_time_max_utc"]="2021-06-01T01:00:00+00:00"
        elif fault=="summary_source":s["child_parquet_sha256"]=c["sources"]["val"]["parquet"]["sha256"]
        elif fault=="summary_population":s["entry_pair_population"]-=1
        elif fault=="summary_control_fit":s["val_fit_rows"]=1
        elif fault=="summary_authority":s["summary_sample_authority"]["sample_count"]+=1
        elif fault=="summary_counts":s["successor_counts_sha256"]="f"*64
        b["contract"]["contract_sha256"]=native.native_sha256({k:v for k,v in b["contract"].items() if k!="contract_sha256"})
        b["contract_sha256"]=b["contract"]["contract_sha256"]
        bb=_write(Path(c["composite"]["base_feature_normalization"]["path"]),b)
        s["manifest_sha256"]=native.native_sha256({k:v for k,v in s.items() if k!="manifest_sha256"})
        sb=_write(Path(c["composite"]["summary_fit_manifest"]["path"]),s)
        co=build_composite_normalization_binding(base_artifact=b,base_path=bb["path"],base_file_sha256=bb["sha256"],
            summary_normalization=s["lifetime_summary_normalization"],summary_manifest_path=sb["path"],
            summary_manifest_file_sha256=sb["sha256"],summary_manifest_sha256=s["manifest_sha256"])
        n["artifacts"]["COMPOSITE_NORMALIZATION.json"]=_write(Path(n["artifacts"]["COMPOSITE_NORMALIZATION.json"]["path"]),co)
        n["composite_normalization_sha256"]=co["composite_normalization_sha256"]
    elif fault=="policy_lineage":l["splits"]["val"]["position_size_policy_sha256"]="f"*64
    elif fault=="control_source":l["splits"]["val"]["control256"]["physical_coordinate_source"]="train"
    elif fault in ("train_subset","control_outside","control_duplicate"):
        key="TRAIN_CALENDAR_PARENT_ROWS" if fault=="train_subset" else "CONTROL256_PARENT_ROWS"
        rb=d["calendar"]["bindings"][key];rows=np.load(rb["path"])
        if fault=="train_subset":rows=rows[:-1]
        elif fault=="control_outside":rows[-1]=c["sources"]["val"]["physical_rows"]
        else:rows[-1]=rows[-2]
        np.save(rb["path"],rows);rb.update(_bind(Path(rb["path"])))
    c["seal"]()
    if fault=="wrong_design":
        c["plan"]["input_bindings"]["design"]={**c["recipe"]["chronological_prefix"]["design"],"sha256":"f"*64}
        l["plan"]=_write(Path(l["plan"]["path"]),c["plan"])
        c["recipe"]["chronological_prefix"]["labels_result"]=_write(Path(c["recipe"]["chronological_prefix"]["labels_result"]["path"]),l)
    if fault=="manifest_bytes":Path(n["artifacts"]["SPLIT_SEQUENCE_BINDING_VAL.json"]["path"]).write_text("{}")
    original=Path.resolve
    def resolve(path,*a,**kw):
        assert "DO_NOT_RESOLVE" not in str(path),"forbidden pointer followed"
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,"resolve",resolve)
    with pytest.raises(RuntimeError):
        native.require_chronological_prefix_recipe(c["recipe"])
