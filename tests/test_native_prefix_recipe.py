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


@pytest.mark.parametrize("phase,planned,accepted",[
    ("horizon",1024,True),("horizon",256,False),(None,1024,False),(None,256,True),
])
def test_physical_recipe_plans_the_fit_budget_only_when_the_design_declares_the_horizon_phase(physical_recipe,monkeypatch,phase,planned,accepted):
    import pandas as pd
    from gx1.contracts.entry_observed_market_horizons_v1 import ENTRY_ONLY_HORIZON_TRAINING_PHASE
    c = physical_recipe
    d = c["design"]
    d["budget"]["planned_optimizer_steps"] = planned
    if phase == "horizon":
        d["scope"]["entry_only_training_phase"] = ENTRY_ONLY_HORIZON_TRAINING_PHASE
    recipe = c["seal"]()
    def forbidden(*a, **kw):
        raise AssertionError("No market materialization or model construction")
    monkeypatch.setattr(pd,"read_parquet",forbidden)
    monkeypatch.setattr(runner.val,"_model",forbidden)
    if accepted:
        assert native.require_chronological_prefix_recipe(recipe)["train_rows"]==4500
    else:
        with pytest.raises(RuntimeError,match="NATIVE_PREFIX_DESIGN_OR_TARGET_MISMATCH"):
            native.require_chronological_prefix_recipe(recipe)


@pytest.mark.parametrize("fault", [None, "unbound_policy", "old_prefix_policy", "extra_steps", "extra_epoch",
    "drop_last", "full_val", "refresh", "missing_initial", "h20_phase", "unbounded_windows"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_m5_single_epoch_requires_own_exact_policy_fresh_measurement_and_actual_ceiling(scope, tmp_path, monkeypatch, fault, dynamic):
    """Policy dispatch mechanics only; physical gates are tested separately."""
    from gx1.contracts import entry_observed_market_horizons_v1 as horizons
    from gx1.contracts.entry_observed_market_v1 import ENTRY_ONLY_M5_PATH_TRAINING_PHASE
    from tests.test_entry_observed_market_horizons_learning import _single_epoch_design
    policy, recipe, write, save = scope
    design, witness = _dynamic_single_epoch_design(tmp_path) if dynamic else _single_epoch_design(tmp_path)
    single = horizons.require_m5_single_epoch_population(design)
    measured = {"artifacts":{"initial_measurement_audit":write("measured.json",{"synthetic":True})}}
    prefix = {"artifacts":{"design":write("design-bound.json",design)}, "entry_m5_path_single_epoch":single}
    monkeypatch.setattr(native, "require_chronological_prefix_recipe", lambda recipe: prefix)
    monkeypatch.setattr(native, "require_chronological_learning_measurement", lambda recipe: measured)
    monkeypatch.setattr(horizons, "require_observed_entry_scope", lambda recipe: {
        "training_phase":witness["training_phase"] if fault != "h20_phase" else horizons.ENTRY_ONLY_HORIZON_TRAINING_PHASE})
    monkeypatch.setattr(native, "_expected_observed_entry_binding", lambda recipe, expected: expected.update(
        entry_observed_market={"synthetic_identity":True}))
    recipe.pop("candidate_resume_origin")
    recipe.update(chronological_prefix=prefix["artifacts"], entry_m5_path_single_epoch=single["binding"],
        entry_observed_market={"synthetic":True}, chronological_learning_measurement=measured["artifacts"]["initial_measurement_audit"],
        run_id="SINGLE_EPOCH",out_bundle_dir=str(tmp_path/"MODEL"))
    policy.pop("native_learning_calibration")
    if dynamic:
        from gx1.contracts.entry_candidate_checkpoint_policy_v1 import DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR
        recipe["val_limits"] = {"max_model_forwards":16, "max_state_views":256,
                                "max_wall_seconds":10800, "progress_interval_forwards":16}
        policy["required_val_profile"] = dict(recipe["val_limits"])
        recipe["trainer_cli"]["checkpoint_monitor"] = DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR
        policy["training_enabled"] = True
        policy["full_epoch_training_allowed"] = True
    expected = {"population_witness":single["binding"], "chronological_prefix":prefix["artifacts"],
        "chronological_learning_measurement":measured["artifacts"]["initial_measurement_audit"],
        "run_id":recipe["run_id"], "out_bundle_dir":recipe["out_bundle_dir"],
        "source_bindings_sha256":recipe["source_bindings_sha256"], "max_invocations":3,
        "optimizer_steps":single["optimizer_steps"], "maximum_trained_entry_rows":single["train_rows"],
        "batch_size":witness["batch_size"], "drop_last":False, "epochs":1, "final_model_variant":"ONLINE",
        "teacher_refresh_allowed":False, "full_epoch_training_allowed":True, "full_val_allowed":False,
        "test_data_used":False, "automatic_extension_allowed":False, "entry_observed_market":{"synthetic_identity":True}}
    policy["entry_m5_path_single_epoch_run"] = expected
    if fault == "old_prefix_policy": policy["chronological_learning_run"] = expected
    elif fault == "extra_steps": expected["optimizer_steps"] += 1
    elif fault == "extra_epoch": expected["epochs"] = 2
    elif fault == "drop_last": expected["drop_last"] = True
    elif fault == "full_val": expected["full_val_allowed"] = True
    elif fault == "refresh": expected["teacher_refresh_allowed"] = True
    elif fault == "missing_initial": recipe.pop("chronological_learning_measurement")
    elif fault == "unbounded_windows": expected["max_invocations"] = None
    elif fault == "training_false": policy["training_enabled"] = False
    elif fault == "training_missing": policy.pop("training_enabled")
    elif fault == "training_integer": policy["training_enabled"] = 1
    elif fault == "epoch_false": policy["full_epoch_training_allowed"] = False
    elif fault == "epoch_missing": policy.pop("full_epoch_training_allowed")
    elif fault == "epoch_integer": policy["full_epoch_training_allowed"] = 1
    save()
    if fault == "unbound_policy": write("NEXT_RUN_POLICY.json", {**policy,"training_enabled":not policy["training_enabled"]})
    budget = {"stop_after_optimizer_steps":single["optimizer_steps"], "stop_after_completed_val_epochs":None,
        "max_invocation_seconds":12000}
    if fault is not None:
        with pytest.raises(RuntimeError): native.require_native_run_scope(recipe,execution_budget=budget)
    else:
        assert native.require_native_run_scope(recipe,invocation_number=3,execution_budget=budget) == single["optimizer_steps"]
        for invalid in (0,4,True):
            with pytest.raises(RuntimeError): native.require_native_run_scope(recipe,invocation_number=invalid)
        for key,value in (("stop_after_optimizer_steps",single["optimizer_steps"]+1),
                ("stop_after_completed_val_epochs",1),("max_invocation_seconds",12001)):
            with pytest.raises(RuntimeError): native.require_native_run_scope(recipe,execution_budget={**budget,key:value})


def _dynamic_single_epoch_design(tmp_path, *, rows=273):
    """Real immutable metadata/row files, not physical target or launch admission."""
    import numpy as np
    import pandas as pd
    from gx1.contracts import entry_observed_market_horizons_v1 as horizons
    from gx1.contracts.entry_model_native_aux_targets_v3 import M5_DYNAMIC_OBSERVED_EMISSION_SCOPE
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    from gx1.contracts.entry_observed_market_v1 import entry_observed_dynamic_m5_path_contract
    from gx1.scripts.materialize_unified_exit_chronological_coordinates_v1 import _epoch0_order
    from gx1.contracts.unified_exit_random_access_sampler_v1 import build_random_access_sampler_contract
    path = tmp_path / "dynamic-actual-train.parquet"
    pd.DataFrame({"time": pd.date_range("2025-04-01T00:00Z", periods=rows, freq="5min")}).to_parquet(path, index=False)
    source = _bind(path)
    target = entry_observed_dynamic_m5_path_contract()
    manifest = _write(tmp_path / "dynamic-train.manifest.json", {"output_data_path": source["path"], "extra": {
        "rows": rows, "emission_scope": M5_DYNAMIC_OBSERVED_EMISSION_SCOPE,
        "aux_head_target_contract": {"dynamic_target_contract": target},
        "dynamic_m5_supervision": {"schema_version": "gx1_dynamic_m5_physical_target_pointers_v1",
            "physical_population_rule": "any_genuine_first_prefix_terminal_or_path_component"}}})
    sampler = build_random_access_sampler_contract(split="train", source_lineage_sha256="a" * 64,
        transition_budget_per_epoch=rows * 4, transitions_per_entry=4, entry_pair_population=rows)
    order = _epoch0_order(sampler)
    bindings = {}
    for key, values in (("eligible_parent_rows", np.arange(rows, dtype=np.int64)), ("epoch0_parent_order", order)):
        path = tmp_path / ("dynamic-" + key + ".npy")
        np.save(path, values, allow_pickle=False)
        bindings[key] = _bind(path)
    steps = -(-rows // 16)
    witness = {"schema_version": horizons.M5_DYNAMIC_SINGLE_EPOCH_POPULATION_SCHEMA,
        "training_phase": ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE, "target_contract": target,
        "emission_scope": M5_DYNAMIC_OBSERVED_EMISSION_SCOPE, "train_parquet": source, "train_manifest": manifest,
        **bindings, "native_sampler_contract": sampler, "entry_eligibility_cutoff_utc": "2026-01-01T00:00Z",
        "train_rows": rows, "optimizer_steps": steps, "batch_size": 16, "epochs": 1,
        "drop_last": False, "test_data_used": False}
    design = {"scope": {"entry_m5_path_single_epoch": _write(tmp_path / "dynamic-population.json", witness),
        "entry_only_training_phase": ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        "emission_scope": M5_DYNAMIC_OBSERVED_EMISSION_SCOPE},
        "calendar": {"train_control_cutoff": witness["entry_eligibility_cutoff_utc"]},
        "budget": {"maximum_trained_entry_rows": rows, "planned_optimizer_steps": steps,
            "epochs_completed": 1, "repeat_or_automatic_extension": False}}
    return design, witness


@pytest.mark.parametrize("rows,last", [(273, 1), (272, 16)])
def test_dynamic_single_epoch_binds_actual_whole_train_manifest_and_keeps_tail(tmp_path, rows, last):
    from gx1.contracts import entry_observed_market_horizons_v1 as horizons
    design, witness = _dynamic_single_epoch_design(tmp_path, rows=rows)
    checked = horizons.require_m5_single_epoch_population(design)
    assert checked["train_manifest"] == witness["train_manifest"]
    assert checked["train_rows"] == rows and checked["last_batch_rows"] == last
    assert checked["optimizer_steps"] == -(-rows // 16)
    assert horizons.planned_prefix_training_rows(design) == rows
    assert horizons.planned_prefix_optimizer_steps(design) == -(-rows // 16)


@pytest.mark.parametrize("fault", ["old_schema", "old_phase", "scope", "target", "manifest_missing",
    "manifest_scope", "manifest_bytes", "eligible_subset", "drop_last", "extra_epoch", "extra_steps"])
def test_dynamic_single_epoch_rejects_semantic_aliases_missing_source_and_partial_population(tmp_path, fault):
    import numpy as np
    from gx1.contracts import entry_observed_market_horizons_v1 as horizons
    design, witness = _dynamic_single_epoch_design(tmp_path)
    if fault == "old_schema": witness["schema_version"] = horizons.M5_SINGLE_EPOCH_POPULATION_SCHEMA
    elif fault == "old_phase": witness["training_phase"] = "entry_only_observed_m5_path"
    elif fault == "scope": design["scope"]["emission_scope"] = "m5_primary_observed"
    elif fault == "target": witness["target_contract"] = {}
    elif fault == "manifest_missing": witness.pop("train_manifest")
    elif fault == "manifest_scope":
        manifest = json.loads(Path(witness["train_manifest"]["path"]).read_text())
        manifest["extra"]["emission_scope"] = "m5_primary_observed"
        witness["train_manifest"] = _write(tmp_path / "wrong-manifest.json", manifest)
    elif fault == "manifest_bytes": Path(witness["train_manifest"]["path"]).write_text("{}")
    elif fault == "eligible_subset":
        for key in ("eligible_parent_rows", "epoch0_parent_order"):
            values = np.load(witness[key]["path"], allow_pickle=False)
            values = values[values != 0]
            path = tmp_path / ("subset-" + key + ".npy")
            np.save(path, values, allow_pickle=False)
            witness[key] = _bind(path)
        witness["train_rows"] -= 1
        witness["optimizer_steps"] = -(-witness["train_rows"] // 16)
        design["budget"]["maximum_trained_entry_rows"] = witness["train_rows"]
        design["budget"]["planned_optimizer_steps"] = witness["optimizer_steps"]
    elif fault == "drop_last": witness["drop_last"] = True
    elif fault == "extra_epoch": witness["epochs"] = 2
    elif fault == "extra_steps": witness["optimizer_steps"] += 1
    design["scope"]["entry_m5_path_single_epoch"] = _write(tmp_path / "dynamic-population.json", witness)
    with pytest.raises(RuntimeError): horizons.require_m5_single_epoch_population(design)


def test_dynamic_phase_without_actual_epoch_witness_never_inherits_old_prefix_budget():
    from gx1.contracts import entry_observed_market_horizons_v1 as horizons
    design = {"scope": {"entry_only_training_phase": "entry_only_dynamic_m5_path"}}
    with pytest.raises(RuntimeError, match="WITNESS_REQUIRED"):
        horizons.planned_prefix_optimizer_steps(design)
    with pytest.raises(RuntimeError, match="WITNESS_REQUIRED"):
        horizons.planned_prefix_training_rows(design)


@pytest.fixture
def dynamic_native_recipe_metadata(prefix_scope, tmp_path, monkeypatch):
    """Recipe join unit; real physical/scanner gates are independently qualified."""
    import copy
    from gx1.contracts import entry_model_native_train_recipe_v1 as env_owner
    from gx1.contracts.entry_model_native_train_launch_v1 import artifact_binding, canonical_json_sha256
    from gx1.contracts.entry_candidate_checkpoint_policy_v1 import DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR
    policy, recipe, _, seal = prefix_scope
    design = copy.deepcopy(json.loads(Path(recipe["chronological_prefix"]["design"]["path"]).read_text()))
    dynamic_design, witness = _dynamic_single_epoch_design(tmp_path, rows=273)
    design["scope"].update(dynamic_design["scope"])
    design["calendar"].update(dynamic_design["calendar"])
    design["calendar"].update(physical_source_splits={"train":"train", "control":"val"},
                              physical_coordinate_namespaces_are_separate=True)
    design["budget"].update(dynamic_design["budget"])
    design["targets"] = {"dynamic_target_contract": witness["target_contract"]}
    names = ("entry_val_parquet", "entry_val_manifest", "m5_prebuilt", "source_bundle_metadata",
             "sequence_source_audit", "val_sequence_source_audit", "entry_measurement_coordinates", "train_cost_authority")
    files = {name: tmp_path / ("dynamic-native-" + name + ".json") for name in names}
    for path in files.values(): _write(path, {})
    files.update(entry_train_parquet=Path(witness["train_parquet"]["path"]),
        entry_train_manifest=Path(witness["train_manifest"]["path"]),
        entry_only_base_normalization=tmp_path/"dynamic-native-base.json")
    norm = {"contract":{"lineage":{"dataset_run_id":"DYNAMIC_UNIT",
        "m5_prebuilt_path":str(files["m5_prebuilt"]), "m5_prebuilt_sha256":_bind(files["m5_prebuilt"])["sha256"]}}}
    normalization = _write(files["entry_only_base_normalization"], norm)
    artifacts = {"design":_write(tmp_path/"dynamic-native-design.json", design),
        "normalization_result":normalization, "labels_result":_write(tmp_path/"dynamic-native-labels.json", {}),
        "native_coordinates":_write(tmp_path/"dynamic-native-coordinates.json", {})}
    physical = {"entry_training_phase":witness["training_phase"], "train_rows":witness["train_rows"],
        "entry_only_base_normalization":normalization, "base_contract_sha256":"d"*64,
        "physical_sources":{split:{kind:_bind(files[f"entry_{split}_{kind}"]) for kind in ("parquet","manifest")}
                            for split in ("train","val")},
        "sequence_source_audits":{"train":_bind(files["sequence_source_audit"]),
                                  "val":_bind(files["val_sequence_source_audit"])}}
    coordinates = {"population_witness":design["scope"]["entry_m5_path_single_epoch"],
        "native_sampler_contract":witness["native_sampler_contract"]}
    monkeypatch.setattr(native, "require_physical_chronological_preprocessing", lambda *a, **kw: physical)
    monkeypatch.setattr(native, "require_physical_native_training_coordinates", lambda *a, **kw: coordinates)
    _write(files["entry_measurement_coordinates"], {"schema_version":"gx1_dynamic_m5_native_coordinate_publication_v1",
        "chronological_prefix":artifacts, "population_witness":coordinates["population_witness"],
        "native_sampler_contract":coordinates["native_sampler_contract"], "native_coordinates_complete":True,
        "entry_measurement_complete":False, "native_launch_admitted":False, "test_data_used":False})
    recipe.pop("exit_reference_policy")
    recipe.update(chronological_prefix=artifacts, entry_m5_path_single_epoch=coordinates["population_witness"],
        dataset_run_id="DYNAMIC_UNIT", files={key:_bind(path) for key,path in files.items()})
    recipe["trainer_cli"].update(epochs=1, save_top_k=0, checkpoint_monitor=DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR)
    recipe["val_limits"] = {"max_model_forwards":16, "max_state_views":256,
                            "max_wall_seconds":10800, "progress_interval_forwards":16}
    policy["required_val_profile"] = dict(recipe["val_limits"])
    source = Path(recipe["source_repo"])/"gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py"
    source.parent.mkdir(parents=True, exist_ok=True); source.write_text("synthetic dynamic constructor source")
    recipe["source_bindings"] = {str(source.relative_to(Path(recipe["source_repo"]))):artifact_binding(source)}
    recipe["source_bindings_sha256"] = canonical_json_sha256(recipe["source_bindings"])
    recipe["recipe_env"] = {**env_owner.MODEL_NATIVE_RECIPE_ENV, "GX1_V10_CKPT_MONITOR":DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR}
    from gx1.contracts import entry_observed_market_horizons_v1 as horizons
    observed_binding = _write(tmp_path/"dynamic-native-observed-plan.json", {"unit":True})
    recipe["entry_observed_market"] = observed_binding
    observed_scope = {"binding":observed_binding, "economics":{"unit":True}, "target_m1_source_sha256":"b"*64,
        "training_phase":witness["training_phase"], "physical_sources":copy.deepcopy(physical["physical_sources"]),
        "target_contract":witness["target_contract"],
        "query_minutes":tuple(witness["target_contract"]["query_contract"]["horizon_minutes"])}
    def checked_scope(value):
        if value.get("entry_observed_market") != observed_binding:
            raise RuntimeError("UNIT_OBSERVED_SCOPE_BINDING_CHANGED")
        return observed_scope
    monkeypatch.setattr(horizons, "require_observed_entry_scope", checked_scope)
    from gx1.contracts import entry_dynamic_m5_loss_scale_v1 as loss_owner
    from tests.test_entry_dynamic_m5_loss_scale import _synthetic_loaded_loss_scales
    loss_scales = _synthetic_loaded_loss_scales(tmp_path / "recipe-loss-owner-boundary")
    loss_identity = loss_owner.dynamic_m5_loss_scale_identity(loss_scales)
    recipe["dynamic_m5_loss_scales"] = loss_scales["binding"]
    policy["chronological_learning_run"]["dynamic_m5_loss_scales"] = loss_scales["binding"]
    # The physical target-lineage owner is the same completed boundary isolated
    # above. Keep artifact bytes/stats/identity checks real; this explicit small
    # synthetic fit makes no fit claim about the fixture's273physical rows.
    stable_scope_fields = ("training_phase", "target_contract", "economics", "target_m1_source_sha256", "query_minutes")
    stable_scope = copy.deepcopy({key: observed_scope[key] for key in stable_scope_fields})
    stable_train_source = copy.deepcopy(observed_scope["physical_sources"]["train"])
    stable_witness_fields = ("train_parquet", "train_manifest", "entry_eligibility_cutoff_utc", "eligible_parent_rows")
    stable_witness = copy.deepcopy({key: witness[key] for key in stable_witness_fields})
    stable_population_binding = copy.deepcopy(coordinates["population_witness"])
    def completed_loss_lineage(checked, population):
        # A later initial/learning plan may copy this scope and replace its plan
        # binding. Only immutable TRAIN target lineage determines these units.
        if ({key: checked.get(key) for key in stable_scope_fields} != stable_scope
                or checked.get("physical_sources", {}).get("train") != stable_train_source
                or population.get("train_rows") != witness["train_rows"]
                or population.get("binding") != stable_population_binding
                or {key: population.get("witness", {}).get(key) for key in stable_witness_fields} != stable_witness):
            raise RuntimeError("UNIT_DYNAMIC_LOSS_TRAIN_LINEAGE_CHANGED")
        return copy.deepcopy(loss_scales["fit_lineage"])
    monkeypatch.setattr(loss_owner, "dynamic_m5_loss_fit_lineage", completed_loss_lineage)
    seal()
    return {"policy":policy, "recipe":recipe, "files":files, "seal":seal,
            "physical":physical, "coordinates":coordinates, "design":design, "witness":witness,
            "observed_scope":observed_scope, "loss_scales":loss_scales, "loss_scale_identity":loss_identity}


def test_dynamic_native_recipe_binds_actual_population_without_exit_or_economics(dynamic_native_recipe_metadata):
    case = dynamic_native_recipe_metadata; recipe = case["recipe"]
    prefix = native.require_chronological_prefix_recipe(recipe)
    assert prefix["entry_m5_path_single_epoch"]["train_rows"] == 273
    assert native._require_dynamic_m5_loss_scale_identity(recipe, prefix=prefix,
        observed_scope=case["observed_scope"]) == case["loss_scale_identity"]
    binding, digest = native.require_fresh_physical_campaign_sampler(recipe)
    assert binding == recipe["entry_m5_path_single_epoch"] and digest == binding["sha256"]
    checked, rows = native.require_native_recipe_metadata(case["seal"](),
        source_repo=Path(recipe["source_repo"]), source_commit=recipe["source_commit"])
    assert checked == recipe and rows == 273
    profile, _, objective = native._require_native_profile_and_economics(recipe, case["policy"], Path(recipe["source_repo"]))
    assert profile == recipe["val_limits"] and objective is None
    assert not {"selected_sampler","random_access_root","economics_readiness"}.intersection(recipe["files"])


def test_dynamic_loss_units_allow_copied_scope_with_later_plan_binding(dynamic_native_recipe_metadata, tmp_path):
    case = dynamic_native_recipe_metadata
    prefix = native.require_chronological_prefix_recipe(case["recipe"])
    copied = copy.deepcopy(case["observed_scope"])
    copied["binding"] = _write(tmp_path / "later-observed-plan.json", {"new_plan_same_train_lineage": True})
    assert native._require_dynamic_m5_loss_scale_identity(case["recipe"], prefix=prefix,
        observed_scope=copied) == case["loss_scale_identity"]


@pytest.mark.parametrize("fault", ["train_source", "target_source", "train_rows"])
def test_dynamic_loss_units_reject_actual_train_lineage_change(dynamic_native_recipe_metadata, fault):
    case = dynamic_native_recipe_metadata
    prefix = native.require_chronological_prefix_recipe(case["recipe"])
    scope = copy.deepcopy(case["observed_scope"])
    if fault == "train_source": scope["physical_sources"]["train"] = scope["physical_sources"]["val"]
    elif fault == "target_source": scope["target_m1_source_sha256"] = "f" * 64
    else:
        prefix = copy.deepcopy(prefix)
        prefix["entry_m5_path_single_epoch"]["train_rows"] += 1
    with pytest.raises(RuntimeError, match="UNIT_DYNAMIC_LOSS_TRAIN_LINEAGE_CHANGED"):
        native._require_dynamic_m5_loss_scale_identity(case["recipe"], prefix=prefix, observed_scope=scope)


@pytest.mark.parametrize("fault", ["exit_policy", "target", "legacy_file", "monitor", "save_top_k", "missing_bound",
    "extra_bound", "bool_bound", "source", "normalization", "coordinate", "witness", "profile", "missing_scope", "scope_phase",
    "missing_loss_scales", "changed_loss_scales", "loss_scale_bytes", "loss_scale_lineage"])
def test_dynamic_native_recipe_rejects_legacy_or_unbound_controls(dynamic_native_recipe_metadata, fault):
    case = dynamic_native_recipe_metadata; recipe = case["recipe"]
    if fault == "exit_policy": recipe["exit_reference_policy"] = {}
    elif fault == "target":
        case["design"]["targets"]["dynamic_target_contract"] = {}
        recipe["chronological_prefix"]["design"] = _write(Path(recipe["chronological_prefix"]["design"]["path"]), case["design"])
    elif fault == "legacy_file": recipe["files"]["economics_readiness"] = recipe["files"]["source_bundle_metadata"]
    elif fault == "monitor": recipe["trainer_cli"]["checkpoint_monitor"] = "entry_policy_pnl"
    elif fault == "save_top_k": recipe["trainer_cli"]["save_top_k"] = 1
    elif fault == "missing_bound": recipe["val_limits"].pop("max_model_forwards")
    elif fault == "extra_bound": recipe["val_limits"]["policy_batch_size"] = 256
    elif fault == "bool_bound": recipe["val_limits"]["progress_interval_forwards"] = True
    elif fault == "source": recipe["files"]["entry_train_parquet"] = recipe["files"]["entry_val_parquet"]
    elif fault == "normalization": recipe["dataset_run_id"] = "OTHER"
    elif fault == "coordinate": recipe["files"]["entry_measurement_coordinates"] = recipe["files"]["source_bundle_metadata"]
    elif fault == "witness": recipe["entry_m5_path_single_epoch"] = recipe["files"]["source_bundle_metadata"]
    elif fault == "missing_scope": recipe.pop("entry_observed_market")
    elif fault == "scope_phase": case["observed_scope"]["training_phase"] = "joint_fitted_q"
    elif fault == "missing_loss_scales": recipe.pop("dynamic_m5_loss_scales")
    elif fault == "changed_loss_scales": recipe["dynamic_m5_loss_scales"] = {**recipe["dynamic_m5_loss_scales"], "sha256":"0"*64}
    elif fault == "loss_scale_bytes":
        path = Path(recipe["dynamic_m5_loss_scales"]["path"]); path.write_bytes(path.read_bytes()+b"\n")
    elif fault == "loss_scale_lineage": case["loss_scales"]["fit_lineage"]["train_control_cutoff"] = "2025-03-01T00:00:00Z"
    else:
        case["policy"]["required_val_profile"]["max_model_forwards"] = 32
        with pytest.raises(RuntimeError, match="DYNAMIC_PROFILE"):
            native._require_native_profile_and_economics(recipe, case["policy"], Path(recipe["source_repo"]))
        return
    with pytest.raises(RuntimeError):
        native.require_native_recipe_metadata(case["seal"](),
            source_repo=Path(recipe["source_repo"]), source_commit=recipe["source_commit"])


@pytest.mark.parametrize("fault", ["training_false", "training_missing", "training_integer", "epoch_false", "epoch_missing", "epoch_integer"])
def test_dynamic_positive_epoch_never_uses_false_or_untyped_global_training_authority(scope, tmp_path, monkeypatch, fault):
    test_m5_single_epoch_requires_own_exact_policy_fresh_measurement_and_actual_ceiling(
        scope, tmp_path, monkeypatch, fault, True)


@pytest.mark.parametrize("training_enabled", [False, None, 1, True])
def test_dynamic_positive_prefix_probe_requires_exact_true_global_training(prefix_scope, monkeypatch, training_enabled):
    policy, recipe, _, seal = prefix_scope
    # Isolate this global-policy dispatch predicate; physical admission has its own tests.
    monkeypatch.setattr(native, "_dynamic_m5_native_design", lambda value: {"unit_dynamic_phase":True})
    monkeypatch.setattr(native, "_require_native_profile_and_economics", lambda *args: None)
    if training_enabled is None: policy.pop("training_enabled")
    else: policy["training_enabled"] = training_enabled
    seal()
    if training_enabled is True:
        assert native.require_chronological_prefix_run(recipe)["optimizer_step_ceiling"] == 256
    else:
        with pytest.raises(RuntimeError, match="NOT_AUTHORIZED"): native.require_chronological_prefix_run(recipe)


def test_entry_execution_profile_joins_constructor_and_recipe_without_changing_full_default(tmp_path, monkeypatch):
    from gx1.features.entry_specialist_feature_groups_v1 import compile_compact_entry_execution_profile
    compiled = compile_compact_entry_execution_profile()
    reference = {key: compiled[key] for key in ("name", "contract_sha256")}
    constructor = tmp_path / "constructor.json"
    recipe = {"files": {"source_bundle_metadata": _write(constructor, {})}}
    monkeypatch.setattr(native, "_dynamic_m5_native_design", lambda _: None)
    assert native.require_dynamic_m5_execution_profile(recipe) is None
    recipe["entry_execution_profile"] = reference
    recipe["files"]["source_bundle_metadata"] = _write(constructor, {"entry_execution_profile": reference})
    monkeypatch.setattr(native, "_dynamic_m5_native_design", lambda _: {"scope": "explicit-dynamic-test-boundary"})
    assert native.require_dynamic_m5_execution_profile(recipe) == reference


@pytest.mark.parametrize("fault", ["recipe_missing", "constructor_missing", "different_hash", "unknown_name", "extra_field", "wrong_type", "non_dynamic", "non_dynamic_none"])
def test_entry_execution_profile_rejects_unbound_or_wrong_model_function(tmp_path, monkeypatch, fault):
    from gx1.features.entry_specialist_feature_groups_v1 import compile_compact_entry_execution_profile
    compiled = compile_compact_entry_execution_profile()
    reference = {key: compiled[key] for key in ("name", "contract_sha256")}
    metadata = {"entry_execution_profile": copy.deepcopy(reference)}
    recipe = {"entry_execution_profile": copy.deepcopy(reference), "files": {}}
    expected = "NATIVE_ENTRY_EXECUTION_PROFILE_RECIPE_CONSTRUCTOR_MISMATCH"
    if fault == "recipe_missing":
        recipe.pop("entry_execution_profile")
    elif fault == "constructor_missing":
        metadata.pop("entry_execution_profile")
    elif fault == "different_hash":
        recipe["entry_execution_profile"]["contract_sha256"] = "0" * 64
    elif fault == "unknown_name":
        metadata["entry_execution_profile"]["name"] = "unregistered-profile"
        recipe["entry_execution_profile"] = copy.deepcopy(metadata["entry_execution_profile"])
        expected = "ENTRY_EXECUTION_PROFILE"
    elif fault == "extra_field":
        metadata["entry_execution_profile"]["arbitrary_feature_selection"] = True
        recipe["entry_execution_profile"] = copy.deepcopy(metadata["entry_execution_profile"])
        expected = "ENTRY_EXECUTION_PROFILE"
    elif fault == "wrong_type":
        metadata["entry_execution_profile"] = "compact"
        recipe["entry_execution_profile"] = "compact"
        expected = "ENTRY_EXECUTION_PROFILE"
    else:
        expected = "NATIVE_ENTRY_EXECUTION_PROFILE_DYNAMIC_ENTRY_REQUIRED"
        if fault == "non_dynamic_none":
            metadata["entry_execution_profile"] = None
            recipe["entry_execution_profile"] = None
    recipe["files"]["source_bundle_metadata"] = _write(tmp_path / "constructor.json", metadata)
    monkeypatch.setattr(native, "_dynamic_m5_native_design", lambda _: None if fault in {"non_dynamic", "non_dynamic_none"} else {"scope": "explicit-dynamic-test-boundary"})
    with pytest.raises(RuntimeError, match=expected):
        native.require_dynamic_m5_execution_profile(recipe)
