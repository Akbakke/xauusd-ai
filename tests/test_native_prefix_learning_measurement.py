"""Fixed-step ONLINE comparison through the native owner; synthetic evidence only."""
import copy
import json
import time
from pathlib import Path

import pytest
import torch

from tests.test_native_prefix_initial_measurement import initial_scope, prefix_scope, scope
from tests.test_native_prefix_recipe import native, runner, _write, _bind
from tests.test_native_prefix_recipe import test_native_campaign_materialization_window_and_final_review_stop as campaign_check
from tests.test_native_prefix_coordinator import PrefixHarness, _prepared, equal_tree, digest, trainer
from gx1.contracts import unified_exit_bounded_val_cohort_v1 as cohorts


@pytest.fixture
def learning_scope(initial_scope, tmp_path):
    policy, recipe, files, seal = initial_scope
    initial_artifacts = recipe.pop('chronological_initial_measurement')
    policy.pop('chronological_initial_measurement')
    measurement = json.loads(Path(initial_artifacts['measurement_binding_result']['path']).read_text())
    observations = {}
    for role in ['train','control']:
        observations[role] = _write(tmp_path/(role+'.json'), {
            'role':role,'optimizer_steps':0,'model_state_sha256':'a'*64,'target_model_state_sha256':'a'*64,
            'test_data_used':False,'cohort':{'plan':recipe['chronological_prefix']['design'],
                'measurement_coordinates':measurement['coordinate_result'],
                'measurement_role':role,'entry_row_indices':list(range(256))}})
    result = {'schema_version':'gx1_native_prefix_initial_measurement_v1',
        'decision':'FROZEN_INITIAL_TARGETS_AND_PREDICTIONS_READY_NO_LEARNING_MEASURED',
        **initial_artifacts,'optimizer_steps':0,'model_state_sha256':'a'*64,'target_model_state_sha256':'a'*64,
        'teacher_refreshed':False,'economic_rollout':False,'test_data_used':False,'observations':observations}
    audit = {'schema_version':'gx1_native_prefix_initial_measurement_audit_v1',
        'decision':'FROZEN_INITIAL_MEASUREMENT_VERIFIED_NO_LEARNING_MEASURED',
        'result':_write(tmp_path/'before.json',result),'optimizer_steps':0,'test_data_used':False,
        'fresh_model_optimizer_ema_scheduler_exactly_preserved':True,'saved_cpu_python_numpy_rng_exactly_preserved':True,
        'receipt':_write(tmp_path/'terminal.json',{'guard_decision':'PASS','outcome':'RESUMABLE',
            'trainer_guard_exit_code':0,'progress_observer_exit_code':0,'test_data_used':False})}
    binding = _write(tmp_path/'audit.json',audit)
    recipe['chronological_learning_measurement'] = binding
    policy['chronological_learning_run'] = {'chronological_prefix':recipe['chronological_prefix'],
        'run_id':recipe['run_id'],'out_bundle_dir':recipe['out_bundle_dir'],
        'source_bindings_sha256':recipe['source_bindings_sha256'],'optimizer_steps':256,
        'maximum_trained_entry_rows':4096,'max_invocations':3,'final_model_variant':'ONLINE',
        'teacher_refresh_allowed':False,'full_epoch_training_allowed':False,'full_val_allowed':False,
        'test_data_used':False,'chronological_learning_measurement':binding}
    seal()
    return policy,recipe,files,seal


def test_learning_scope_binds_initial_evidence_and_fixed_budget(learning_scope):
    _, recipe, _, _ = learning_scope
    for invocation in [1,2,3]: assert native.require_native_run_scope(recipe,invocation_number=invocation) == 256
    value = native.require_chronological_learning_measurement(recipe)
    assert value['initialization']['optimizer_state_empty'] is True
    assert set(value['initial_measurement']['observations']) == {'train','control'}
    budget = {'stop_after_optimizer_steps':256,'stop_after_completed_val_epochs':None,'max_invocation_seconds':12000}
    assert native.require_native_run_scope(recipe,execution_budget=budget) == 256
    for steps in [0,255,257]:
        with pytest.raises(RuntimeError,match='PREFIX_BUDGET'):
            native.require_native_run_scope(recipe,execution_budget={**budget,'stop_after_optimizer_steps':steps})


@pytest.mark.parametrize('fault',['unbound_policy','changed_audit','changed_observation','changed_design','mixed_scope'])
def test_learning_admission_rejects_changed_frozen_evidence(learning_scope, fault):
    policy,recipe,_,seal = learning_scope
    ab = recipe['chronological_learning_measurement']; audit = json.loads(Path(ab['path']).read_text())
    if fault == 'unbound_policy': policy['chronological_learning_run'].pop('chronological_learning_measurement')
    elif fault == 'changed_audit': Path(ab['path']).write_text('{}')
    elif fault == 'mixed_scope': policy['chronological_initial_measurement'] = {}
    else:
        result = json.loads(Path(audit['result']['path']).read_text())
        ob = result['observations']['control']
        if fault == 'changed_observation': Path(ob['path']).write_text('{}')
        else:
            observation = json.loads(Path(ob['path']).read_text()); observation['cohort']['plan'] = {'wrong':True}
            result['observations']['control'] = _write(Path(ob['path']),observation)
            audit['result'] = _write(Path(audit['result']['path']),result)
            binding = _write(Path(ab['path']),audit)
            recipe['chronological_learning_measurement'] = binding
            policy['chronological_learning_run']['chronological_learning_measurement'] = binding
    seal()
    with pytest.raises(RuntimeError): native.require_native_run_scope(recipe,invocation_number=1)


@pytest.mark.parametrize('fault',[None,'false','unbound','missing_measurement','initial','diagnostic','no_prefix'])
def test_train_only_measurement_is_explicitly_bound(learning_scope,fault):
    policy,recipe,_,seal=learning_scope
    recipe['chronological_train_only_measurement']=False if fault=='false' else True
    if fault!='unbound':policy['chronological_learning_run']['chronological_train_only_measurement']=True
    if fault=='missing_measurement':recipe.pop('chronological_learning_measurement')
    if fault=='initial':recipe['chronological_initial_measurement']={}
    if fault=='diagnostic':recipe['entry_gradient_diagnostic']={}
    if fault=='no_prefix':recipe.pop('chronological_prefix')
    seal()
    if fault:
        with pytest.raises(RuntimeError):native.require_native_run_scope(recipe,invocation_number=1)
    else:
        assert native.require_native_run_scope(recipe,invocation_number=1)==256


def test_learning_campaign_stops_for_review_at_fixed256(learning_scope,monkeypatch,tmp_path):
    campaign_check(learning_scope,monkeypatch,tmp_path)


@pytest.mark.parametrize('steps,reason',[(128,'invocation_wall_limit'),(256,'optimizer_step_ceiling')])
def test_native_dispatch_restores_initial_state_and_measures_only_final_online(learning_scope,monkeypatch,tmp_path,steps,reason):
    _,recipe,files,seal = learning_scope; rb=seal(); seen=[]
    budget={'stop_after_optimizer_steps':256,'stop_after_completed_val_epochs':None,'max_invocation_seconds':12000}
    monkeypatch.setattr(runner.trainer,'_require_cuda_trainer_guard_execution',lambda **kw:seen.append('guard'))
    monkeypatch.setattr(runner.trainer,'_resolve_device',lambda _:torch.device('cpu'))
    monkeypatch.setattr(runner,'_require_native_full_train_recipe',lambda *a:(recipe,files,{}))
    monkeypatch.setattr(runner.launch_owner,'require_candidate_execution_budget',lambda *a,**kw:budget)
    monkeypatch.setattr(runner,'_build_bound_full_train_components',lambda **kw:{'chronological_prefix':kw['chronological_prefix']})
    monkeypatch.setattr(runner,'_restore_prefix_initial_measurement_state',lambda **kw:seen.append('restore'))
    def train(**kw):
        assert kw['execution_budget']['stop_after_optimizer_steps']==256
        seen.append('native_train')
        raise runner.trainer._CandidateExecutionPaused({'reason':reason,'global_optimizer_steps':steps})
    monkeypatch.setattr(runner,'_run_bound_full_train_candidate',train)
    def measure(**kw):
        assert kw['optimizer_steps']==256 and kw['scope']['initial_measurement']['optimizer_steps']==0
        seen.append('final_online'); return {'path':'synthetic','sha256':'a'*64}
    monkeypatch.setattr(runner,'_run_prefix_initial_measurement',measure)
    receipt=tmp_path/'pause.json';receipt.write_text('{}')
    monkeypatch.setattr(runner.trainer,'_write_candidate_execution_pause_receipt',lambda *a,**kw:receipt)
    monkeypatch.setattr(runner,'_native_resume_state',lambda **kw:{'global_optimizer_steps':steps})
    result=runner.run_guarded_native_candidate_invocation(recipe_path=Path(rb['path']),recipe_file_sha256=rb['sha256'],
        execution_budget_path=tmp_path/'budget.json',execution_budget_file_sha256='b'*64)
    assert seen==['guard','restore','native_train']+(['final_online'] if steps==256 else [])
    assert ('chronological_final_measurement' in result['pause']) == (steps==256)


@pytest.mark.parametrize('derived',[pytest.param(False,id='original'),pytest.param(True,id='derived')])
@pytest.mark.parametrize('fault',[None,'target','cohort'])
@pytest.mark.parametrize('train_only',[False,True])
def test_final_online_uses_identical_initial_targets_and_preserves_trained_session(tmp_path,monkeypatch,fault,train_only,derived):
    monkeypatch.setattr(trainer, "_copy_frozen_prefix_reference_model", lambda m: copy.deepcopy(m).eval().requires_grad_(False))
    if derived and not train_only: pytest.skip('Derived targets require the TRAIN-only scope.')
    artifacts=tmp_path/'artifacts';artifacts.mkdir()
    h=PrefixHarness(tmp_path/'run',_prepared(artifacts));output=h.root/'CANDIDATE'
    initial,pause=h.run_prefix(output,0)
    initial_hash=trainer._model_state_sha256(initial)
    scope_value={'initialization':{'online_model_state_sha256':initial_hash},
        'measurement':{'coordinate_result':{'test':True}},
        'artifacts':{'initialization_result':{'test':True},'measurement_binding_result':{'test':True},
                    'initial_measurement_audit':{'test':True}}}
    phase={'final':False}
    measured_roles=[]
    def cohort(*a,role):
        value={'role':role,'parent_entry_row_indices':list(range(256)),'entry_row_indices':list(range(256))}
        if phase['final'] and fault=='cohort':value['changed']=True
        return value
    monkeypatch.setattr(cohorts,'build_chronological_measurement_cohort',cohort)
    def measure(**kw):
        if phase['final'] and train_only:assert kw['evaluation_cohort']['role']=='train'
        measured_roles.append(kw['evaluation_cohort']['role'])
        assert not kw['model'].training and not kw['candidate_target_model'].training
        assert trainer._model_state_sha256(kw['candidate_target_model'])==initial_hash
        prediction=float(kw['model'].weight[0,0].detach())
        delta=1 if phase['final'] and fault=='target' else 0
        entry=[{'entry_row_index':i,'parent_entry_row_index':i,'predicted_q_bps':[prediction,prediction,0],
                'target_q_bps':[i/100+delta,-i/100,0],'target_valid':[True]*3} for i in range(256)]
        if phase['final'] and derived:
            for row in entry: row['target_q_bps'][1] -= 7
        anchor=[{'entry_row_index':i,'state_index':0,'prediction_hold_bps':[prediction]*2,
                 'target_hold_bps':[i/10,-i/10]} for i in range(256)]
        sampled=[{**row,'state_index':j} for row in anchor for j in range(4)]
        torch.rand(3)
        return None,{'bounded_entry_observations':entry,'bounded_exit_anchor_observations':anchor,
                     'bounded_exit_sampled_observations':sampled},None
    monkeypatch.setattr(runner.val,'_entry_representations',measure)
    components={**h.last_kwargs,'train_probe_ds':object()}
    before=runner._run_prefix_initial_measurement(components=components,scope=scope_value,
        recipe={'chronological_prefix':h.data.value},output=output,device=torch.device('cpu'),
        invocation_started=time.monotonic(),pause_evidence=pause)
    scope_value['artifacts']['initial_measurement_result']=before
    scope_value['initial_measurement']=json.loads(Path(before['path']).read_text())
    if derived:
        saved=json.loads(Path(scope_value['initial_measurement']['observations']['train']['path']).read_text())
        entry_rows=copy.deepcopy(saved['diagnostics']['bounded_entry_observations'])
        for row in entry_rows: row['target_q_bps'][1] -= 7
        scope_value['entry_baseline']={'entry_observations':entry_rows}
        scope_value['entry_baseline_result']={'path':'derived-result.json','sha256':'d'*64}
    trained,pause=h.run_prefix(output,256,expected_pointer=digest(h.pointer(output)))
    model_hash=trainer._model_state_sha256(trained); assert model_hash!=initial_hash
    checkpoint=h.state(output); pointer=h.pointer(output).read_bytes()
    rng=trainer._attended_session_rng_state(device=torch.device('cpu'));phase['final']=True
    components={**h.last_kwargs,'train_probe_ds':object()}
    kwargs=dict(components=components,scope=scope_value,recipe={'chronological_prefix':h.data.value},
        output=output,device=torch.device('cpu'),invocation_started=time.monotonic()-1800,
        pause_evidence=pause,optimizer_steps=256)
    if train_only:kwargs['recipe']['chronological_train_only_measurement']=True
    measured_roles.clear()
    if fault:
        with pytest.raises(RuntimeError,match='FROZEN_TARGET_CHANGED|COHORT_OR_TEACHER_CHANGED'):
            runner._run_prefix_initial_measurement(**kwargs)
    else:
        after=runner._run_prefix_initial_measurement(**kwargs);result=json.loads(Path(after['path']).read_text())
        assert result['optimizer_steps']==256 and result['selected_model_variant']=='ONLINE'
        assert measured_roles==(['train'] if train_only else ['train','control'])
        assert result['measurement_roles']==measured_roles and set(result['observations'])==set(measured_roles)
        assert result['frozen_targets_exactly_preserved'] is True and result['target_model_state_sha256']==initial_hash
        assert result['model_state_sha256']==model_hash and result['initial_measurement']==before
        if derived: assert result['derived_entry_target_baseline']==scope_value['entry_baseline_result']
    assert trained.training and h.pointer(output).read_bytes()==pointer
    equal_tree(h.state(output),checkpoint)
    equal_tree(trainer._attended_session_rng_state(device=torch.device('cpu')),rng)
    assert len(h.batches)==256 and h.validation_batches==0 and len(set(h.teacher_hashes))==1


def test_learning_accepts_train_only_initial_but_rejects_old_online_function(learning_scope):
    policy, recipe, _, seal = learning_scope
    ab = recipe['chronological_learning_measurement']
    audit = json.loads(Path(ab['path']).read_text())
    result = json.loads(Path(audit['result']['path']).read_text())
    result['observations'].pop('control')
    audit['result'] = _write(Path(audit['result']['path']), result)
    recipe['chronological_learning_measurement'] = _write(Path(ab['path']), audit)
    policy['chronological_learning_run']['chronological_learning_measurement'] = recipe['chronological_learning_measurement']
    seal()
    with pytest.raises(RuntimeError, match='INITIAL_MEASUREMENT_INVALID'):
        native.require_native_run_scope(recipe)
    recipe['chronological_train_only_measurement'] = True
    policy['chronological_learning_run']['chronological_train_only_measurement'] = True
    seal()
    assert native.require_native_run_scope(recipe, invocation_number=1) == 256
    source = Path(recipe['source_repo']) / 'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py'
    source.write_text('changed online function; initial tensor hash unchanged')
    with pytest.raises(RuntimeError, match='INITIAL_MODEL_SOURCE_CHANGED'):
        native.require_native_run_scope(recipe)
