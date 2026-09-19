from pathlib import Path
import hashlib
import json
import subprocess
import pytest
from scripts.collect_gx1_handover_readonly import native_status, next_run_readiness


def _git(repo, *args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


@pytest.fixture
def fixture(tmp_path):
    repo=tmp_path/'repo';repo.mkdir()
    _git(repo,'init','-q','-b','work/gx1-current')
    _git(repo,'config','user.name','Fixture');_git(repo,'config','user.email','fixture@example.invalid')
    (repo/'source.txt').write_text('source\n')
    _git(repo,'add','.');_git(repo,'commit','-qm','fixture')
    artifact=tmp_path/'artifact.json';artifact.write_text('{"decision":"COMPLETE_WITH_RIGHT_CENSORING","entry_exit_policy_metrics":{"full_cohort_authoritative":false}}')
    runtime=tmp_path/'runtime';runtime.mkdir()
    session=tmp_path/'session';session.mkdir()
    pointer={'session_contract_sha256':'session-digest','phase':'train','epoch_index':1,'global_optimizer_steps':19908}
    (session/'CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json').write_text(json.dumps(pointer))
    b={'path':str(artifact),'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()}
    binding={'schema_version':'gx1_native_handover_binding_v1','source_repo':str(repo),'source_commit':_git(repo,'rev-parse','HEAD'),
             'runtime_root':str(runtime),'training_session':str(session),'session_contract_sha256':'session-digest',
             'immutable_artifacts':{'plan':b},'completed_val_result':b,'observed_stop':{'reason':'user_requested_review'}}
    path=tmp_path/'COMPLETED_RUN.json';path.write_text(json.dumps(binding))
    return repo,path,artifact,pointer


def test_completed_val_remains_visible_after_pointer_advanced(fixture,monkeypatch):
    repo,path,artifact,pointer=fixture
    monkeypatch.setattr('scripts.collect_gx1_handover_readonly._native_processes',lambda _:[])
    out=native_status(path)
    assert out['checkpoint']==pointer
    assert out['process_observation']=='NO_NATIVE_PROCESS_OBSERVED'
    assert out['recorded_operator_stop']['reason']=='user_requested_review'
    assert out['completed_val']['decision']=='COMPLETE_WITH_RIGHT_CENSORING'
    assert not out['completed_val']['entry_exit_policy_metrics']['full_cohort_authoritative']
    assert out['state_payload_rehashed'] is False and out['test_accessed'] is False
    artifact.write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):native_status(path)


def _policy(repo):
    source=Path(__file__).resolve().parents[1]
    p=json.loads((source/'NEXT_RUN_POLICY.json').read_text())
    p['canonical_source_repo']=str(repo)
    p['required_evidence']={role:None for role in ('risk_objective','checkpoint_transition','learning_calibration',
                                                 'gpu_batch256_parity','end_to_end_throughput','resume_equivalence')}
    clock=repo/'clock.ps1';clock.write_text('fixture')
    p['gpu_clock_launcher']={'path':str(clock),'sha256':hashlib.sha256(clock.read_bytes()).hexdigest()}
    path=repo/'NEXT_RUN_POLICY.json';path.write_text(json.dumps(p))
    _git(repo,'add','.');_git(repo,'commit','-qm','policy fixture')
    return path,p


def test_current_work_is_observed_separately_from_completed_run(fixture, monkeypatch):
    repo, binding, _, old_pointer = fixture
    current = binding.parent / 'current'; current.mkdir()
    new_pointer = {'global_optimizer_steps': 5809, 'phase': 'train'}
    (current/'CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json').write_text(json.dumps(new_pointer))
    status = {'status': 'RECORDED_IDLE', 'source_repo': str(repo),
              'source_commit': 'training-source', 'session_directory': str(current),
              'next_action': 'Measure learning before larger training',
              'learning_gate': {'decision': 'NOT_PROVEN'}}
    (binding.parent/'RUNNING_NATIVE_CALIBRATION.json').write_text(json.dumps(status))
    monkeypatch.setattr('scripts.collect_gx1_handover_readonly._native_processes',
                        lambda _: [{'pid': '123'}])
    out = native_status(binding)
    assert out['checkpoint'] == old_pointer
    assert out['current_work']['checkpoint'] == new_pointer
    assert out['current_work']['recorded_status'] == 'RECORDED_IDLE'
    assert out['current_work']['process_observation'] == 'RUNNING'
    assert out['current_work']['observation_is_run_authority'] is False
    assert out['current_work']['full_epoch_training_allowed'] is False
    assert out['current_work']['learning_gate']['decision'] == 'NOT_PROVEN'


def test_current_work_source_only_does_not_probe_processes_or_checkpoints(fixture, monkeypatch):
    repo, binding, *_ = fixture
    (binding.parent/'RUNNING_NATIVE_CALIBRATION.json').write_text(json.dumps({
        'status': 'RECORDED_IDLE', 'source_repo': str(repo),
        'session_directory': str(binding.parent/'missing-session')}))
    def unexpected(_):
        raise AssertionError('source-only must not inspect native processes')
    monkeypatch.setattr('scripts.collect_gx1_handover_readonly._native_processes', unexpected)
    out = native_status(binding, source_only=True)
    assert 'checkpoint' not in out['current_work']
    assert 'process_observation' not in out['current_work']


@pytest.mark.parametrize('scope_kind',['learning','diagnostic','initial','frozen_train_policy'])
@pytest.mark.parametrize('case',['active','before_first_checkpoint','changed_recipe'])
def test_current_policy_handover_never_reports_previous_run_checkpoint(fixture,monkeypatch,case,scope_kind):
    diagnostic = scope_kind in {"diagnostic", "frozen_train_policy"}
    from scripts.collect_gx1_handover_readonly import _current_work_status
    repo,binding,*_=fixture
    run=binding.parent/'CURRENT_RUN';run.mkdir();output=run/'CANDIDATE_BUNDLE'
    session=run/'.gx1-candidate-training-session.CANDIDATE_BUNDLE';session.mkdir()
    new={'global_optimizer_steps':128,'phase':'train'}
    if case!='before_first_checkpoint':
        (session/'CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json').write_text(json.dumps(new))
    (binding.parent/'RUNNING_NATIVE_CALIBRATION.json').write_text(json.dumps({
        'status':'RECORDED_BEFORE_START','source_repo':str(repo),'source_commit':'old-source',
        'session_directory':str(binding.parent/'session'),'next_action':'old note'}))
    scope={'run_id':run.name,'out_bundle_dir':str(output)}
    (binding.parent/'NEXT_RUN_POLICY.json').write_text(json.dumps({{'diagnostic':'entry_gradient_diagnostic','learning':'chronological_learning_run','initial':'chronological_initial_measurement','frozen_train_policy':'frozen_train_policy_evaluation'}[scope_kind]:scope}))
    recipe=run/'recipe.json';recipe.write_text(json.dumps({**scope,'source_repo':str(repo),'source_commit':'current-source'}))
    (run/'PREPARATION_RESULT.json').write_text(json.dumps({'source_commit':'current-source',
        'recipe':{'path':str(recipe),'sha256':hashlib.sha256(recipe.read_bytes()).hexdigest()},
        'runtime_root':str(binding.parent/'runtime')}))
    monkeypatch.setattr('scripts.collect_gx1_handover_readonly._native_processes',lambda _:[{'pid':'780'}])
    if case=='changed_recipe':
        recipe.write_text('{}')
        with pytest.raises(ValueError,match='hash mismatch'):_current_work_status(binding.parent,source_only=False)
    else:
        out=_current_work_status(binding.parent,source_only=False)
        if diagnostic:
            assert out['checkpoint']['global_optimizer_steps']==19908
            assert out['checkpoint_selection']=='preserved_training_origin_not_diagnostic_progress'
            assert out['diagnostic_source_commit']=='current-source'
            assert out['diagnostic_artifact_root']==str(run)
        else:
            assert out['checkpoint']==(None if case=='before_first_checkpoint' else new)
            assert out['session_directory']==str(session) and out['training_source_commit']=='current-source'
        assert out['declared_run_id']=='CURRENT_RUN' and 'do not relaunch' in out['next_action']


@pytest.mark.parametrize('latest_key', ['completed_main_encoder_fixed256','completed_entry_fuse_fixed256','completed_convergence512','completed_frozen_train_policy'])
def test_closed_scope_keeps_latest_terminal_measurement(fixture, monkeypatch, latest_key):
    from scripts.collect_gx1_handover_readonly import _current_work_status
    repo, binding, *_ = fixture
    session = binding.parent / 'completed-main'; session.mkdir()
    steps = 512 if latest_key in {'completed_convergence512','completed_frozen_train_policy'} else 256
    pointer = {'global_optimizer_steps': steps, 'phase': 'train'}
    (session/'CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json').write_text(json.dumps(pointer))
    receipt = session/'receipt.json'; receipt.write_text('{"guard_decision":"PASS"}')
    result = session/'result.json'; result.write_text(json.dumps({'optimizer_steps':steps}))
    def bind(path):
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    latest = {'run_id': 'MAIN256', 'receipt': bind(receipt), 'result': bind(result),
              'learning_review_complete': False}
    (binding.parent/'NEXT_RUN_POLICY.json').write_text('{"training_enabled":false}')
    (binding.parent/'RUNNING_NATIVE_CALIBRATION.json').write_text(json.dumps({
        'status': 'COMPLETE_REVIEW_PENDING', 'source_repo': str(repo),
        'source_commit': 'training-source', 'session_directory': str(session),
        'next_action': 'Review saved outputs; no relaunch',
        'completed_residual_normalized_fixed256': {'run_id': 'OLD'},
        'completed_main_encoder_fixed256': {'run_id':'OLDER_MAIN'},
        **({'completed_entry_fuse_fixed256':{'run_id':'OLDER_FUSE'}} if latest_key=='completed_convergence512' else {}),
        **({'completed_convergence512':{'run_id':'OLDER_512'}} if latest_key=='completed_frozen_train_policy' else {}),
        latest_key: latest}))
    monkeypatch.setattr('scripts.collect_gx1_handover_readonly._native_processes', lambda _: [])
    out = _current_work_status(binding.parent, source_only=False)
    assert out['latest_completed_native'] == latest
    assert out['last_completed_run_id'] == 'MAIN256' and 'declared_run_id' not in out
    assert out['checkpoint'] == pointer and out['training_source_commit'] == 'training-source'
    assert out['latest_terminal_receipt']['guard_decision'] == 'PASS'
    assert out['latest_final_measurement']['optimizer_steps'] == steps
    assert out['next_action'] == 'Review saved outputs; no relaunch'
    assert out['observation_is_run_authority'] is False
    receipt.write_text('{}')
    with pytest.raises(ValueError, match='Completed native evidence hash mismatch'):
        _current_work_status(binding.parent, source_only=False)
    assert 'latest_terminal_receipt' not in _current_work_status(binding.parent, source_only=True)


def test_enabled_flag_alone_cannot_bypass_missing_proofs(fixture):
    repo,*_=fixture;path,p=_policy(repo)
    p['training_enabled']=True;path.write_text(json.dumps(p))
    out=next_run_readiness(repo)
    assert out['decision']=='BLOCKED'
    for role in ('risk_objective','gpu_batch256_parity','end_to_end_throughput','resume_equivalence'):
        assert role+'_missing' in out['blocked_reasons']
    assert out['training_started'] is False


@pytest.mark.parametrize('field,value',[('policy_batch_size',128),('cpu_pipeline_workers',4),('max_wall_seconds',4200)])
def test_slower_profile_is_rejected(fixture,field,value):
    repo,*_=fixture;path,p=_policy(repo)
    p['required_val_profile'][field]=value;path.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='PERFORMANCE_PROFILE_INVALID'):next_run_readiness(repo)


@pytest.mark.parametrize('module',['gx1.models.entry_v10.entry_v10_ctx_train_v3','gx1.scripts.run_unified_exit_random_access_fixed_step_v1','gx1.scripts.run_unified_exit_random_access_val_v1'])
def test_retired_training_routes_stop_before_gpu(module):
    repo=Path(__file__).resolve().parents[1]
    result=subprocess.run(['bash',str(repo/'scripts/gx1_capped_run.sh'),'--class','trainer','--mem','20G','--swap','512M','--',str(repo/'.venv/bin/python'),'-m',module],cwd=repo,capture_output=True,text=True)
    assert result.returncode==75
    assert 'retired training route' in result.stderr


def test_native_route_cannot_start_while_review_is_pending():
    repo=Path(__file__).resolve().parents[1]
    result=subprocess.run(['bash',str(repo/'scripts/gx1_capped_run.sh'),'--class','trainer','--mem','20G','--swap','512M','--',str(repo/'.venv/bin/python'),'-m','gx1.scripts.run_unified_exit_native_candidate_window_v1'],cwd=repo,capture_output=True,text=True)
    assert result.returncode==75
    assert 'campaign' in result.stderr.lower()


def test_handover_has_no_legacy_fallback():
    source=Path(__file__).resolve().parents[1]/'scripts/gx1_handover.sh'
    text=source.read_text()
    assert 'COMPLETED_RUN.json' in text and 'NEXT_RUN_POLICY.json' in text
    assert 'PROJECT_STATE_xau_direction_launch' not in text
    assert '--require-next-run' not in text


@pytest.fixture
def bound_window(fixture, monkeypatch):
    from gx1.contracts import unified_exit_native_candidate_campaign_v1 as owner
    repo, *_ = fixture
    policy_path, policy = _policy(repo)
    # Bind the original economics-origin fixture, not the current operator run.
    for field in ('exit_backup_steps', 'exit_value_initialization',
                  'train_population_scope', 'gradient_clipping_policy', 'learning_gate',
                  'exit_reference_policy', 'reference_learning_plan'):
        policy.pop(field, None)
    monkeypatch.setattr(owner, '__file__', str(repo / 'gx1/contracts/native.py'))

    def write(name, value):
        path = repo.parent / name
        path.write_text(json.dumps(value))
        return {'path': str(path), 'sha256': owner.file_sha256(path)}

    risk = {'decision': 'PASS', 'test_data_used': False, 'maximum_holding_seconds': None,
            'absolute_loss_limit_bps': None, 'reward_accounting': 'liquidation_advantage_v1',
            'economics_objective_schema': 'gx1_unified_exit_economics_objective_v4'}
    policy['training_enabled'] = False
    policy['required_evidence']['risk_objective'] = write('risk.json', risk)
    policy['native_learning_calibration'] = {
        'schema_version': 'gx1_native_learning_calibration_scope_v1', 'optimizer_step_ceilings': [16, 32],
        'full_epoch_training_allowed': False, 'test_data_used': False,
    }
    recipe = {
        'schema_version': 'gx1_unified_exit_random_access_full_train_recipe_v1', 'profile': 'candidate',
        'test_data_used': False, 'source_repo': str(repo), 'source_bindings_sha256': 'a' * 64,
        'val_limits': policy['required_val_profile'], 'trainer_cli': {'batch_size': 16},
        'candidate_resume_origin': {'schema_version': 'gx1_candidate_economics_transition_origin_v1',
            'contract': write('origin-contract.json', {'fixture': 'contract'}),
            'pointer': write('origin-pointer.json', {'fixture': 'pointer'})},
        'files': {'economics_readiness': write('economics.json', {'economics_objective_contract': {
            'schema_version': risk['economics_objective_schema'], 'reward_accounting': risk['reward_accounting'],
            'contract_sha256': 'c' * 64}})},
    }
    window = {'schema_version': owner.WINDOW_SCHEMA, 'invocation_number': 1,
              'max_invocation_seconds': 12000, 'test_data_used': False,
              **{name: str(repo.parent / name) for name in
                 ('budget_path', 'progress_path', 'campaign_cursor_path', 'training_session_directory')}}

    def publish():
        policy_path.write_text(json.dumps(policy))
        if _git(repo, 'status', '--porcelain'):
            _git(repo, 'add', '.'); _git(repo, 'commit', '-qm', 'bound policy fixture')
        recipe['source_commit'] = _git(repo, 'rev-parse', 'HEAD')
        recipe['next_run_policy'] = {'path': str(policy_path), 'sha256': owner.file_sha256(policy_path)}
        recipe['recipe_sha256'] = owner.native_sha256({k: v for k, v in recipe.items() if k != 'recipe_sha256'})
        window['recipe'] = write('recipe.json', recipe)
        window['policy_sha256'] = owner.canonical_sha256({k: v for k, v in window.items() if k != 'policy_sha256'})
        binding = write('window.json', window)
        return {'native_window_policy': Path(binding['path']), 'native_window_policy_file_sha256': binding['sha256']}

    return repo, policy, recipe, window, write, publish


@pytest.mark.parametrize('invocation,ceiling', [(1, 16), (2, 32)])
def test_bound_native_window_admits_only_declared_calibration(bound_window, invocation, ceiling):
    repo, policy, recipe, window, write, publish = bound_window
    window['invocation_number'] = invocation
    arguments = publish()
    result = next_run_readiness(repo, **arguments)
    assert result['decision'] == 'READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES'
    assert result['blocked_reasons'] == []
    assert result['native_window_scope']['optimizer_step_ceiling'] == ceiling
    assert result['native_window_scope']['full_epoch_training_allowed'] is False
    assert result['training_started'] is False
    unbound = next_run_readiness(repo)
    assert unbound['decision'] == 'BLOCKED'
    assert 'native_window_binding_required' in unbound['blocked_reasons']
    assert 'operator_stop_not_resolved' in unbound['blocked_reasons']


@pytest.mark.parametrize('which', ['window', 'recipe', 'policy'])
def test_bound_native_window_rejects_modified_hash_bound_file(bound_window, which):
    repo, policy, recipe, window, write, publish = bound_window
    arguments = publish()
    path = {'window': arguments['native_window_policy'], 'recipe': Path(window['recipe']['path']),
            'policy': Path(recipe['next_run_policy']['path'])}[which]
    path.write_text(path.read_text() + '\n')
    with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
        next_run_readiness(repo, **arguments)


@pytest.mark.parametrize('change', ['unbounded', 'third-window', 'full-without-proofs'])
def test_calibration_exception_cannot_become_unbounded_training(bound_window, change):
    repo, policy, recipe, window, write, publish = bound_window
    if change == 'unbounded':
        policy['native_learning_calibration']['optimizer_step_ceilings'] = [16, None]
    elif change == 'third-window':
        window['invocation_number'] = 3
    else:
        policy['training_enabled'] = True
    arguments = publish()
    with pytest.raises(RuntimeError):
        next_run_readiness(repo, **arguments)


def test_bound_full_scope_uses_all_owner_roles_and_native_profile(bound_window):
    repo, policy, recipe, window, write, publish = bound_window
    roles = ('checkpoint_transition', 'learning_calibration', 'gpu_batch256_parity',
             'end_to_end_throughput', 'resume_equivalence')
    proof = {'decision': 'PASS', 'test_data_used': False,
             'economics_objective_contract_sha256': 'c' * 64, 'source_bindings_sha256': 'a' * 64,
             'training_origin_pointer_sha256': recipe['candidate_resume_origin']['pointer']['sha256'],
             'native_val_profile': recipe['val_limits']}
    for role in roles:
        policy['required_evidence'][role] = write(role + '.json', {**proof, 'evidence_role': role})
    policy['training_enabled'] = True
    result = next_run_readiness(repo, **publish())
    assert result['decision'] == 'READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES'
    assert result['native_window_scope']['optimizer_step_ceiling'] is None
    assert result['native_window_scope']['full_epoch_training_allowed'] is True
    assert next_run_readiness(repo)['decision'] == 'BLOCKED'
    for role in roles:
        original = policy['required_evidence'][role]
        policy['required_evidence'][role] = write('invalid-' + role + '.json', {**proof, 'evidence_role': role,
                                                            'source_bindings_sha256': 'f' * 64})
        arguments = publish()
        with pytest.raises(RuntimeError, match='EVIDENCE_NOT_PASS'):
            next_run_readiness(repo, **arguments)
        policy['required_evidence'][role] = original


def test_bound_calibration_does_not_override_dirty_source_or_clock(bound_window):
    repo, policy, recipe, window, write, publish = bound_window
    arguments = publish()
    (repo / 'source.txt').write_text('uncommitted source change\n')
    result = next_run_readiness(repo, **arguments)
    assert result['decision'] == 'BLOCKED'
    assert 'source_not_clean_committed' in result['blocked_reasons']
    (repo / 'clock.ps1').write_text('unbound launcher bytes')
    with pytest.raises(ValueError, match='CLOCK_LAUNCHER_MISMATCH'):
        next_run_readiness(repo, **arguments)


def test_bound_calibration_keeps_risk_and_canonical_branch_checks(bound_window):
    repo, policy, recipe, window, write, publish = bound_window
    arguments = publish()
    _git(repo, 'checkout', '-qb', 'wrong-branch')
    with pytest.raises(ValueError, match='CANONICAL_BRANCH_REQUIRED'):
        next_run_readiness(repo, **arguments)
    _git(repo, 'checkout', '-q', 'work/gx1-current')
    risk = json.loads(Path(policy['required_evidence']['risk_objective']['path']).read_text())
    risk['maximum_holding_seconds'] = 30
    policy['required_evidence']['risk_objective'] = write('risk.json', risk)
    arguments = publish()
    with pytest.raises(RuntimeError, match='RISK_OBJECTIVE_INVALID'):
        next_run_readiness(repo, **arguments)


@pytest.mark.parametrize('arguments', [{'native_window_policy': Path('/missing')},
                                     {'native_window_policy_file_sha256': 'a' * 64}])
def test_native_window_keyword_pair_is_mandatory(fixture, arguments):
    repo, *_ = fixture
    with pytest.raises(ValueError, match='BINDING_PAIR_REQUIRED'):
        next_run_readiness(repo, **arguments)


@pytest.mark.parametrize('arguments', [
    ['--require-next-run', '--native-window-policy', '/missing'],
    ['--require-next-run', '--native-window-policy-file-sha256', 'a' * 64],
    ['--native-window-policy', '/missing', '--native-window-policy-file-sha256', 'a' * 64],
])
def test_native_window_cli_requires_pair_and_require_next_run(monkeypatch, arguments):
    from scripts.collect_gx1_handover_readonly import main
    monkeypatch.setattr('sys.argv', ['collector', *arguments])
    with pytest.raises(SystemExit) as rejected:
        main()
    assert rejected.value.code == 2
