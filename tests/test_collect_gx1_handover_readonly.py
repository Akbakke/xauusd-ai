from pathlib import Path
import hashlib
import json
import subprocess
import pytest
from scripts.collect_gx1_handover_readonly import current_status, next_run_readiness


def _git(repo, *args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


@pytest.fixture
def fixture(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'work/gx1-current')
    _git(repo, 'config', 'user.name', 'Fixture')
    _git(repo, 'config', 'user.email', 'fixture@example.invalid')
    (repo / 'source.txt').write_text('source\n')
    _git(repo, 'add', '.')
    _git(repo, 'commit', '-qm', 'fixture')
    return repo


def _policy(repo):
    source=Path(__file__).resolve().parents[1]
    p=json.loads((source/'NEXT_RUN_POLICY.json').read_text())
    p['canonical_source_repo']=str(repo)
    terminal = repo.parent / 'terminal.json'
    terminal.write_text(json.dumps({'exit_code': 1, 'source_unchanged': True, 'test_data_used': False}))
    p['current_work'] = {
        'latest_terminal': {'path': str(terminal), 'sha256': hashlib.sha256(terminal.read_bytes()).hexdigest()},
        'terminal_exit_code': 1, 'source_unchanged_at_terminal': True,
        'full_benchmark_completed': False, 'sampler_selected': False,
    }
    p['required_evidence']={role:None for role in ('risk_objective','checkpoint_transition','learning_calibration',
                                                 'gpu_batch256_parity','end_to_end_throughput','resume_equivalence')}
    clock=repo/'clock.ps1';clock.write_text('fixture')
    p['gpu_clock_launcher']={'path':str(clock),'sha256':hashlib.sha256(clock.read_bytes()).hexdigest()}
    path=repo/'NEXT_RUN_POLICY.json';path.write_text(json.dumps(p))
    _git(repo,'add','.');_git(repo,'commit','-qm','policy fixture')
    return path,p


def test_current_terminal_replaces_historical_checkpoint_selection(fixture, monkeypatch):
    repo = fixture
    _policy(repo)
    monkeypatch.setattr('scripts.collect_gx1_handover_readonly._current_processes', lambda _: [])
    out = current_status(repo)
    assert out['latest_terminal_receipt']['exit_code'] == 1
    assert out['process_observation'] == 'NO_CURRENT_PYTHON_WORKLOAD_OBSERVED'
    assert not out['current_work']['full_benchmark_completed']
    assert not out['current_work']['sampler_selected']
    assert 'checkpoint' not in out
    assert out['test_accessed'] is False and out['state_payload_rehashed'] is False


@pytest.mark.parametrize('change', ['bytes', 'state', 'missing'])
def test_current_terminal_fails_closed(fixture, monkeypatch, change):
    repo = fixture
    policy_path, policy = _policy(repo)
    monkeypatch.setattr('scripts.collect_gx1_handover_readonly._current_processes', lambda _: [])
    binding = policy['current_work']['latest_terminal']
    terminal = Path(binding['path'])
    if change == 'bytes':
        terminal.write_text('{}')
    elif change == 'state':
        terminal.write_text(json.dumps({'exit_code': 0, 'source_unchanged': True, 'test_data_used': False}))
        binding['sha256'] = hashlib.sha256(terminal.read_bytes()).hexdigest()
        policy_path.write_text(json.dumps(policy))
    else:
        terminal.unlink()
    with pytest.raises(ValueError):
        current_status(repo)


def test_source_only_never_reads_terminal_or_processes(fixture, monkeypatch):
    repo = fixture
    path, policy = _policy(repo)
    Path(policy['current_work']['latest_terminal']['path']).unlink()
    def unexpected(_):
        raise AssertionError('source-only must not inspect workloads')
    monkeypatch.setattr('scripts.collect_gx1_handover_readonly._current_processes', unexpected)
    out = current_status(repo, source_only=True)
    assert 'latest_terminal_receipt' not in out
    assert 'current_processes' not in out


def test_current_policy_has_no_historical_fallback(fixture):
    path, policy = _policy(fixture)
    policy.pop('current_work')
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match='NO_HISTORICAL_FALLBACK'):
        current_status(fixture, source_only=True)


@pytest.mark.parametrize('kind', ['file', 'repo'])
def test_symlink_parent_cannot_hide_test_path(tmp_path, kind):
    from scripts.collect_gx1_handover_readonly import _regular_file, _repo
    sealed = tmp_path / 'TEST'
    sealed.mkdir()
    (sealed / 'terminal.json').write_text('{}')
    (sealed / 'repo').mkdir()
    alias = tmp_path / 'seemingly-safe'
    alias.symlink_to(sealed, target_is_directory=True)
    with pytest.raises(ValueError, match='canonical path'):
        if kind == 'file':
            _regular_file(str(alias / 'terminal.json'), label='terminal')
        else:
            _repo(str(alias / 'repo'))


@pytest.mark.parametrize('invocation', ['absolute', '.venv/bin/python', './.venv/bin/python'])
def test_current_process_observer_includes_external_benchmark_operator(fixture, monkeypatch, invocation):
    import scripts.collect_gx1_handover_readonly as collector
    repo = fixture
    prefix = (str(repo / '.venv/bin/python') if invocation == 'absolute' else invocation) + ' '
    snapshot = 'PID PPID ELAPSED PCPU RSS COMMAND\n'
    snapshot += '123 1 00:10 99.0 200 ' + prefix + '/outside/OPERATOR.py measure\n'
    snapshot += '124 1 00:10 99.0 200 /other/.venv/bin/python -m gx1.scripts.some_job\n'
    snapshot += '125 1 00:10 99.0 200 ' + prefix + '/outside/OPERATOR.py run\n'
    snapshot += '126 1 00:10 99.0 200 ' + prefix + '/outside/OPERATOR.py run\n'
    monkeypatch.setattr(collector.subprocess, 'run', lambda *_a, **_k: subprocess.CompletedProcess([], 0, snapshot))
    monkeypatch.setattr(collector.os, 'getpid', lambda: 999)
    def cwd(path):
        if path == '/proc/125/cwd':
            raise FileNotFoundError(path)
        if path == '/proc/126/cwd':
            return '/other'
        return str(repo)
    monkeypatch.setattr(collector.os, 'readlink', cwd)
    processes = collector._current_processes(repo)
    assert [p['pid'] for p in processes] == ['123']
    assert processes[0]['command'].endswith('OPERATOR.py measure')


def test_enabled_flag_alone_cannot_bypass_missing_proofs(fixture):
    repo=fixture;path,p=_policy(repo)
    p['training_enabled']=True;path.write_text(json.dumps(p))
    out=next_run_readiness(repo)
    assert out['decision']=='BLOCKED'
    for role in ('risk_objective','gpu_batch256_parity','end_to_end_throughput','resume_equivalence'):
        assert role+'_missing' in out['blocked_reasons']
    assert out['training_started'] is False


@pytest.mark.parametrize('field,value',[('policy_batch_size',128),('cpu_pipeline_workers',4),('max_wall_seconds',4200)])
def test_slower_profile_is_rejected(fixture,field,value):
    repo=fixture;path,p=_policy(repo)
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
    assert 'NEXT_RUN_POLICY.json' in text
    assert 'COMPLETED_RUN.json' not in text and 'RUNNING_NATIVE_CALIBRATION.json' not in text
    assert 'PROJECT_STATE_xau_direction_launch' not in text
    assert '--require-next-run' not in text


@pytest.fixture
def bound_window(fixture, monkeypatch):
    from gx1.contracts import unified_exit_native_candidate_campaign_v1 as owner
    repo = fixture
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


def test_current_cleanup_policy_has_no_legacy_calibration_exception(bound_window):
    repo, policy, recipe, window, write, publish = bound_window
    policy.pop('native_learning_calibration')
    assert policy['training_enabled'] is False
    assert policy['current_work']['full_benchmark_completed'] is False
    assert policy['current_work']['sampler_selected'] is False
    with pytest.raises(RuntimeError, match='NATIVE_TRAINING_BLOCKED_CALIBRATION_SCOPE_REQUIRED'):
        next_run_readiness(repo, **publish())


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
    repo = fixture
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


def _identity_repo(tmp_path):
    from scripts.collect_gx1_handover_readonly import LAUNCH_STATE_NAME
    repo=tmp_path/'identity'
    repo.mkdir()
    _git(repo,'init','-q','-b','work/gx1-current')
    _git(repo,'config','user.name','Fixture')
    _git(repo,'config','user.email','fixture@example.invalid')
    (repo/'.gitignore').write_text('*.log\n__pycache__/\n.pytest_cache/\n.venv/\n')
    state={'reviewed_local_runtime_exclusions':{'schema_version':'gx1_reviewed_local_runtime_exclusions_v1',
                                                'paths':['.claude/worktrees/','.env','.venv/']}}
    (repo/LAUNCH_STATE_NAME).write_text(json.dumps(state))
    (repo/'pkg').mkdir()
    (repo/'pkg'/'module.py').write_text('value = 1\n')
    _git(repo,'add','.')
    _git(repo,'commit','-qm','identity fixture')
    return repo


def test_source_identity_admits_reviewed_exclusions_and_caches(tmp_path):
    from scripts.collect_gx1_handover_readonly import source_identity
    repo=_identity_repo(tmp_path)
    (repo/'pkg'/'__pycache__').mkdir()
    (repo/'pkg'/'__pycache__'/'module.cpython-310.pyc').write_bytes(b'\0')
    (repo/'.pytest_cache').mkdir()
    (repo/'.pytest_cache'/'README.md').write_text('cache\n')
    (repo/'.venv').mkdir()
    (repo/'.venv'/'pyvenv.cfg').write_text('home = /usr/bin\n')
    identity=source_identity(repo)
    assert identity['source_identity_gate']=='READY_CLEAN_WORKTREE__REVIEWED_LOCAL_EXCLUSIONS'
    assert (identity['changed_path_count'],identity['prunable_worktree_count'],identity['unexpected_ignored_path_count'])==(0,0,0)
    assert identity['reviewed_ignored_path_count']==identity['ignored_path_count']==3
    assert identity['head_commit']==_git(repo,'rev-parse','HEAD')
    assert source_identity(repo)['worktree_fingerprint']==identity['worktree_fingerprint']


def test_source_identity_blocks_unreviewed_ignored_content(tmp_path):
    from scripts.collect_gx1_handover_readonly import source_identity
    repo=_identity_repo(tmp_path)
    (repo/'retired'/'__pycache__').mkdir(parents=True)
    (repo/'retired'/'__pycache__'/'gone.cpython-310.pyc').write_bytes(b'\0')
    (repo/'run.log').write_text('evidence\n')
    identity=source_identity(repo)
    assert identity['source_identity_gate']=='BLOCK_UNEXPECTED_IGNORED_CONTENT'
    assert identity['unexpected_ignored_paths']==['retired/','run.log']
    assert identity['unexpected_ignored_path_count']==2


def test_source_identity_binds_untracked_bytes_and_blocks_dirty_tree(tmp_path):
    from scripts.collect_gx1_handover_readonly import source_identity
    repo=_identity_repo(tmp_path)
    clean=source_identity(repo)['worktree_fingerprint']
    note=repo/'note.txt'
    note.write_text('a\n')
    first=source_identity(repo)
    note.write_text('b\n')
    second=source_identity(repo)
    assert first['source_identity_gate']==second['source_identity_gate']=='BLOCK_DIRTY_WORKTREE'
    assert len({clean,first['worktree_fingerprint'],second['worktree_fingerprint']})==3


def test_source_identity_rejects_symlinked_virtual_environment(tmp_path):
    from scripts.collect_gx1_handover_readonly import source_identity
    repo=_identity_repo(tmp_path)
    target=tmp_path/'other-venv'
    target.mkdir()
    (target/'pyvenv.cfg').write_text('home = /usr/bin\n')
    (repo/'.venv').symlink_to(target)
    with pytest.raises(ValueError,match='virtual environment'):
        source_identity(repo)


def test_source_identity_rejects_widened_exclusions(tmp_path):
    from scripts.collect_gx1_handover_readonly import LAUNCH_STATE_NAME, source_identity
    repo=_identity_repo(tmp_path)
    state=json.loads((repo/LAUNCH_STATE_NAME).read_text())
    state['reviewed_local_runtime_exclusions']['paths'].append('data/')
    (repo/LAUNCH_STATE_NAME).write_text(json.dumps(state))
    _git(repo,'commit','-qam','widen exclusions')
    with pytest.raises(ValueError,match='exclusions are invalid'):
        source_identity(repo)


@pytest.mark.parametrize('source_only,expected',[(False,0),(True,2)])
def test_handover_prints_identity_lines_and_source_only_blocks(monkeypatch,capsys,source_only,expected):
    import scripts.collect_gx1_handover_readonly as collector
    identity={'head_commit':'c'*40,'worktree_fingerprint':'f'*64,'changed_path_count':0,'ignored_path_count':1,
              'prunable_worktree_count':0,'reviewed_ignored_path_count':0,'unexpected_ignored_path_count':1,
              'source_identity_gate':'BLOCK_UNEXPECTED_IGNORED_CONTENT','unexpected_ignored_paths':['stale/']}
    monkeypatch.setattr(collector,'current_status',lambda *_a,**_k:{'decision':'OBSERVATION_ONLY_NOT_RUN_AUTHORITY'})
    monkeypatch.setattr(collector,'source_identity',lambda _repo:identity)
    monkeypatch.setattr('sys.argv',['collector',
                                    *(['--source-only'] if source_only else [])])
    assert collector.main()==expected
    lines=capsys.readouterr().out.splitlines()
    assert 'unexpected_ignored_path_count: 1' in lines and 'prunable_worktree_count: 0' in lines
    assert 'unexpected_ignored_paths: ["stale/"]' in lines
