"""Exact V2 INITIAL containment only; generated metadata, no GPU/model jobs."""
import hashlib,shlex,subprocess
from pathlib import Path
import pytest
from gx1.contracts import gx1_capped_execution_v1 as g
SHA='a'*64
MODULE='gx1.audit.native_future_state_provider_hlc_initial_v1'
BASENAME='NATIVE_PROVIDER_HLC_WHOLE_TRAIN_INITIAL_20261123_002'
PARENT='914e29a516d6762b59f6c8c21801e40af751e238'
GUARDS={'gx1/contracts/gx1_capped_execution_v1.py':'3a785511650592f990e644759f13d54da615313272bc1b6a4d6be89434b35563','scripts/gx1_capped_run.sh':'0c97154564fdc6cbb9759b989e825ed0776033ba357de22b01ff13bfdf93d1e4'}
ENV = {
    'GX1_CAPPED_MEMORY_BYTES': '21474836480', 'GX1_CAPPED_SWAP_BYTES': '536870912',
    'GX1_CAPPED_TASKS_MAX': '128', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
    'OPENBLAS_NUM_THREADS': '8', 'NUMEXPR_NUM_THREADS': '8', 'VECLIB_MAXIMUM_THREADS': '8',
    'GX1_NATIVE_PROVIDER_HLC_INITIAL': 'true', 'GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256': SHA,
    'GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE': 'false', 'GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256': '',
    'GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX': 'false', 'GX1_NATIVE_FUTURE_STATE_BIAS_SUFFIX_PLAN_SHA256': '',
    'GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN': 'false', 'GX1_NATIVE_FUTURE_STATE_BIAS_FROZEN_PLAN_SHA256': '',
    'GX1_NATIVE_FUTURE_STATE_THROUGHPUT': 'false', 'GX1_NATIVE_FUTURE_STATE_THROUGHPUT_PLAN_SHA256': '',
    'GX1_NATIVE_FUTURE_STATE_TRAINING': 'false', 'GX1_NATIVE_FUTURE_STATE_TRAINING_PLAN_SHA256': '',
    'GX1_NATIVE_FUTURE_STATE_TRAINING_LANE': '', 'GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION': 'false',
    'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS': 'false',
    'GX1_TRAINER_MAX_WALL_SECONDS': '3600', 'GX1_TRAINER_MODEL_MAX_WALL_SECONDS': '3570',
    'GX1_TRAINER_ATTENDED_STAGE_REQUIRED': 'false', 'GX1_TRAINER_GPU_INDEX': '0',
    'GX1_TRAINER_GPU_MAX_POWER_LIMIT_W': '250', 'GX1_TRAINER_GPU_MAX_POWER_DRAW_W': '250',
    'GX1_TRAINER_GPU_MAX_CORE_TEMP_C': '75', 'GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C': '80',
    'GX1_TRAINER_GPU_MAX_MEMORY_USED_MIB': '12288', 'GX1_TRAINER_GPU_MONITOR_INTERVAL_SECONDS': '1',
    'GX1_TRAINER_CPU_MAX_TEMP_C': '80', 'GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W': '125',
    'GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES': '8589934592',
    'GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES': '8589934592',
    'GX1_POWER_BENCHMARK_SCOPE_JSON': '', 'GX1_POWER_BENCHMARK_SCOPE_SHA256': '',
}

ENV.update({'GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE':'false','GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE_PLAN_SHA256':'',
 'GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256':'','GX1_NATIVE_FUTURE_STATE_CUDA_LANE':'',
 'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256':'','GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE':''})


def environment(monkeypatch):
    for name,value in ENV.items():monkeypatch.setenv(name,value)


def test_initial_resource_scope_has_no_training_or_optimizer_authority(monkeypatch):
    environment(monkeypatch);monkeypatch.setattr(g,'require_guarded_cuda_trainer_execution',lambda:{'actual_scope':True})
    p=g.require_native_provider_hlc_initial_execution(plan_sha256=SHA)
    assert p['actual_scope']and p['kind']=='provider_hlc_initial'and p['read_only']is True
    assert p['training_authority_granted']is p['optimizer_authority_granted']is False
    assert (p['outer_seconds'],p['internal_seconds'],p['kill_after_seconds'])==(3600,3570,10)
    assert p['cpu_affinity']==list(range(19))and p['numerical_threads']==8
    assert p['plan']['sha256']==SHA and p['output_root']=='/home/andre2/GX1_RUNS/'+BASENAME


@pytest.mark.parametrize('field',['GX1_NATIVE_PROVIDER_HLC_INITIAL','GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256','GX1_NATIVE_FUTURE_STATE_FULL_TRAIN_CAPTURE',
 'GX1_NATIVE_FUTURE_STATE_TRAINING','GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256','GX1_CAPPED_MEMORY_BYTES','OMP_NUM_THREADS',
 'GX1_TRAINER_MAX_WALL_SECONDS','GX1_TRAINER_MODEL_MAX_WALL_SECONDS','GX1_TRAINER_GPU_MAX_POWER_LIMIT_W','GX1_TRAINER_CPU_MAX_TEMP_C','GX1_TRAINER_HOST_MIN_COMMIT_HEADROOM_BYTES'])
def test_initial_rejects_other_modes_deadlines_resources_and_plan(monkeypatch,field):
    environment(monkeypatch);monkeypatch.setenv(field,ENV[field]+'wrong')
    monkeypatch.setattr(g,'require_guarded_cuda_trainer_execution',lambda:pytest.fail('bad env reached kernel scope'))
    with pytest.raises(RuntimeError,match='PROVIDER_HLC_INITIAL_ENV_INVALID'):g.require_native_provider_hlc_initial_execution(plan_sha256=SHA)


@pytest.mark.parametrize('bad',['A'*64,'f'*63,None])
def test_initial_bad_sha_rejected_before_scope(monkeypatch,bad):
    monkeypatch.setattr(g,'require_guarded_cuda_trainer_execution',lambda:pytest.fail('bad SHA reached scope'))
    with pytest.raises(RuntimeError,match='PLAN_SHA256_INVALID'):g.require_native_provider_hlc_initial_execution(plan_sha256=bad)


def harness(tmp_path,*,case=None,configure=False):
    repo=tmp_path/'repo';repo.mkdir();config=repo/'configs/research';config.mkdir(parents=True)
    plan=config/(BASENAME+'.json');plan.write_text('{"fixture":true}\n');digest=hashlib.sha256(plan.read_bytes()).hexdigest()
    argv=[str(repo/'.venv/bin/python'),'-m',MODULE,'--plan-sha256',digest]
    if case=='preflight':argv+=['--preflight']
    elif case=='extra':argv+=['--unsafe']
    elif case=='wrong_module':argv[2]='gx1.audit.native_future_state_training_v1'
    elif case=='wrong_python':argv[0]='/usr/bin/python3'
    elif case=='wrong_sha':argv[4]='f'*64
    elif case=='symlink':
        target=plan.with_suffix('.original');plan.rename(target);plan.symlink_to(target)
    if case=='existing_output':(repo/'initial-output').mkdir()
    state,functions=(g._NATIVE_PROVIDER_HLC_INITIAL_GUARD_SHELL_ADDITIONS[i][2]for i in (0,1))
    functions=functions.replace('/home/andre2/GX1_RUNS/'+BASENAME,str(repo/'initial-output'))
    script=state+functions+'\nREPO_ROOT='+shlex.quote(str(repo))+'\nCANONICAL_TRAINER_PYTHON='+shlex.quote(str(repo/'.venv/bin/python'))
    if configure:
        values={'JOB_CLASS':'trainer','ATTENDED_SMOKE':'false','CUDA_PRODUCER_GUARD':'false','MEM':'20G','SWAP':'512M','POWER_BENCHMARK_SCOPE_JSON':'','POWER_BENCHMARK_SCOPE_SHA256':''}
        if case=='audit_class':values['JOB_CLASS']='audit'
        script+=''.join('\n'+k+'='+shlex.quote(v)for k,v in values.items())
        script+='\nconfigure_native_provider_hlc_initial '+' '.join(map(shlex.quote,argv))
        script+='\nprintf "%s,%s,%s,%s,%s,%s,%s\n" "$NATIVE_PROVIDER_HLC_INITIAL" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_GPU_MAX_POWER_DRAW_W" "$TRAINER_CPU_MAX_TEMP_C" "$TRAINER_CPU_MAX_PACKAGE_POWER_W" "$TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES"\n'
    else:script+='\nmatches_bound_native_provider_hlc_initial '+' '.join(map(shlex.quote,argv))+'\n'
    return subprocess.run(['bash','-c',script],cwd=repo,text=True,capture_output=True)


@pytest.mark.parametrize('case',[None,'preflight'])
def test_exact_initial_shell_cli_and_optional_preflight(tmp_path,case):assert harness(tmp_path,case=case).returncode==0


@pytest.mark.parametrize('case',['extra','wrong_module','wrong_python','wrong_sha','symlink'])
def test_initial_other_commands_cannot_enter_route(tmp_path,case):assert harness(tmp_path,case=case).returncode!=0


def test_initial_configure_exact_safety_and_fresh_root(tmp_path):
    a=tmp_path/'a';a.mkdir();r=harness(a,configure=True)
    assert r.returncode==0,r.stderr
    assert r.stdout.strip()=='true,3600,3570,250,80,125,8589934592'
    b=tmp_path/'b';b.mkdir();assert harness(b,configure=True,case='existing_output').returncode==75
    c=tmp_path/'c';c.mkdir();assert harness(c,configure=True,case='audit_class').returncode==75


@pytest.mark.parametrize('field',[None,'GX1_NATIVE_FUTURE_STATE_TRAINING','GX1_TRAINER_MODEL_MAX_WALL_SECONDS','GX1_TRAINER_CPU_MAX_PACKAGE_POWER_W'])
def test_nested_initial_requires_same_mode_deadline_and_host_caps(field):
    values=dict(ENV)
    if field:values[field]+='wrong'
    script='NATIVE_PROVIDER_HLC_INITIAL=true\nNATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256='+SHA+'\n'
    script+=''.join('export '+k+'='+shlex.quote(v)+'\n'for k,v in values.items())
    script+=g._NATIVE_PROVIDER_HLC_INITIAL_GUARD_SHELL_ADDITIONS[3][2]
    result=subprocess.run(['bash','-c',script],text=True,capture_output=True)
    assert result.returncode==(75 if field else 0),result.stderr


def test_new_source_bridge_preserves_all_old_guard_code_and_shell_bytes(monkeypatch):
    use_historical_guard_snapshot(monkeypatch, '27fc84b9111d198ad0a6f6570d7e8efa7d0f1d49')
    proof=g.native_provider_hlc_initial_source_bridge(GUARDS,producer_commit=PARENT)
    assert proof['checked_guard_owners']==2 and proof['old_guard_routes_preserved']is True


@pytest.mark.parametrize('bad',['producer','hash','extra'])
def test_new_source_bridge_rejects_unbound_anchor(monkeypatch, bad):
    use_historical_guard_snapshot(monkeypatch, '27fc84b9111d198ad0a6f6570d7e8efa7d0f1d49')
    owners=dict(GUARDS);producer=PARENT
    if bad=='producer':producer='a'*40
    elif bad=='hash':owners['scripts/gx1_capped_run.sh']='f'*64
    else:owners['gx1/extra.py']='f'*64
    with pytest.raises(RuntimeError):g.native_provider_hlc_initial_source_bridge(owners,producer_commit=producer)


@pytest.mark.parametrize('name',['gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh'])
def test_new_source_bridge_rejects_any_unrelated_source_delta(monkeypatch,name):
    use_historical_guard_snapshot(monkeypatch, '27fc84b9111d198ad0a6f6570d7e8efa7d0f1d49')
    repo=Path(g.__file__).resolve().parents[2];path=repo/name;original=Path.read_bytes
    def changed(p):
        raw=original(p)
        if p==path:raw+=b'\nUNEXPECTED_OLD_OWNER_DELTA = 1\n'if name.endswith('.py')else b'\n# old shell delta\n'
        return raw
    monkeypatch.setattr(Path,'read_bytes',changed)
    with pytest.raises(RuntimeError):g.native_provider_hlc_initial_source_bridge(GUARDS,producer_commit=PARENT)


# Exact ALL-TF routes reuse containment values, never consumed route authority.
ALL_KIND = 'initial'
ALL_BASENAME = 'NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_002'
ALL_OWNER = 'native_future_state_provider_hlc_initial_v1.py'
ALL_PARENT = 'f33da18dd6a0e3d59e4876fa26f8e62ec5ab97b8'
ALL_GUARDS = {
    'gx1/contracts/gx1_capped_execution_v1.py': 'bbe6d6fe74ae289365759718bef1b045550c8250df08f937d87716b93a6d2c90',
    'scripts/gx1_capped_run.sh': 'ca334f3e843bd0d227883b783421c6bbc475a808b3b9a3d8c4ef6a0b622c5cc1',
}
ALL_MODEL_SHA = '6d6ac22819e54c7b12aebcf4e1afe27b990d8a4d90d16500e352c7353d99cca0'


def all_tf_environment_values(digest, lane='warm'):
    values = dict(ENV)
    values.update({
        'GX1_NATIVE_PROVIDER_HLC_INITIAL': 'false',
        'GX1_NATIVE_PROVIDER_HLC_INITIAL_PLAN_SHA256': '',
        'GX1_NATIVE_PROVIDER_HLC_REPLAY': 'false',
        'GX1_NATIVE_PROVIDER_HLC_REPLAY_PLAN_SHA256': '',
        'GX1_NATIVE_PROVIDER_HLC_REPLAY_LANE': '',
        'GX1_NATIVE_PROVIDER_HLC_LEARNING': 'false',
        'GX1_NATIVE_PROVIDER_HLC_LEARNING_PLAN_SHA256': '',
        'GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME': 'false',
        'GX1_NATIVE_PROVIDER_HLC_LEARNING_RESUME_PLAN_SHA256': '',
        'GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL': 'true' if ALL_KIND == 'initial' else 'false',
        'GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256': digest if ALL_KIND == 'initial' else '',
        'GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY': 'true' if ALL_KIND == 'replay' else 'false',
        'GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_PLAN_SHA256': digest if ALL_KIND == 'replay' else '',
        'GX1_NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE': lane if ALL_KIND == 'replay' else '',
    })
    return values


def all_tf_python_fixture(monkeypatch, tmp_path, *, lane='warm', preflight=False):
    import json
    import sys
    repo = tmp_path / 'repo'
    (repo / 'configs/research').mkdir(parents=True)
    (repo / 'gx1/contracts').mkdir(parents=True)
    (repo / 'gx1/audit').mkdir(parents=True)
    runs = tmp_path / 'runs'; runs.mkdir()
    owner = repo / 'gx1/audit' / ALL_OWNER; owner.write_text('# fixture owner\n')
    root = runs / (ALL_BASENAME + ('_' + lane if ALL_KIND == 'replay' else ''))
    plan = {'stage': ALL_BASENAME.lower(), 'recipe': {'backbone_gradient_policy': None},
            'model_contract': {'contract_sha256': ALL_MODEL_SHA}}
    if ALL_KIND == 'initial':
        plan['output_root'] = str(root)
    else:
        plan['output_roots'] = {name: str(runs / (ALL_BASENAME + '_' + name)) for name in ('warm', 'cold')}
    path = repo / 'configs/research' / (ALL_BASENAME + '.json')
    path.write_text(json.dumps(plan))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    real_path = Path
    path_adapter = lambda value: runs if str(value) == '/home/andre2/GX1_RUNS' else real_path(value)
    path_adapter.cwd = real_path.cwd
    monkeypatch.setattr(g, 'Path', path_adapter)
    monkeypatch.setattr(g, '__file__', str(repo / 'gx1/contracts/gx1_capped_execution_v1.py'))
    monkeypatch.chdir(repo)
    argv = [str(owner), '--all-tf-plan-sha256', digest]
    if ALL_KIND == 'replay': argv += ['--lane', lane]
    if preflight: argv += ['--preflight']
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(sys, 'executable', str(repo / '.venv/bin/python'))
    for key, value in all_tf_environment_values(digest, lane).items(): monkeypatch.setenv(key, value)
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: {'kernel_scope': True})
    return path, root, digest, plan


def all_tf_require(digest, lane='warm'):
    function = getattr(g, 'require_native_provider_hlc_all_tf_' + ALL_KIND + '_execution')
    return function(plan_sha256=digest, **({'lane': lane} if ALL_KIND == 'replay' else {}))


@pytest.mark.parametrize('preflight', [False, True])
@pytest.mark.parametrize('lane', ['warm', 'cold'] if ALL_KIND == 'replay' else ['warm'])
def test_all_tf_exact_python_route_retains_containment_and_separate_root(monkeypatch, tmp_path, preflight, lane):
    path, root, digest, _ = all_tf_python_fixture(monkeypatch, tmp_path, preflight=preflight, lane=lane)
    proof = all_tf_require(digest, lane)
    assert proof['kernel_scope'] and proof['kind'] == 'provider_hlc_all_tf_' + ALL_KIND
    assert proof['plan'] == {'path': str(path), 'sha256': digest}
    assert proof['output_root'] == str(root)
    assert proof['training_authority_granted'] is proof['optimizer_authority_granted'] is False
    assert proof['outer_seconds'] == (3600 if ALL_KIND == 'initial' else 1800)
    assert proof['internal_seconds'] == (3570 if ALL_KIND == 'initial' else 1770)
    assert proof['cpu_affinity'] == list(range(19)) and proof['numerical_threads'] == 8


@pytest.mark.parametrize('case', ['policy', 'missing_policy', 'contract_sha', 'contract_policy', 'stage', 'root',
                                  'extra_argv', 'old_flag', 'wrong_owner', 'wrong_python', 'symlink_plan',
                                  'existing_root', 'symlink_root', 'symlink_parent'])
def test_all_tf_python_binding_rejects_wrong_policy_route_plan_or_root(monkeypatch, tmp_path, case):
    import json
    import sys
    path, root, digest, plan = all_tf_python_fixture(monkeypatch, tmp_path)
    if case == 'policy': plan['recipe']['backbone_gradient_policy'] = 'M5_primary'
    elif case == 'missing_policy': del plan['recipe']['backbone_gradient_policy']
    elif case == 'contract_sha': plan['model_contract']['contract_sha256'] = 'f' * 64
    elif case == 'contract_policy': plan['model_contract']['backbone_gradient_policy'] = None
    elif case == 'stage': plan['stage'] = 'consumed_reference'
    elif case == 'root':
        if ALL_KIND == 'initial': plan['output_root'] = str(root) + '_other'
        else: plan['output_roots']['cold'] += '_other'
    elif case == 'extra_argv': sys.argv += ['--unsafe']
    elif case == 'old_flag': sys.argv[1] = '--plan-sha256'
    elif case == 'wrong_owner': sys.argv[0] = str(path.parent / 'wrong.py')
    elif case == 'wrong_python': monkeypatch.setattr(sys, 'executable', '/usr/bin/python3')
    elif case == 'symlink_plan':
        target = path.with_suffix('.original'); path.rename(target); path.symlink_to(target)
    elif case == 'existing_root': root.mkdir()
    elif case == 'symlink_root': root.symlink_to(root.parent / 'absent')
    elif case == 'symlink_parent':
        parent = root.parent; target = parent.with_name('real-runs'); parent.rename(target); parent.symlink_to(target)
    if case in ('policy', 'missing_policy', 'contract_sha', 'contract_policy', 'stage', 'root'):
        path.write_text(json.dumps(plan)); digest = hashlib.sha256(path.read_bytes()).hexdigest(); sys.argv[2] = digest
        for key, value in all_tf_environment_values(digest).items(): monkeypatch.setenv(key, value)
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('invalid binding reached kernel proof'))
    with pytest.raises(RuntimeError): all_tf_require(digest)


@pytest.mark.parametrize('field', [
    'GX1_CAPPED_MEMORY_BYTES', 'GX1_CAPPED_SWAP_BYTES', 'GX1_CAPPED_TASKS_MAX', 'OMP_NUM_THREADS',
    'GX1_TRAINER_MAX_WALL_SECONDS', 'GX1_TRAINER_MODEL_MAX_WALL_SECONDS',
    'GX1_TRAINER_GPU_MAX_POWER_DRAW_W', 'GX1_TRAINER_GPU_MAX_CORE_TEMP_C',
    'GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C', 'GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES',
    'GX1_NATIVE_PROVIDER_HLC_INITIAL', 'GX1_NATIVE_PROVIDER_HLC_REPLAY',
])
def test_all_tf_python_rejects_changed_profile_and_cross_mode_before_kernel(monkeypatch, tmp_path, field):
    _, _, digest, _ = all_tf_python_fixture(monkeypatch, tmp_path)
    monkeypatch.setenv(field, all_tf_environment_values(digest)[field] + 'wrong')
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('invalid mode reached kernel proof'))
    with pytest.raises(RuntimeError, match='ENV_INVALID'): all_tf_require(digest)


def test_all_tf_mode_cannot_borrow_consumed_python_guard(monkeypatch, tmp_path):
    _, _, digest, _ = all_tf_python_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('ALL-TF reached old kernel proof'))
    function = getattr(g, 'require_native_provider_hlc_' + ALL_KIND + '_execution')
    with pytest.raises(RuntimeError, match='ENV_INVALID'):
        function(plan_sha256=digest, **({'lane': 'warm'} if ALL_KIND == 'replay' else {}))


def all_tf_shell_harness(tmp_path, *, case=None, configure=False, lane='warm', dispatch=False):
    repo = tmp_path / 'repo'; (repo / 'configs/research').mkdir(parents=True)
    path = repo / 'configs/research' / (ALL_BASENAME + '.json'); path.write_text('{"fixture":true}\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    argv = [str(repo / '.venv/bin/python'), '-m', MODULE, '--all-tf-plan-sha256', digest]
    if ALL_KIND == 'replay': argv += ['--lane', lane]
    if case == 'preflight': argv += ['--preflight']
    elif case == 'extra': argv += ['--unsafe']
    elif case == 'wrong_module': argv[2] = 'gx1.audit.native_future_state_training_v1'
    elif case == 'wrong_python': argv[0] = '/usr/bin/python3'
    elif case == 'wrong_sha': argv[4] = 'f' * 64
    elif case == 'old_flag': argv[3] = '--plan-sha256'
    elif case == 'reordered': argv[3:5] = list(reversed(argv[3:5]))
    elif case == 'wrong_lane': argv[6] = 'main'
    elif case == 'missing_lane': argv = argv[:5]
    elif case == 'symlink':
        target = path.with_suffix('.original'); path.rename(target); path.symlink_to(target)
    elif case == 'missing_plan': path.unlink()
    output = repo / ('all-output' + ('_' + lane if ALL_KIND == 'replay' else ''))
    if case == 'existing_output': output.mkdir()
    elif case == 'output_symlink': output.symlink_to(repo / 'absent')
    additions = g._NATIVE_PROVIDER_HLC_ALL_TF_GUARD_SHELL_ADDITIONS
    old_state = g._NATIVE_PROVIDER_HLC_INITIAL_GUARD_SHELL_ADDITIONS[0][2] + g._NATIVE_PROVIDER_HLC_REPLAY_GUARD_SHELL_ADDITIONS[0][2]
    functions = additions[1][2].replace('/home/andre2/GX1_RUNS/' + ALL_BASENAME, str(repo / 'all-output'))
    script = old_state + additions[0][2] + functions + '\nREPO_ROOT=' + shlex.quote(str(repo))
    script += '\nCANONICAL_TRAINER_PYTHON=' + shlex.quote(str(repo / '.venv/bin/python'))
    if configure or dispatch:
        values = {'JOB_CLASS': 'trainer', 'ATTENDED_SMOKE': 'false', 'CUDA_PRODUCER_GUARD': 'false',
                  'MEM': '20G', 'SWAP': '512M', 'POWER_BENCHMARK_SCOPE_JSON': '', 'POWER_BENCHMARK_SCOPE_SHA256': ''}
        if case == 'audit_class': values['JOB_CLASS'] = 'audit'
        elif case == 'wrong_memory': values['MEM'] = '4G'
        elif case == 'attended': values['ATTENDED_SMOKE'] = 'true'
        script += ''.join('\n' + key + '=' + shlex.quote(value) for key, value in values.items())
        if dispatch:
            script += '\nvalidate_target_command() {\n' + additions[2][2] + '\nreturn 88\n}\nvalidate_target_command '
        else:
            script += '\nconfigure_native_provider_hlc_all_tf_' + ALL_KIND + ' '
        script += ' '.join(map(shlex.quote, argv))
        script += '\nprintf "%s,%s,%s,%s,%s,%s,%s\\n" "$NATIVE_PROVIDER_HLC_ALL_TF_' + ALL_KIND.upper() + '" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_GPU_MAX_POWER_DRAW_W" "$NATIVE_PROVIDER_HLC_INITIAL" "$NATIVE_PROVIDER_HLC_REPLAY" "$TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES"\n'
    else:
        script += '\nmatches_bound_native_provider_hlc_all_tf_' + ALL_KIND + ' ' + ' '.join(map(shlex.quote, argv)) + '\n'
    return subprocess.run(['bash', '-c', script], cwd=repo, text=True, capture_output=True)


@pytest.mark.parametrize('case', [None, 'preflight'])
@pytest.mark.parametrize('lane', ['warm', 'cold'] if ALL_KIND == 'replay' else ['warm'])
def test_all_tf_shell_accepts_only_exact_new_flag_and_plan(tmp_path, case, lane):
    result = all_tf_shell_harness(tmp_path, case=case, lane=lane)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('case', ['extra', 'wrong_module', 'wrong_python', 'wrong_sha', 'old_flag',
                                  'reordered', 'symlink', 'missing_plan'] + (['wrong_lane', 'missing_lane'] if ALL_KIND == 'replay' else []))
def test_all_tf_shell_rejects_wrong_command_and_plan(tmp_path, case):
    assert all_tf_shell_harness(tmp_path, case=case).returncode != 0


@pytest.mark.parametrize('dispatch', [False, True])
@pytest.mark.parametrize('lane', ['warm', 'cold'] if ALL_KIND == 'replay' else ['warm'])
def test_all_tf_configure_uses_same_limits_and_keeps_consumed_mode_bits_false(tmp_path, dispatch, lane):
    result = all_tf_shell_harness(tmp_path, configure=True, dispatch=dispatch, lane=lane)
    assert result.returncode == 0, result.stderr
    deadline = '3600,3570' if ALL_KIND == 'initial' else '1800,1770'
    assert result.stdout.strip() == 'true,' + deadline + ',250,false,false,8589934592'


@pytest.mark.parametrize('case', ['existing_output', 'output_symlink', 'audit_class', 'wrong_memory', 'attended'])
def test_all_tf_configure_rejects_other_resources_and_nonfresh_root(tmp_path, case):
    assert all_tf_shell_harness(tmp_path, configure=True, case=case).returncode == 75


@pytest.mark.parametrize('field', [None, 'GX1_NATIVE_PROVIDER_HLC_INITIAL', 'GX1_NATIVE_PROVIDER_HLC_REPLAY',
                                   'GX1_TRAINER_MODEL_MAX_WALL_SECONDS', 'GX1_TRAINER_GPU_MAX_POWER_DRAW_W'])
def test_all_tf_nested_mode_hash_and_profile_must_match_parent(field):
    values = all_tf_environment_values(SHA)
    if field: values[field] += 'wrong'
    state = g._NATIVE_PROVIDER_HLC_ALL_TF_GUARD_SHELL_ADDITIONS[0][2]
    script = state + '\nNATIVE_PROVIDER_HLC_ALL_TF_' + ALL_KIND.upper() + '=true\n'
    script += 'NATIVE_PROVIDER_HLC_ALL_TF_' + ALL_KIND.upper() + '_PLAN_SHA256=' + SHA + '\n'
    if ALL_KIND == 'replay': script += 'NATIVE_PROVIDER_HLC_ALL_TF_REPLAY_LANE=warm\n'
    script += ''.join('export ' + key + '=' + shlex.quote(value) + '\n' for key, value in values.items())
    script += g._NATIVE_PROVIDER_HLC_ALL_TF_GUARD_SHELL_ADDITIONS[3][2]
    result = subprocess.run(['bash', '-c', script], text=True, capture_output=True)
    assert result.returncode == (75 if field else 0), result.stderr


def test_all_tf_source_bridge_requires_exact_f33_old_guard_bytes():
    function = getattr(g, 'native_provider_hlc_all_tf_' + ALL_KIND + '_source_bridge')
    proof = function(ALL_GUARDS, producer_commit=ALL_PARENT)
    assert proof['checked_guard_owners'] == 2 and proof['old_guard_routes_preserved'] is True
    assert all(value['normalized_sha256'] == value['original_sha256'] for value in proof['owners'].values())


@pytest.mark.parametrize('case', ['producer', 'hash', 'extra'])
def test_all_tf_source_bridge_rejects_unbound_parent(case):
    owners = dict(ALL_GUARDS); parent = ALL_PARENT
    if case == 'producer': parent = 'f' * 40
    elif case == 'hash': owners['scripts/gx1_capped_run.sh'] = 'f' * 64
    else: owners['gx1/extra.py'] = 'f' * 64
    function = getattr(g, 'native_provider_hlc_all_tf_' + ALL_KIND + '_source_bridge')
    with pytest.raises(RuntimeError): function(owners, producer_commit=parent)


@pytest.mark.parametrize('name', ['gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh'])
def test_all_tf_source_bridge_rejects_unrelated_parent_edit(monkeypatch, name):
    path = Path(g.__file__).resolve().parents[2] / name
    before = Path.read_bytes
    def changed(candidate):
        value = before(candidate)
        return value + b'\n# unrelated source mutation\n' if candidate == path else value
    monkeypatch.setattr(Path, 'read_bytes', changed)
    function = getattr(g, 'native_provider_hlc_all_tf_' + ALL_KIND + '_source_bridge')
    with pytest.raises(RuntimeError): function(ALL_GUARDS, producer_commit=ALL_PARENT)



def use_historical_guard_snapshot(monkeypatch, commit):
    """Qualify an old one-use bridge on its actual post-route generation."""
    repo = Path(g.__file__).resolve().parents[2]
    snapshots = {repo / name: subprocess.check_output(['git', 'show', commit + ':' + name], cwd=repo)
                 for name in ('gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh')}
    original = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes', lambda path: snapshots[path] if path in snapshots else original(path))


def test_all_tf_consumed_old_bridge_rejects_current_guard_generation():
    function = getattr(g, 'native_provider_hlc_' + ALL_KIND + '_source_bridge')
    with pytest.raises(RuntimeError):
        function(GUARDS, producer_commit=PARENT)


@pytest.mark.parametrize('bad', [None, True, 'A' * 64, 'f' * 63])
def test_all_tf_invalid_sha_stops_before_scope(monkeypatch, bad):
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('bad SHA reached kernel'))
    with pytest.raises(RuntimeError):
        all_tf_require(bad)
