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


def test_all_tf_source_bridge_requires_exact_f33_old_guard_bytes(monkeypatch):
    use_historical_guard_snapshot(monkeypatch, '9d7b39da9214f74dcf6a7b8bf1a0fd3a719da29f')
    function = getattr(g, 'native_provider_hlc_all_tf_' + ALL_KIND + '_source_bridge')
    proof = function(ALL_GUARDS, producer_commit=ALL_PARENT)
    assert proof['checked_guard_owners'] == 2 and proof['old_guard_routes_preserved'] is True
    assert all(value['normalized_sha256'] == value['original_sha256'] for value in proof['owners'].values())


@pytest.mark.parametrize('case', ['producer', 'hash', 'extra'])
def test_all_tf_source_bridge_rejects_unbound_parent(monkeypatch, case):
    use_historical_guard_snapshot(monkeypatch, '9d7b39da9214f74dcf6a7b8bf1a0fd3a719da29f')
    owners = dict(ALL_GUARDS); parent = ALL_PARENT
    if case == 'producer': parent = 'f' * 40
    elif case == 'hash': owners['scripts/gx1_capped_run.sh'] = 'f' * 64
    else: owners['gx1/extra.py'] = 'f' * 64
    function = getattr(g, 'native_provider_hlc_all_tf_' + ALL_KIND + '_source_bridge')
    with pytest.raises(RuntimeError): function(owners, producer_commit=parent)


@pytest.mark.parametrize('name', ['gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh'])
def test_all_tf_source_bridge_rejects_unrelated_parent_edit(monkeypatch, name):
    use_historical_guard_snapshot(monkeypatch, '9d7b39da9214f74dcf6a7b8bf1a0fd3a719da29f')
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


# Only one fixed003 scoring lineage; no caller-controlled plan or output path.
SEGMENT_BASE = 'NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL_20261124_003'
SEGMENT_PARENT = '9d7b39da9214f74dcf6a7b8bf1a0fd3a719da29f'
SEGMENT_RECIPE = {'backbone_gradient_policy': None, 'batch_rows': 256, 'batches': 1385, 'drop_last': False, 'last_rows': 246, 'no_optimization': True, 'rows': 354550, 'seed': 20260911, 'shuffle': False}
SEGMENT_EXECUTION = {'GPU_core_stop_C': 75, 'GPU_memory_stop_C': 80, 'GPU_power_draw_stop_W': 250, 'GPU_power_limit_W': 250, 'class': 'trainer', 'cpu_affinity': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 'cuda_peak_reserved_bytes_max': 12884901888, 'device': 'cuda:0', 'host_CPU_package_stop_W': 125, 'host_CPU_stop_C': 80, 'host_min_available_bytes': 8589934592, 'host_min_commit_headroom_bytes': 8589934592, 'internal_seconds': 3570, 'kill_after_seconds': 10, 'memory': '20G', 'numerical_threads': 8, 'outer_seconds': 3600, 'swap': '512M', 'tasks_max': 128, 'watchdog_interval_seconds': 1}
SEGMENT_GUARDS = {
    'gx1/contracts/gx1_capped_execution_v1.py': '926fbdf60e1a4204b3f4c38a39440d519a3a62b6c2cc76e4b5d8b5bd2db8ca3f',
    'scripts/gx1_capped_run.sh': '0450ec5f0393807c618e7112cb4a67b0ecceba480649b465282543f4d9ca3b66',
}


def initial_segment_fixture(monkeypatch, tmp_path, *, preflight=False):
    import copy
    import json
    import sys
    old_path, _, _, _ = all_tf_python_fixture(monkeypatch, tmp_path)
    repo = old_path.parents[2]
    base_root = tmp_path / 'runs' / SEGMENT_BASE
    (base_root / 'PLANS').mkdir(parents=True)
    base_path = repo / 'configs/research' / (SEGMENT_BASE + '.json')
    base = {'schema_version': 'gx1_native_provider_hlc_all_tf_resumable_initial_base_plan_v1',
            'stage': SEGMENT_BASE.lower(), 'output_root': str(base_root), 'recipe': copy.deepcopy(SEGMENT_RECIPE),
            'execution': copy.deepcopy(SEGMENT_EXECUTION), 'model_contract': {'contract_sha256': ALL_MODEL_SHA}}
    base_path.write_text(json.dumps(base, sort_keys=True))
    base_ref = {'path': str(base_path), 'sha256': hashlib.sha256(base_path.read_bytes()).hexdigest()}
    segment = {'schema_version': 'gx1_native_provider_hlc_all_tf_initial_segment_plan_v1',
               'stage': SEGMENT_BASE.lower(), 'base_root': str(base_root), 'base_plan': base_ref,
               'recipe': copy.deepcopy(SEGMENT_RECIPE), 'execution': copy.deepcopy(SEGMENT_EXECUTION),
               'model_contract': {'contract_sha256': ALL_MODEL_SHA}}
    digest = initial_segment_write(base_root, segment, monkeypatch)
    if preflight: sys.argv += ['--preflight']
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: {'kernel_scope': True})
    return {'base_root': base_root, 'base_path': base_path, 'base': base, 'segment': segment,
            'digest': digest, 'plan_path': base_root / 'PLANS' / (digest + '.json'),
            'output_root': base_root / ('SEGMENT_' + digest)}


def initial_segment_write(base_root, segment, monkeypatch):
    import json
    import sys
    raw = json.dumps(segment, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    (base_root / 'PLANS' / (digest + '.json')).write_bytes(raw)
    monkeypatch.setattr(sys, 'argv', [sys.argv[0], '--all-tf-score-segment-plan-sha256', digest])
    for key, value in all_tf_environment_values(digest).items(): monkeypatch.setenv(key, value)
    return digest


@pytest.mark.parametrize('preflight', [False, True])
def test_initial_segment_exact_sha_derived_plan_root_and_none_profile(monkeypatch, tmp_path, preflight):
    fixture = initial_segment_fixture(monkeypatch, tmp_path, preflight=preflight)
    proof = g.require_native_provider_hlc_all_tf_initial_segment_execution(plan_sha256=fixture['digest'])
    assert proof['kernel_scope'] and proof['kind'] == 'provider_hlc_all_tf_initial_segment'
    assert proof['plan'] == {'path': str(fixture['plan_path']), 'sha256': fixture['digest']}
    assert proof['base_plan'] == fixture['segment']['base_plan']
    assert proof['output_root'] == str(fixture['output_root'])
    assert proof['read_only'] is True and proof['training_authority_granted'] is proof['optimizer_authority_granted'] is False
    assert (proof['outer_seconds'], proof['internal_seconds'], proof['kill_after_seconds']) == (3600, 3570, 10)


@pytest.mark.parametrize('case', ['own_plan_path', 'own_output_root', 'base_root', 'base_ref_path', 'base_ref_sha',
                                  'base_ref_extra', 'stage', 'schema', 'model_sha', 'model_policy', 'policy',
                                  'rows', 'batch', 'deadline', 'memory'])
def test_initial_segment_rejects_arbitrary_paths_wrong_lineage_policy_and_profile(monkeypatch, tmp_path, case):
    fixture = initial_segment_fixture(monkeypatch, tmp_path); segment = fixture['segment']
    if case == 'own_plan_path': segment['plan_path'] = '/tmp/arbitrary.json'
    elif case == 'own_output_root': segment['output_root'] = '/tmp/arbitrary-output'
    elif case == 'base_root': segment['base_root'] += '_other'
    elif case == 'base_ref_path': segment['base_plan']['path'] = '/tmp/arbitrary-base.json'
    elif case == 'base_ref_sha': segment['base_plan']['sha256'] = 'f' * 64
    elif case == 'base_ref_extra': segment['base_plan']['bytes'] = 1
    elif case == 'stage': segment['stage'] = 'arbitrary'
    elif case == 'schema': segment['schema_version'] = 'arbitrary'
    elif case == 'model_sha': segment['model_contract']['contract_sha256'] = 'f' * 64
    elif case == 'model_policy': segment['model_contract']['backbone_gradient_policy'] = None
    elif case == 'policy': segment['recipe']['backbone_gradient_policy'] = 'M5_primary'
    elif case == 'rows': segment['recipe']['rows'] -= 1
    elif case == 'batch': segment['recipe']['batch_rows'] = 64
    elif case == 'deadline': segment['execution']['outer_seconds'] = 7200
    elif case == 'memory': segment['execution']['memory'] = '4G'
    digest = initial_segment_write(fixture['base_root'], segment, monkeypatch)
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('bad segment reached kernel'))
    with pytest.raises(RuntimeError): g.require_native_provider_hlc_all_tf_initial_segment_execution(plan_sha256=digest)


@pytest.mark.parametrize('case', ['schema', 'stage', 'root', 'model', 'recipe', 'execution'])
def test_initial_segment_authenticates_bound_base_bytes_and_fields(monkeypatch, tmp_path, case):
    import json
    fixture = initial_segment_fixture(monkeypatch, tmp_path); base = fixture['base']
    if case == 'schema': base['schema_version'] = 'arbitrary'
    elif case == 'stage': base['stage'] = 'arbitrary'
    elif case == 'root': base['output_root'] += '_other'
    elif case == 'model': base['model_contract']['contract_sha256'] = 'f' * 64
    elif case == 'recipe': base['recipe']['backbone_gradient_policy'] = 'M5_primary'
    elif case == 'execution': base['execution']['GPU_power_limit_W'] = 300
    fixture['base_path'].write_text(json.dumps(base, sort_keys=True))
    fixture['segment']['base_plan']['sha256'] = hashlib.sha256(fixture['base_path'].read_bytes()).hexdigest()
    digest = initial_segment_write(fixture['base_root'], fixture['segment'], monkeypatch)
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('bad base reached kernel'))
    with pytest.raises(RuntimeError): g.require_native_provider_hlc_all_tf_initial_segment_execution(plan_sha256=digest)


@pytest.mark.parametrize('case', ['segment_symlink', 'base_symlink', 'plans_symlink', 'root_symlink',
                                  'existing_output', 'output_symlink', 'old_flag', 'extra', 'wrong_owner',
                                  'wrong_python', 'missing_sha_plan'])
def test_initial_segment_rejects_noncanonical_path_or_inexact_argv(monkeypatch, tmp_path, case):
    import sys
    fixture = initial_segment_fixture(monkeypatch, tmp_path)
    if case in ('segment_symlink', 'base_symlink'):
        path = fixture['plan_path'] if case == 'segment_symlink' else fixture['base_path']
        target = path.with_suffix('.original'); path.rename(target); path.symlink_to(target)
    elif case in ('plans_symlink', 'root_symlink'):
        path = fixture['base_root'] / 'PLANS' if case == 'plans_symlink' else fixture['base_root']
        target = path.with_name(path.name + '_real'); path.rename(target); path.symlink_to(target)
    elif case == 'existing_output': fixture['output_root'].mkdir()
    elif case == 'output_symlink': fixture['output_root'].symlink_to(tmp_path / 'absent')
    elif case == 'old_flag': sys.argv[1] = '--all-tf-plan-sha256'
    elif case == 'extra': sys.argv += ['--unsafe']
    elif case == 'wrong_owner': sys.argv[0] = str(tmp_path / 'arbitrary.py')
    elif case == 'wrong_python': monkeypatch.setattr(sys, 'executable', '/usr/bin/python3')
    elif case == 'missing_sha_plan': fixture['plan_path'].unlink()
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('bad route reached kernel'))
    with pytest.raises(RuntimeError): g.require_native_provider_hlc_all_tf_initial_segment_execution(plan_sha256=fixture['digest'])


@pytest.mark.parametrize('field', ['GX1_CAPPED_MEMORY_BYTES', 'GX1_CAPPED_TASKS_MAX', 'OMP_NUM_THREADS',
                                   'GX1_TRAINER_MODEL_MAX_WALL_SECONDS', 'GX1_TRAINER_GPU_MAX_POWER_DRAW_W',
                                   'GX1_TRAINER_GPU_MAX_CORE_TEMP_C', 'GX1_TRAINER_HOST_MIN_AVAILABLE_MEMORY_BYTES',
                                   'GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL', 'GX1_NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_PLAN_SHA256'])
def test_initial_segment_keeps_every_existing_containment_class(monkeypatch, tmp_path, field):
    fixture = initial_segment_fixture(monkeypatch, tmp_path)
    monkeypatch.setenv(field, all_tf_environment_values(fixture['digest'])[field] + 'wrong')
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('bad profile reached kernel'))
    with pytest.raises(RuntimeError, match='ENV_INVALID'):
        g.require_native_provider_hlc_all_tf_initial_segment_execution(plan_sha256=fixture['digest'])


def test_initial_segment_cannot_borrow_normal_initial_authority(monkeypatch, tmp_path):
    fixture = initial_segment_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('segment reached old authority'))
    with pytest.raises(RuntimeError):
        g.require_native_provider_hlc_all_tf_initial_execution(plan_sha256=fixture['digest'])


def initial_segment_shell_harness(tmp_path, *, case=None, configure=False):
    repo = tmp_path / 'repo'; repo.mkdir()
    base = tmp_path / 'runs' / SEGMENT_BASE; plans = base / 'PLANS'; plans.mkdir(parents=True)
    raw = b'{"fixture":true}\n'; digest = hashlib.sha256(raw).hexdigest()
    path = plans / (digest + '.json'); path.write_bytes(raw)
    output = base / ('SEGMENT_' + digest)
    argv = [str(repo / '.venv/bin/python'), '-m', MODULE, '--all-tf-score-segment-plan-sha256', digest]
    if case == 'preflight': argv += ['--preflight']
    elif case == 'extra': argv += ['--unsafe']
    elif case == 'wrong_module': argv[2] = 'gx1.audit.native_provider_hlc_replay_v1'
    elif case == 'wrong_python': argv[0] = '/usr/bin/python3'
    elif case == 'old_flag': argv[3] = '--all-tf-plan-sha256'
    elif case == 'wrong_sha': argv[4] = 'f' * 64
    elif case == 'missing_plan': path.unlink()
    elif case == 'plan_symlink':
        target = path.with_suffix('.original'); path.rename(target); path.symlink_to(target)
    elif case == 'plans_symlink':
        target = base / 'REAL_PLANS'; plans.rename(target); plans.symlink_to(target)
    elif case == 'root_symlink':
        target = base.with_name(base.name + '_real'); base.rename(target); base.symlink_to(target)
    elif case == 'existing_output': output.mkdir()
    elif case == 'output_symlink': output.symlink_to(tmp_path / 'absent')
    additions = g._NATIVE_PROVIDER_HLC_ALL_TF_INITIAL_SEGMENT_GUARD_SHELL_ADDITIONS
    state = g._NATIVE_PROVIDER_HLC_INITIAL_GUARD_SHELL_ADDITIONS[0][2]
    state += g._NATIVE_PROVIDER_HLC_REPLAY_GUARD_SHELL_ADDITIONS[0][2]
    state += g._NATIVE_PROVIDER_HLC_ALL_TF_GUARD_SHELL_ADDITIONS[0][2]
    functions = additions[0][2].replace('/home/andre2/GX1_RUNS/' + SEGMENT_BASE, str(base))
    script = state + functions + '\nREPO_ROOT=' + shlex.quote(str(repo))
    script += '\nCANONICAL_TRAINER_PYTHON=' + shlex.quote(str(repo / '.venv/bin/python'))
    if configure:
        values = {'JOB_CLASS': 'trainer', 'ATTENDED_SMOKE': 'false', 'CUDA_PRODUCER_GUARD': 'false',
                  'MEM': '20G', 'SWAP': '512M', 'POWER_BENCHMARK_SCOPE_JSON': '', 'POWER_BENCHMARK_SCOPE_SHA256': ''}
        if case == 'audit_class': values['JOB_CLASS'] = 'audit'
        elif case == 'wrong_memory': values['MEM'] = '4G'
        script += ''.join('\n' + key + '=' + shlex.quote(value) for key, value in values.items())
        script += '\nvalidate_target_command() {\n' + additions[1][2] + '\nreturn 88\n}\nvalidate_target_command '
        script += ' '.join(map(shlex.quote, argv))
        script += '\nprintf "%s,%s,%s,%s,%s,%s\\n" "$NATIVE_PROVIDER_HLC_ALL_TF_INITIAL" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_GPU_MAX_POWER_DRAW_W" "$TRAINER_OUT_BUNDLE_DIR" "$NATIVE_PROVIDER_HLC_INITIAL"\n'
    else:
        script += '\nmatches_bound_native_provider_hlc_all_tf_initial_segment ' + ' '.join(map(shlex.quote, argv)) + '\n'
    return subprocess.run(['bash', '-c', script], cwd=repo, text=True, capture_output=True), output


@pytest.mark.parametrize('case', [None, 'preflight'])
def test_initial_segment_shell_only_accepts_sha_named_fixed_lineage(tmp_path, case):
    result, _ = initial_segment_shell_harness(tmp_path, case=case)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('case', ['extra', 'wrong_module', 'wrong_python', 'old_flag', 'wrong_sha', 'missing_plan',
                                  'plan_symlink', 'plans_symlink', 'root_symlink'])
def test_initial_segment_shell_rejects_arbitrary_commands_paths_and_aliases(tmp_path, case):
    assert initial_segment_shell_harness(tmp_path, case=case)[0].returncode != 0


def test_initial_segment_configure_retains_profile_and_derives_output(tmp_path):
    result, output = initial_segment_shell_harness(tmp_path, configure=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'true,3600,3570,250,' + str(output) + ',false'


@pytest.mark.parametrize('case', ['existing_output', 'output_symlink', 'audit_class', 'wrong_memory'])
def test_initial_segment_configure_rejects_nonfresh_or_other_profile(tmp_path, case):
    assert initial_segment_shell_harness(tmp_path, case=case, configure=True)[0].returncode == 75


def test_initial_segment_source_bridge_restores_9d_then_full_f33():
    proof = g.native_provider_hlc_all_tf_initial_segment_source_bridge(SEGMENT_GUARDS, producer_commit=SEGMENT_PARENT)
    assert proof['checked_guard_owners'] == 2 and proof['old_guard_routes_preserved'] is proof['f33_guard_routes_preserved'] is True
    assert all(value['normalized_sha256'] == value['original_sha256'] for value in proof['owners'].values())


@pytest.mark.parametrize('name', ['gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh'])
def test_initial_segment_source_bridge_rejects_nonsegment_parent_edit(monkeypatch, name):
    path = Path(g.__file__).resolve().parents[2] / name
    original = Path.read_bytes
    def changed(candidate):
        value = original(candidate)
        return value + b'\n# unrelated parent mutation\n' if candidate == path else value
    monkeypatch.setattr(Path, 'read_bytes', changed)
    with pytest.raises(RuntimeError):
        g.native_provider_hlc_all_tf_initial_segment_source_bridge(SEGMENT_GUARDS, producer_commit=SEGMENT_PARENT)


def test_initial_segment_old_all_tf_source_bridge_rejects_new_generation():
    with pytest.raises(RuntimeError):
        g.native_provider_hlc_all_tf_initial_source_bridge(ALL_GUARDS, producer_commit=ALL_PARENT)
