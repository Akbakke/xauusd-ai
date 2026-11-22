"""Exact004 containment/source tests; synthetic metadata only, no CUDA/model."""
import ast
import hashlib
from pathlib import Path
import shlex
import subprocess

import pytest
from gx1.contracts import gx1_capped_execution_v1 as g

SHA = 'a' * 64
MODULE = 'gx1.audit.continue_native_future_state_bias_suffix_v1'
BASENAME = 'NATIVE_FUTURE_STATE_BIAS_SUFFIX_CONTINUE_20261122_004'
PARENT = '8ead4ba4e9c8fb4fcd35e6cd40818d6920fe16ed'
ENV = {
    'GX1_CAPPED_MEMORY_BYTES': '21474836480', 'GX1_CAPPED_SWAP_BYTES': '536870912',
    'GX1_CAPPED_TASKS_MAX': '128', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
    'OPENBLAS_NUM_THREADS': '8', 'NUMEXPR_NUM_THREADS': '8', 'VECLIB_MAXIMUM_THREADS': '8',
    'GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE': 'true', 'GX1_NATIVE_FUTURE_STATE_BIAS_CONTINUE_PLAN_SHA256': SHA,
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


def setup_env(monkeypatch):
    for name, value in ENV.items(): monkeypatch.setenv(name, value)


def test_exact004_containment_never_grants_training_or_optimizer_authority(monkeypatch):
    setup_env(monkeypatch)
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: {'canonical_scope': True})
    proof = g.require_native_future_state_bias_suffix_continuation_execution(plan_sha256=SHA)
    assert proof['canonical_scope'] and proof['kind'] == 'bias_suffix_continuation'
    assert proof['class'] == 'trainer' and proof['execution_mode'] == 'canonical'
    assert proof['training_authority_granted'] is proof['optimizer_authority_granted'] is False
    assert proof['outer_seconds'] == 3600 and proof['internal_seconds'] == 3570
    assert proof['cpu_affinity'] == list(range(19)) and proof['numerical_threads'] == 8
    assert proof['plan_path'].endswith('/configs/research/' + BASENAME + '.json')
    assert proof['output_root'] == '/home/andre2/GX1_RUNS/' + BASENAME
    assert proof['plan']['sha256'] == SHA


@pytest.mark.parametrize('name', sorted(ENV))
def test_every004_mode_scope_thread_hardware_field_rejects_substitution(monkeypatch, name):
    setup_env(monkeypatch); monkeypatch.setenv(name, ENV[name] + 'wrong')
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('invalid environment reached scope'))
    with pytest.raises(RuntimeError, match='SUFFIX_CONTINUE_ENV_INVALID'):
        g.require_native_future_state_bias_suffix_continuation_execution(plan_sha256=SHA)


@pytest.mark.parametrize('bad', ['', 'A' * 64, 'x' * 64, 'a' * 63, None, 1])
def test_bad004_plan_hash_fails_before_scope(monkeypatch, bad):
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda: pytest.fail('bad hash reached scope'))
    with pytest.raises(RuntimeError, match='SUFFIX_CONTINUE_PLAN_SHA256_INVALID'):
        g.require_native_future_state_bias_suffix_continuation_execution(plan_sha256=bad)


def test_actual_scope_failure_cannot_be_hidden(monkeypatch):
    setup_env(monkeypatch)
    def fail(): raise RuntimeError('real cgroup/lock proof failed')
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', fail)
    with pytest.raises(RuntimeError, match='real cgroup/lock proof failed'):
        g.require_native_future_state_bias_suffix_continuation_execution(plan_sha256=SHA)


def shell_harness(tmp_path, *, preflight=False, mutation=None, configure=False, override=None):
    repo = tmp_path / 'canonical';repo.mkdir();config = repo / 'configs/research';config.mkdir(parents=True)
    plan = config / (BASENAME + '.json');plan.write_text('{"synthetic004":true}\n')
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    argv = [str(repo / '.venv/bin/python'), '-m', MODULE, '--plan-sha256', digest]
    if preflight: argv.append('--preflight')
    if mutation is not None: argv = mutation(argv, plan)
    state, functions = (g._NATIVE_FUTURE_STATE_BIAS_CONTINUE_GUARD_SHELL_ADDITIONS[i][2] for i in (0,1))
    # Pure shell algorithm tests use a declared temporary output parent.
    functions = functions.replace('/home/andre2/GX1_RUNS/' + BASENAME, str(repo / 'output004'))
    script = state + functions + '\nREPO_ROOT=' + shlex.quote(str(repo))
    script += '\nCANONICAL_TRAINER_PYTHON=' + shlex.quote(str(repo / '.venv/bin/python'))
    if configure:
        values = {'JOB_CLASS':'trainer','ATTENDED_SMOKE':'false','CUDA_PRODUCER_GUARD':'false','MEM':'20G','SWAP':'512M',
                  'POWER_BENCHMARK_SCOPE_JSON':'','POWER_BENCHMARK_SCOPE_SHA256':''}
        values.update(override or {})
        script += ''.join('\n' + name + '=' + shlex.quote(value) for name,value in values.items())
        script += '\nconfigure_native_future_state_bias_suffix_continuation ' + ' '.join(map(shlex.quote,argv))
        script += '\nprintf "%s,%s,%s,%s,%s,%s,%s\\n" "$NATIVE_FUTURE_STATE_BIAS_CONTINUE" "$TRAINER_DEVICE" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_GPU_MAX_POWER_LIMIT_W" "$TRAINER_GPU_MAX_POWER_DRAW_W" "$TRAINER_GPU_MAX_CORE_TEMP_C"\n'
    else:
        script += '\nmatches_bound_native_future_state_bias_suffix_continuation ' + ' '.join(map(shlex.quote,argv)) + '\n'
    return subprocess.run(['bash','-c',script],cwd=repo,text=True,capture_output=True)


@pytest.mark.parametrize('preflight',[False,True])
def test_exact004_command_and_optional_preflight_match(tmp_path,preflight):
    assert shell_harness(tmp_path,preflight=preflight).returncode == 0


@pytest.mark.parametrize('case',['extra','wrong_python','wrong_module','wrong_sha','wrong_flag','missing_plan','symlink_plan','old003_digest'])
def test_004_cli_and_plan_substitutions_fail(tmp_path,case):
    def mutate(argv,plan):
        if case=='extra':argv+=['--unsafe']
        elif case=='wrong_python':argv[0]='/usr/bin/python3'
        elif case=='wrong_module':argv[2]='gx1.audit.diagnose_native_future_state_bias_suffix_v1'
        elif case=='wrong_sha':argv[4]='f'*64
        elif case=='wrong_flag':argv[3]='--plan'
        elif case=='missing_plan':plan.unlink()
        elif case=='symlink_plan':
            other=plan.with_suffix('.original');plan.rename(other);plan.symlink_to(other)
        else:argv[4]='7ee2dc2c3d2d4c99bc149dc66904e01fef2a874f112592eb0f2042ff38754215'
        return argv
    result=shell_harness(tmp_path,mutation=mutate)
    assert result.returncode!=0 and 'command not found' not in result.stderr


def test_004_shell_configuration_preserves_strict_resource_caps(tmp_path):
    result=shell_harness(tmp_path,configure=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='true,cuda,3600,3570,250,250,75'


@pytest.mark.parametrize('field,value',[('JOB_CLASS','audit'),('MEM','4G'),('SWAP','1G'),('ATTENDED_SMOKE','true'),('CUDA_PRODUCER_GUARD','true'),('POWER_BENCHMARK_SCOPE_JSON','/tmp/other'),('POWER_BENCHMARK_SCOPE_SHA256','f'*64)])
def test_004_shell_rejects_other_containment_and_power_scopes(tmp_path,field,value):
    result=shell_harness(tmp_path,configure=True,override={field:value})
    assert result.returncode==75
    assert 'exact canonical trainer containment' in result.stderr


def parent_guard_owners():
    repo=Path(g.__file__).resolve().parents[2]
    owners={f'gx1/synthetic003_owner_{i}.py':'b'*64 for i in range(72)}
    for name in ('gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh'):
        owners[name]=hashlib.sha256(subprocess.check_output(['git','show',PARENT+':'+name],cwd=repo)).hexdigest()
    return owners


def test_003_guard_authorities_and_all_old_shell_bytes_are_preserved():
    proof=g.native_future_state_bias_suffix_continuation_source_bridge(parent_guard_owners(),producer_commit=PARENT)
    assert proof['checked_guard_owners']==2 and proof['other_parent_owners_to_be_checked_by_caller']==72
    repo=Path(g.__file__).resolve().parents[2]
    for name,cell in proof['owners'].items():assert cell['current_sha256']==hashlib.sha256((repo/name).read_bytes()).hexdigest()
    text=(repo/'scripts/gx1_capped_run.sh').read_text()
    for anchor,position,fragment in reversed(g._NATIVE_FUTURE_STATE_BIAS_CONTINUE_GUARD_SHELL_ADDITIONS):
        composite=fragment+anchor if position=='before'else anchor+fragment
        assert text.count(fragment)==text.count(composite)==1;text=text.replace(composite,anchor,1)
    assert text.encode()==subprocess.check_output(['git','show',PARENT+':scripts/gx1_capped_run.sh'],cwd=repo)


@pytest.mark.parametrize('case',['producer','count','guard_sha','path','digest_type'])
def test_004_bridge_rejects_unbound_parent_inventory(case):
    owners=parent_guard_owners();producer=PARENT
    if case=='producer':producer='a'*40
    elif case=='count':owners.pop('gx1/synthetic003_owner_0.py')
    elif case=='guard_sha':owners['scripts/gx1_capped_run.sh']='f'*64
    elif case=='path':owners['../outside.py']=owners.pop('gx1/synthetic003_owner_0.py')
    else:owners['gx1/synthetic003_owner_0.py']=1
    with pytest.raises(RuntimeError):g.native_future_state_bias_suffix_continuation_source_bridge(owners,producer_commit=producer)


@pytest.mark.parametrize('mutation',['old_python','old_shell','missing_function','duplicate_constant','different_constant'])
def test_004_bridge_rejects_any_change_outside_new_exact_owners(monkeypatch,mutation):
    repo=Path(g.__file__).resolve().parents[2];actual=Path.read_bytes;target=repo/'gx1/contracts/gx1_capped_execution_v1.py'
    if mutation=='old_shell':
        target=repo/'scripts/gx1_capped_run.sh';changed=actual(target).replace(b'TRAINER_CPU_MAX_TEMP_C=80',b'TRAINER_CPU_MAX_TEMP_C=99')
    else:
        tree=ast.parse(actual(target))
        if mutation=='old_python':
            owner=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='require_guarded_cuda_trainer_execution');owner.body.append(ast.Expr(value=ast.Constant(value='unreviewed003owner')))
        elif mutation=='missing_function':tree.body=[n for n in tree.body if not(isinstance(n,ast.FunctionDef)and n.name=='require_native_future_state_bias_suffix_continuation_execution')]
        else:
            node=next(n for n in tree.body if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='_NATIVE_FUTURE_STATE_BIAS_CONTINUE_GUARD_SHELL_ADDITIONS')
            if mutation=='duplicate_constant':tree.body.append(node)
            else:node.value.elts[0].elts[0]=ast.Constant(value='wrong insertion anchor')
        changed=ast.unparse(tree).encode()
    monkeypatch.setattr(Path,'read_bytes',lambda path,*a,**kw:changed if path==target else actual(path,*a,**kw))
    with pytest.raises(RuntimeError):g.native_future_state_bias_suffix_continuation_source_bridge(parent_guard_owners(),producer_commit=PARENT)
