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


def test_new_source_bridge_preserves_all_old_guard_code_and_shell_bytes():
    proof=g.native_provider_hlc_initial_source_bridge(GUARDS,producer_commit=PARENT)
    assert proof['checked_guard_owners']==2 and proof['old_guard_routes_preserved']is True


@pytest.mark.parametrize('bad',['producer','hash','extra'])
def test_new_source_bridge_rejects_unbound_anchor(bad):
    owners=dict(GUARDS);producer=PARENT
    if bad=='producer':producer='a'*40
    elif bad=='hash':owners['scripts/gx1_capped_run.sh']='f'*64
    else:owners['gx1/extra.py']='f'*64
    with pytest.raises(RuntimeError):g.native_provider_hlc_initial_source_bridge(owners,producer_commit=producer)


@pytest.mark.parametrize('name',['gx1/contracts/gx1_capped_execution_v1.py','scripts/gx1_capped_run.sh'])
def test_new_source_bridge_rejects_any_unrelated_source_delta(monkeypatch,name):
    repo=Path(g.__file__).resolve().parents[2];path=repo/name;original=Path.read_bytes
    def changed(p):
        raw=original(p)
        if p==path:raw+=b'\nUNEXPECTED_OLD_OWNER_DELTA = 1\n'if name.endswith('.py')else b'\n# old shell delta\n'
        return raw
    monkeypatch.setattr(Path,'read_bytes',changed)
    with pytest.raises(RuntimeError):g.native_provider_hlc_initial_source_bridge(GUARDS,producer_commit=PARENT)
