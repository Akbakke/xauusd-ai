"""Exact YEAR and FIVE_YEAR routes only; shell extraction never launches guards, data or CUDA."""
from __future__ import annotations
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
import pytest

REPO=Path(__file__).resolve().parents[1]
RUNNER=REPO/'scripts/gx1_capped_run.sh'
BASENAMES=('NATIVE_M5_TA_ATR03_YEAR_PREPARE_20261120_001',
    'NATIVE_M5_TA_ATR03_YEAR_MECHANICS_20261120_001',
    'NATIVE_M5_TA_ATR03_YEAR_FIT_20261120_001',
    'NATIVE_M5_TA_ATR03_FIVE_YEAR_PREPARE_20261120_001',
    'NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_001',
    'NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_001',
    'NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_003',
    'NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_002',
    'NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_002',
    'NATIVE_M5_TA_ATR03_THROUGHPUT_20261120_004',
    'NATIVE_M5_TA_ATR03_FIVE_YEAR_MECHANICS_20261120_003',
    'NATIVE_M5_TA_ATR03_FIVE_YEAR_FIT_20261120_003',
    'NATIVE_M5_TA_ATR03_FIVE_YEAR_WINDOW_20261120_001',
    'NATIVE_M5_TA_ATR03_FIVE_YEAR_CONTINUE_20261120_001',
    'NATIVE_M5_TA_ATR03_FIVE_YEAR_WINDOW_20261120_002',
    'NATIVE_M5_TA_ATR03_FIVE_YEAR_CONTINUE_20261120_002')


def _fixture(tmp_path,basename):
    folder=tmp_path/'configs/research';folder.mkdir(parents=True)
    path=folder/(basename+'.json')
    if '_WINDOW_'in basename or '_CONTINUE_'in basename:
        seconds=10800 if basename.endswith('_002')else 1800
        path.write_text(json.dumps({'schema_version':'gx1_native_m5_main_checkpoint_window_plan_v1',
            'execution':{'outer_seconds':seconds}})+'\n')
    else:path.write_text('{"test_only":true}\n')
    python=str(tmp_path/'.venv/bin/python')
    args=[python,'-m','gx1.audit.qualify_native_m5_full_gc_direction_v1','--plan',str(path),
        '--plan-sha256',hashlib.sha256(path.read_bytes()).hexdigest(),'--device','cuda',
        '--out-dir','/home/andre2/GX1_RUNS/'+basename]
    signature='matches_bound_native_m5_full_gc_fit() {'
    source=RUNNER.read_text();function=signature+source.split(signature,1)[1].split('\n}\n',1)[0]+'\n}\n'
    prefix='\n'.join(['set -eu','REPO_ROOT='+shlex.quote(str(tmp_path)),
        'CANONICAL_TRAINER_PYTHON='+shlex.quote(python),
        'NATIVE_M5_FULL_GC_FIT_TRAINER_MODULE='+shlex.quote(args[2]),function])
    return path,args,prefix,source


@pytest.mark.parametrize('basename',BASENAMES)
@pytest.mark.parametrize('variant',('valid','preflight','wrong_device','cross_output',
    'wrong_sha','unregistered_plan','extra_flag','symlink_plan','noncanonical_python','wrong_module'))
def test_native_year_route_is_literal_and_sha_bound(tmp_path,basename,variant):
    path,args,prefix,_=_fixture(tmp_path,basename)
    if variant=='preflight':args.append('--preflight')
    elif variant=='wrong_device':args[8]='cpu'
    elif variant=='cross_output':args[10]='/home/andre2/GX1_RUNS/'+BASENAMES[(BASENAMES.index(basename)+1)%len(BASENAMES)]
    elif variant=='wrong_sha':args[6]='0'*64
    elif variant=='unregistered_plan':args[4]=str(path.with_name(path.stem[:-3]+'099.json'))
    elif variant=='extra_flag':args.append('--cold-resume')
    elif variant=='symlink_plan':
        actual=path.with_suffix('.original');path.rename(actual);path.symlink_to(actual)
    elif variant=='noncanonical_python':args[0]='/usr/bin/python3'
    elif variant=='wrong_module':args[2]='gx1.scripts.research_ta_campaign_v1'
    result=subprocess.run(['bash','-c',prefix+'\nmatches_bound_native_m5_full_gc_fit '+shlex.join(args)],
        capture_output=True,text=True)
    assert (result.returncode==0)==(variant in('valid','preflight')),result.stderr


@pytest.mark.parametrize('basename,seconds',zip(BASENAMES,(1800,1800,3600,5400,1800,10800,1800,1800,10800,1800,1800,10800,1800,1800,10800,10800)))
def test_native_year_route_uses_existing_gpu_guards_and_exact_deadline(tmp_path,basename,seconds):
    _,args,prefix,source=_fixture(tmp_path,basename)
    signature='  if [[ "'+'$'+'{3:-}" == "$NATIVE_M5_FULL_GC_FIT_TRAINER_MODULE" ]]; then'
    branch=signature+source.split(signature,1)[1].split('\n  if ! is_direct_python',1)[0]
    shell=prefix+'\nATTENDED_SMOKE=false\nCUDA_PRODUCER_GUARD=false\nvalidate_native_route() {\n'+branch+'\n}\n'
    shell+='validate_native_route '+shlex.join(args)+'\n'
    shell+='printf "%s\\n" "$TRAINER_DEVICE" "$TRAINER_OUT_BUNDLE_DIR" "$TRAINER_MAX_WALL_SECONDS" '
    shell+='"$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_GPU_MAX_POWER_LIMIT_W" "$TRAINER_GPU_MAX_POWER_DRAW_W" "$TRAINER_GPU_MAX_CORE_TEMP_C"\n'
    result=subprocess.run(['bash','-c',shell],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.splitlines()==['cuda',args[10],str(seconds),str(seconds),'300','310','85']


@pytest.mark.parametrize('schema,seconds',[
    ('wrong',1800),('gx1_native_m5_main_checkpoint_window_plan_v1',0),
    ('gx1_native_m5_main_checkpoint_window_plan_v1',1801),
    ('gx1_native_m5_main_checkpoint_window_plan_v1',10801)])
def test_checkpoint_window_rejects_unsupported_guard_deadline(tmp_path,schema,seconds):
    path,args,prefix,source=_fixture(tmp_path,'NATIVE_M5_TA_ATR03_FIVE_YEAR_WINDOW_20261120_001')
    path.write_text(json.dumps({'schema_version':schema,'execution':{'outer_seconds':seconds}}))
    args[6]=hashlib.sha256(path.read_bytes()).hexdigest()
    signature='  if [[ "'+'$'+'{3:-}" == "$NATIVE_M5_FULL_GC_FIT_TRAINER_MODULE" ]]; then'
    branch=signature+source.split(signature,1)[1].split('\n  if ! is_direct_python',1)[0]
    shell=prefix+'\nATTENDED_SMOKE=false\nCUDA_PRODUCER_GUARD=false\nvalidate_native_route() {\n'+branch+'\n}\n'
    result=subprocess.run(['bash','-c',shell+'validate_native_route '+shlex.join(args)],capture_output=True,text=True)
    assert result.returncode==75
    assert 'exact supported execution deadline'in result.stderr


def _future_cpu_fixture(tmp_path):
    folder=tmp_path/'configs/research';folder.mkdir(parents=True)
    plan=folder/'NATIVE_M5_FUTURE_STATE_CPU_MECHANICS_20261121_001.json'
    plan.write_text('{"synthetic_guard_fixture":true}\n')
    python=str(tmp_path/'.venv/bin/python')
    args=[python,'-m','gx1.audit.qualify_native_m5_full_gc_direction_v1',
        '--native-future-state-cpu-mechanics','--plan-sha256',
        hashlib.sha256(plan.read_bytes()).hexdigest(),'--lane','continuous']
    source=RUNNER.read_text()
    signature='matches_bound_native_future_state_cpu_mechanics() {'
    function=signature+source.split(signature,1)[1].split('\n}\n',1)[0]+'\n}\n'
    prefix='\n'.join(['set -eu','REPO_ROOT='+shlex.quote(str(tmp_path)),
        'CANONICAL_TRAINER_PYTHON='+shlex.quote(python),
        'NATIVE_M5_FULL_GC_FIT_TRAINER_MODULE='+shlex.quote(args[2]),function])
    return plan,args,prefix,source


@pytest.mark.parametrize('variant',('continuous','replay','preflight','wrong_sha','bad_sha',
    'wrong_lane','extra_flag','extra_plan','wrong_module','wrong_python','wrong_flag',
    'reordered','symlink_plan','missing_plan'))
def test_native_future_state_cpu_mechanics_exact_command(tmp_path,variant):
    plan,args,prefix,_=_future_cpu_fixture(tmp_path)
    if variant=='replay':args[-1]='replay'
    elif variant=='preflight':args.append('--preflight')
    elif variant=='wrong_sha':args[5]='0'*64
    elif variant=='bad_sha':args[5]='true'
    elif variant=='wrong_lane':args[7]='fit'
    elif variant=='extra_flag':args.append('--train')
    elif variant=='extra_plan':args.extend(['--plan','/tmp/other.json'])
    elif variant=='wrong_module':args[2]='gx1.scripts.research_ta_campaign_v1'
    elif variant=='wrong_python':args[0]='/usr/bin/python3'
    elif variant=='wrong_flag':args[3]='--native-future-state-cpu-mechanics-extra'
    elif variant=='reordered':args[4:8]=args[6:8]+args[4:6]
    elif variant=='symlink_plan':
        real=plan.with_suffix('.original');plan.rename(real);plan.symlink_to(real)
    elif variant=='missing_plan':plan.unlink()
    result=subprocess.run(['bash','-c',prefix+'\nmatches_bound_native_future_state_cpu_mechanics '+shlex.join(args)],capture_output=True,text=True)
    assert (result.returncode==0)==(variant in ('continuous','replay','preflight')),result.stderr


@pytest.mark.parametrize('variant',('valid','preflight','audit','trainer','wrong_memory',
    'wrong_swap','cuda','attended','wrong_lane','extra','wrapped_python'))
def test_native_future_state_cpu_mechanics_dispatch_is_private_exact_scope(tmp_path,variant):
    _,args,prefix,source=_future_cpu_fixture(tmp_path)
    begin='  if [[ "$native_cpu_mechanics_reference" == true ]]; then'
    branch=begin+source.split(begin,1)[1].split('\n  if [[ "$JOB_CLASS" == "audit"',1)[0]
    job='producer';memory='12G';swap='512M';cuda='false';attended='false'
    if variant=='preflight':args.append('--preflight')
    elif variant in ('audit','trainer'):job=variant
    elif variant=='wrong_memory':memory='20G'
    elif variant=='wrong_swap':swap='0'
    elif variant=='cuda':cuda='true'
    elif variant=='attended':attended='true'
    elif variant=='wrong_lane':args[-1]='main'
    elif variant=='extra':args.append('--optimizer-steps=40')
    elif variant=='wrapped_python':args=['env',*args]
    shell=prefix+'\n'+ '\n'.join(['native_cpu_mechanics_reference=true',
        'JOB_CLASS='+job,'MEM='+memory,'SWAP='+swap,'CUDA_PRODUCER_GUARD='+cuda,
        'ATTENDED_SMOKE='+attended,'validate_future_cpu() {',branch,'}',
        'validate_future_cpu '+shlex.join(args),
        'printf "%s\\n" "$NATIVE_FUTURE_STATE_CPU_MECHANICS" "$NATIVE_FUTURE_STATE_CPU_MECHANICS_SHA256" "$NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE"'])
    result=subprocess.run(['bash','-c',shell],capture_output=True,text=True)
    if variant in ('valid','preflight'):
        assert result.returncode==0,result.stderr
        assert result.stdout.splitlines()==['true',args[5],'continuous']
    else:
        assert result.returncode==75
        assert 'CPU mechanics is not exactly bound' in result.stderr


def test_native_future_state_cpu_mechanics_timeout_and_mode_proof_are_runner_owned():
    source=RUNNER.read_text()
    assert 'if [[ "$target_arg" == *--native-future-state-cpu-mechanics* ]]; then' in source
    assert source.index('validate_target_command "$@"') < source.index('systemd-run --user --scope')
    scoped=source.split('if [[ "$GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS" == true ]]; then',1)[1].split('\nfi',1)[0]
    assert '/usr/bin/timeout --signal=TERM --kill-after=10s 900s "$@"' in scoped
    for key in ('GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS',
                'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_PLAN_SHA256',
                'GX1_NATIVE_FUTURE_STATE_CPU_MECHANICS_LANE'):
        assert '--setenv='+key+'=' in source



def _future_cuda_fixture(tmp_path):
    folder=tmp_path/'configs/research';folder.mkdir(parents=True)
    plans={}
    for role in ('MECHANICS','INITIAL'):
        plan=folder/('NATIVE_M5_FUTURE_STATE_CUDA_'+role+'_20261121_001.json')
        plan.write_text(json.dumps({'synthetic_guard_role':role})+'\n');plans[role]=plan
    python=str(tmp_path/'.venv/bin/python')
    args=[python,'-m','gx1.audit.qualify_native_m5_full_gc_direction_v1',
        '--native-future-state-cuda-qualification','--plan-sha256',
        hashlib.sha256(plans['MECHANICS'].read_bytes()).hexdigest(),'--lane','continuous']
    source=RUNNER.read_text();signature='matches_bound_native_future_state_cuda_qualification() {'
    function=signature+source.split(signature,1)[1].split('\n}\n',1)[0]+'\n}\n'
    prefix='\n'.join(['set -eu','REPO_ROOT='+shlex.quote(str(tmp_path)),
        'CANONICAL_TRAINER_PYTHON='+shlex.quote(python),
        'NATIVE_M5_FULL_GC_FIT_TRAINER_MODULE='+shlex.quote(args[2]),function])
    return plans,args,prefix,source


@pytest.mark.parametrize('variant', ('continuous','replay','initial','preflight','wrong_sha','bad_sha',
    'wrong_lane','extra_flag','extra_plan','wrong_module','wrong_python','wrong_flag',
    'reordered','symlink_plan','missing_plan','cross_plan_sha'))
def test_native_future_state_cuda_qualification_exact_command(tmp_path,variant):
    plans,args,prefix,_=_future_cuda_fixture(tmp_path);plan=plans['MECHANICS']
    if variant=='replay':args[-1]='replay'
    elif variant=='initial':
        args[-1]='initial';args[5]=hashlib.sha256(plans['INITIAL'].read_bytes()).hexdigest()
    elif variant=='preflight':args.append('--preflight')
    elif variant=='wrong_sha':args[5]='0'*64
    elif variant=='bad_sha':args[5]='invalid'
    elif variant=='wrong_lane':args[-1]='fit'
    elif variant=='extra_flag':args.append('--train')
    elif variant=='extra_plan':args.extend(['--plan',str(plan)])
    elif variant=='wrong_module':args[2]='gx1.scripts.research_ta_campaign_v1'
    elif variant=='wrong_python':args[0]='/usr/bin/python3'
    elif variant=='wrong_flag':args[3]+='-extra'
    elif variant=='reordered':args[4:8]=args[6:8]+args[4:6]
    elif variant=='symlink_plan':
        actual=plan.with_suffix('.original');plan.rename(actual);plan.symlink_to(actual)
    elif variant=='missing_plan':plan.unlink()
    elif variant=='cross_plan_sha':args[-1]='initial'
    result=subprocess.run(['bash','-c',prefix+'\nmatches_bound_native_future_state_cuda_qualification '+shlex.join(args)],
        capture_output=True,text=True)
    assert (result.returncode==0)==(variant in ('continuous','replay','initial','preflight')),result.stderr


@pytest.mark.parametrize('variant', ('continuous','replay','initial','preflight','audit','producer',
    'wrong_memory','wrong_swap','cuda','attended','benchmark','extra','output_exists','output_symlink'))
def test_native_future_state_cuda_qualification_dispatch_is_private_canonical_scope(tmp_path,variant):
    plans,args,prefix,source=_future_cuda_fixture(tmp_path)
    begin='  if [[ "$native_cuda_qualification_reference" == true ]]; then'
    branch=begin+source.split(begin,1)[1].split('\n  if [[ "$native_cpu_mechanics_reference"',1)[0]
    # Only the synthetic output prefix changes; actual GX1_RUNS is never touched.
    output_parent=tmp_path/'runs';output_parent.mkdir()
    branch=branch.replace('/home/andre2/GX1_RUNS',str(output_parent))
    job='trainer';memory='20G';swap='512M';cuda='false';attended='false';benchmark=''
    if variant=='replay':args[-1]='replay'
    elif variant=='initial':
        args[-1]='initial';args[5]=hashlib.sha256(plans['INITIAL'].read_bytes()).hexdigest()
    elif variant=='preflight':args.append('--preflight')
    elif variant in ('audit','producer'):job=variant
    elif variant=='wrong_memory':memory='12G'
    elif variant=='wrong_swap':swap='0'
    elif variant=='cuda':cuda='true'
    elif variant=='attended':attended='true'
    elif variant=='benchmark':benchmark='/unit/other.json'
    elif variant=='extra':args.append('--max-wall-seconds=9999')
    output=output_parent/('NATIVE_M5_FUTURE_STATE_CUDA_INITIAL_20261121_001' if args[7]=='initial'
        else 'NATIVE_M5_FUTURE_STATE_CUDA_MECHANICS_20261121_001/'+args[7])
    output.parent.mkdir(exist_ok=True)
    # Existing mechanics parent may contain guard logs; output itself stays fresh.
    (output.parent/'synthetic.guard.log').write_text('guard fixture\n')
    if variant=='output_exists':output.mkdir()
    elif variant=='output_symlink':output.symlink_to(tmp_path/'missing-output')
    shell=prefix+'\n'+'\n'.join(['native_cuda_qualification_reference=true',
        'JOB_CLASS='+job,'MEM='+memory,'SWAP='+swap,'CUDA_PRODUCER_GUARD='+cuda,
        'ATTENDED_SMOKE='+attended,'POWER_BENCHMARK_SCOPE_JSON='+shlex.quote(benchmark),
        'POWER_BENCHMARK_SCOPE_SHA256=','TRAINER_EXECUTION_MODE=canonical',
        'TRAINER_GPU_MAX_MEMORY_TEMP_C=80','TRAINER_GPU_MAX_MEMORY_USED_MIB=12288',
        'TRAINER_GPU_MONITOR_INTERVAL_SECONDS=1','validate_future_cuda() {',branch,'}',
        'validate_future_cuda '+shlex.join(args),
        'printf "%s\\n" "$NATIVE_FUTURE_STATE_CUDA_QUALIFICATION" "$NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256" '
        '"$NATIVE_FUTURE_STATE_CUDA_LANE" "$TRAINER_EXECUTION_MODE" "$TRAINER_DEVICE" "$TRAINER_OUT_BUNDLE_DIR" '
        '"$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_GPU_MAX_POWER_LIMIT_W" '
        '"$TRAINER_GPU_MAX_POWER_DRAW_W" "$TRAINER_GPU_MAX_CORE_TEMP_C" "$TRAINER_GPU_MAX_MEMORY_TEMP_C" '
        '"$TRAINER_GPU_MAX_MEMORY_USED_MIB" "$TRAINER_GPU_MONITOR_INTERVAL_SECONDS"'])
    result=subprocess.run(['bash','-c',shell],capture_output=True,text=True)
    if variant in ('continuous','replay','initial','preflight'):
        assert result.returncode==0,result.stderr
        seconds='5400' if args[7]=='initial' else '1800'
        assert result.stdout.splitlines()==['true',args[5],args[7],'canonical','cuda',str(output),
            seconds,seconds,'300','310','85','80','12288','1']
    else:
        assert result.returncode==75
        assert 'native future-state CUDA' in result.stderr



def test_native_future_state_cuda_qualification_precedes_old_selectors_and_exports_only_bound_mode():
    source=RUNNER.read_text()
    begin=source.index('  if [[ "$native_cuda_qualification_reference" == true ]]; then')
    assert begin < source.index('  if [[ "$native_cpu_mechanics_reference" == true ]]; then')
    assert begin < source.index('  if [[ "$JOB_CLASS" == "audit" || "$JOB_CLASS" == "producer" ]]; then')
    branch=source[begin:].split('\n  if [[ "$native_cpu_mechanics_reference"',1)[0]
    assert branch.index('matches_bound_native_future_state_cuda_qualification "$@"') < branch.index('NATIVE_FUTURE_STATE_CUDA_QUALIFICATION=true')
    assert 'TRAINER_EXECUTION_MODE=' not in branch
    for key in ('GX1_NATIVE_FUTURE_STATE_CUDA_QUALIFICATION','GX1_NATIVE_FUTURE_STATE_CUDA_PLAN_SHA256',
                'GX1_NATIVE_FUTURE_STATE_CUDA_LANE'):
        assert '--setenv='+key+'=' in source
    nested=source.split('  if [[ "$NATIVE_FUTURE_STATE_CUDA_QUALIFICATION" == true ]]; then',1)[1]
    assert nested.index('nested native CUDA qualification differs') < nested.index('exec "$GPU_GUARD_PATH" "$@"')
    assert 'NATIVE_M5_FUTURE_STATE_CUDA_MECHANICS_20261121_001/$8' in branch
    assert 'NATIVE_M5_FUTURE_STATE_CUDA_INITIAL_20261121_001' in branch
