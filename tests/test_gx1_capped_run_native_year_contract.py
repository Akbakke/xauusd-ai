"""Exact YEAR routes only; shell extraction never launches guards, data or CUDA."""
from __future__ import annotations
import hashlib
import shlex
import subprocess
from pathlib import Path
import pytest

REPO=Path(__file__).resolve().parents[1]
RUNNER=REPO/'scripts/gx1_capped_run.sh'
BASENAMES=('NATIVE_M5_TA_ATR03_YEAR_PREPARE_20261120_001',
    'NATIVE_M5_TA_ATR03_YEAR_MECHANICS_20261120_001',
    'NATIVE_M5_TA_ATR03_YEAR_FIT_20261120_001')


def _fixture(tmp_path,basename):
    folder=tmp_path/'configs/research';folder.mkdir(parents=True)
    path=folder/(basename+'.json');path.write_text('{"test_only":true}\n')
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
    elif variant=='cross_output':args[10]='/home/andre2/GX1_RUNS/'+BASENAMES[(BASENAMES.index(basename)+1)%3]
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


@pytest.mark.parametrize('basename,seconds',zip(BASENAMES,(1800,1800,3600)))
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
