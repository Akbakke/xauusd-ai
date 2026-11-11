"""Execute the actual frozen-head command gate; no launcher or model."""
import hashlib
import os
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE = 'gx1.audit.observe_market_state_ta_c171_short_heads_v1'
FLAG = '--market-state-ta-c171-short-heads-plan'
ROOT = '/home/andre2/GX1_RUNS/MARKET_STATE_TA_C171_FULL_USED_PREFIX_20261111_001'


def gate(tmp_path, changes=None, extra=()):
    source = (REPO/'scripts/gx1_capped_run.sh').read_text()
    name = 'matches_bound_market_state_ta_c171_short_heads'
    helper = name+'() {\n'+source.split(name+'() {\n',1)[1].split('\n}\n',1)[0]+'\n}\n'
    lines = source.splitlines()
    start = next(i for i,line in enumerate(lines) if line.strip() == 'if [[ "${3:-}" == "$TA_C171_HEAD_PRODUCER_MODULE" ]]; then')
    end = next(i for i in range(start,len(lines)) if lines[i].strip() == 'TRAINER_MODEL_MAX_WALL_SECONDS=1200')+2
    block = '\n'.join(lines[start:end])
    assert block.count(ROOT) == 1
    # Map only the deployment directory to the temporary filesystem fixture.
    root = tmp_path/'capture';root.mkdir(exist_ok=True)
    plan=root/'PLAN.json';plan.write_text('{"synthetic":true}')
    digest=hashlib.sha256(plan.read_bytes()).hexdigest()
    target=['/canonical/python','-m',MODULE,FLAG,str(plan),FLAG+'-sha256',digest,
            '--device','cuda','--out-dir',str(root)]
    for index,value in (changes or {}).items():
        target[index]=value.replace('$ROOT',str(root))
    script=helper+'''invoke() {
NUMERICAL_THREAD_COUNT=1
TRAINER_GPU_MAX_POWER_LIMIT_W=160
TRAINER_MAX_WALL_SECONDS=3600
TRAINER_MODEL_MAX_WALL_SECONDS=3600
TRAINER_GPU_MAX_POWER_DRAW_W=170
TRAINER_GPU_MAX_CORE_TEMP_C=65
TRAINER_GPU_MAX_MEMORY_TEMP_C=80
'''+block.replace(ROOT,str(root))+'''
printf '%s %s %s %s %s %s %s\\n' "$TRAINER_GPU_MAX_POWER_LIMIT_W" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_GPU_MAX_POWER_DRAW_W" "$TRAINER_GPU_MAX_CORE_TEMP_C" "$TRAINER_GPU_MAX_MEMORY_TEMP_C" "$NUMERICAL_THREAD_COUNT"
}
invoke "$@"
'''
    env={**os.environ,'CANONICAL_TRAINER_PYTHON':'/canonical/python',
         'TA_C171_HEAD_PRODUCER_MODULE':MODULE}
    return subprocess.run(['bash','-c',script,'head-gate',*target,*extra],env=env,
                          capture_output=True,text=True,timeout=10)


def test_actual_exact_observer_route_preserves_draw_and_temperature_stops(tmp_path):
    result=gate(tmp_path)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='250 1200 1200 170 65 80 8'


@pytest.mark.parametrize('changes,extra',[
    ({4:'/home/andre2/GX1_RUNS/MARKET_STATE_TA_C171_SHORT_INITIAL_HEADS_20261111_001/PLAN.json'},()),
    ({4:'/home/andre2/GX1_RUNS/MARKET_STATE_TA_C171_SHORT_INITIAL_HEADS_20261111_002/PLAN.json'},()),
    ({4:'/home/andre2/GX1_RUNS/MARKET_STATE_TA_C171_QUERY_PROJECTION_GRADIENT_20261111_001/PLAN.json'},()),
    ({4:'/home/andre2/GX1_RUNS/MARKET_STATE_TA_C171_SHORT_JOINT_BOUNDARIES_20261111_001/PLAN.json'},()),
    ({3:'--unrelated-plan'},()),
    ({4:'$ROOT/other.json'},()),
    ({6:'0'*64},()),
    ({8:'cpu'},()),
    ({10:'$ROOT/other'},()),
    ({},('--preflight',)),
    ({},('--device','cuda')),
])
def test_actual_observer_selector_fails_before_guard_or_claim(tmp_path,changes,extra):
    result=gate(tmp_path,changes,extra)
    assert result.returncode==75 and result.stdout==''
    assert 'not exactly bound' in result.stderr


def test_existing_dispatcher_does_not_receive_new_observer_exception(tmp_path):
    result=gate(tmp_path,{2:'gx1.scripts.validate_entry_model_native_technical_checkpoint_v1'})
    assert result.returncode==0 and result.stdout.strip()=='160 3600 3600 170 65 80 1'


def test_dispatch_preflight_has_no_run_and_rejects_ambiguous_flags(monkeypatch,tmp_path):
    from gx1.audit import observe_market_state_ta_c171_short_heads_v1 as observer
    dispatcher=observer
    calls=[]
    monkeypatch.setattr(observer,'preflight',lambda **kw:calls.append(('preflight',kw)) or {})
    monkeypatch.setattr(observer,'run',lambda **kw:calls.append(('run',kw)) or {})
    args=[FLAG,str(tmp_path/'PLAN.json'),FLAG+'-sha256','a'*64,
          '--device','cuda','--out-dir',str(tmp_path)]
    assert dispatcher.main(args+['--preflight'])==0
    assert [x[0] for x in calls]==['preflight']
    assert calls[0][1]['plan_sha']=='a'*64
    assert dispatcher.main(args)==0
    assert calls[1][0]=='run' and calls[1][1]['device']=='cuda'
    for extra in (['--device','cuda'],['--preflight','--preflight'],['--train']):
        with pytest.raises(SystemExit):dispatcher.main(args+extra)
