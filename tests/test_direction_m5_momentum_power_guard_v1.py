import copy
import hashlib
import os
import subprocess
from pathlib import Path

import pytest
from gx1.audit import direction_m5_momentum_intervention_v1 as audit
from tests.test_gx1_capped_run_contract import TRAINER_GUARD, _fake_nvidia_smi, _guard_env

REPO = Path(__file__).resolve().parents[1]


def test_exact_250w_selector_requires_all_arguments_and_bound_plan(tmp_path):
    source=(REPO/'scripts/gx1_capped_run.sh').read_text()
    body=source.split('matches_bound_m5_intervention() {\n',1)[1].split('\n}\n',1)[0]
    shell='matches_bound_m5_intervention() {\n'+body+'\n}\nmatches_bound_m5_intervention "$@"'
    plan=tmp_path/'plan.json';plan.write_text('{"synthetic":true}')
    output=str(tmp_path/'output');digest=hashlib.sha256(plan.read_bytes()).hexdigest()
    python='/canonical/python';module='gx1.scripts.validate_entry_model_native_technical_checkpoint_v1'
    target=[python,'-m',module,'--direction-m5-momentum-intervention-plan',str(plan),
        '--direction-m5-momentum-intervention-plan-sha256',digest,'--device','cuda','--out-dir',output]
    env={**os.environ,'CANONICAL_TRAINER_PYTHON':python,'TECHNICAL_VALIDATION_PRODUCER_MODULE':module}
    def invoke(args,expected_plan=str(plan)):
        return subprocess.run(['bash','-c',shell,'selector',expected_plan,output,*args],env=env,capture_output=True).returncode
    assert invoke(target)==0
    changed=target.copy();changed[6]='0'*64;assert invoke(changed)!=0
    changed=target.copy();changed[2]='unrelated.module';assert invoke(changed)!=0
    changed=target.copy();changed[-1]=output+'-other';assert invoke(changed)!=0
    assert invoke(target+['--device','cuda'])!=0
    symlink=tmp_path/'link.json';symlink.symlink_to(plan);changed=target.copy();changed[4]=str(symlink)
    assert invoke(changed,expected_plan=str(symlink))!=0
    plan.write_text('{"synthetic":false}');assert invoke(target)!=0


@pytest.mark.parametrize('telemetry,passes,reason',[
    ('50, 70, 130, 250, 1000',True,''),
    ('50, 70, 171, 250, 1000',False,'GPU draw'),
    ('50, 70, 130, 251, 1000',False,'configured GPU power limit'),
    ('66, 70, 130, 250, 1000',False,'core temperature'),
    ('50, 81, 130, 250, 1000',False,'memory temperature'),
])
def test_guard_retains_draw_and_thermal_stops_at_250w(tmp_path,telemetry,passes,reason):
    fake=_fake_nvidia_smi(tmp_path,telemetry)
    env=_guard_env(device='cuda',nvidia_smi_path=fake,execution_mode='cuda_producer',
        max_power_limit_w=250,max_power_draw_w=170)
    env.update(GX1_CAPPED_CLASS='producer',GX1_TRAINER_GPU_MAX_CORE_TEMP_C='65',GX1_TRAINER_GPU_MAX_MEMORY_TEMP_C='80')
    result=subprocess.run(['bash',str(TRAINER_GUARD),'/bin/true'],cwd=REPO,env=env,text=True,capture_output=True,timeout=15)
    if passes:assert result.returncode==0,result.stderr
    else:assert result.returncode==75 and reason in result.stderr


def test_historical_fit_provenance_rejects_mismatch_and_pins_current_runner():
    runner='scripts/gx1_capped_run.sh';guard='scripts/gx1_guarded_trainer_exec.sh';old=b'original runner bytes'
    oldref={'path':'/repo/'+runner,'sha256':hashlib.sha256(old).hexdigest()}
    newref={'path':oldref['path'],'sha256':'2'*64};guardref={'path':'/repo/'+guard,'sha256':'3'*64}
    owners={runner:oldref,guard:guardref,**{f'other_{i}':{'path':f'/repo/{i}','sha256':'4'*64} for i in range(61)}}
    recipe={'path':'/repo/fit.json','sha256':'5'*64};reviewref={'path':'/repo/review.json','sha256':'6'*64};review={'source_commit':'7'*40}
    plan={'inputs':{'recipe':recipe,'fit_review':reviewref},'sources':{runner:newref,guard:guardref}}
    launch={'source_owners':owners,'candidate_mode':'price_only'}
    proof={'status':'PASS','fit_authority_full_validation_passed':True,'review_and_fit_source_hashes_verified':True,
        'fit_source_owners_count':63,'model_forward_calls':0,'optimizer_steps':0,'test_data_used':False,
        'recipe':recipe,'review':reviewref,'historical_fit_source_head':review['source_commit'],
        'historical_and_current_execution_sources_equal':{runner:oldref,guard:guardref}}
    frozen=audit.verified_fit_source_transition(proof,plan,launch,review,old)
    assert frozen['source_owners'][runner]==newref and launch['source_owners'][runner]==oldref
    assert frozen['source_owners'][guard]==guardref and len(frozen['source_owners'])==63
    with pytest.raises(RuntimeError,match='FIT_SOURCE_TRANSITION'):
        audit.verified_fit_source_transition(proof,plan,launch,review,b'tampered')
    corrupt=copy.deepcopy(proof);corrupt['fit_authority_full_validation_passed']=False
    with pytest.raises(RuntimeError,match='FIT_SOURCE_TRANSITION'):
        audit.verified_fit_source_transition(corrupt,plan,launch,review,old)
    corruptplan=copy.deepcopy(plan);corruptplan['sources'][guard]['sha256']='9'*64
    with pytest.raises(RuntimeError,match='FIT_SOURCE_TRANSITION'):
        audit.verified_fit_source_transition(proof,corruptplan,launch,review,old)
