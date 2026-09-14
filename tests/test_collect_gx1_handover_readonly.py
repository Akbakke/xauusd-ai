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
    clock=repo/'clock.ps1';clock.write_text('fixture')
    p['gpu_clock_launcher']={'path':str(clock),'sha256':hashlib.sha256(clock.read_bytes()).hexdigest()}
    path=repo/'NEXT_RUN_POLICY.json';path.write_text(json.dumps(p))
    _git(repo,'add','.');_git(repo,'commit','-qm','policy fixture')
    return path,p


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
    assert result.returncode==78
    assert 'operator_stop_not_resolved' in result.stdout


def test_handover_has_no_legacy_fallback():
    source=Path(__file__).resolve().parents[1]/'scripts/gx1_handover.sh'
    text=source.read_text()
    assert 'COMPLETED_RUN.json' in text and 'NEXT_RUN_POLICY.json' in text
    assert 'PROJECT_STATE_xau_direction_launch' not in text
    assert '--require-next-run' not in text
