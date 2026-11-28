"""Exact V3 development admission/resource boundaries; no data or hardware IO."""
import copy
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from gx1.contracts import gx1_capped_execution_v1 as g

NAME = 'NATIVE_ALL_TF_PROVIDER_HLC_V3_DEVELOPMENT_20261127_001'
PHASES = ('cohort', 'warm', 'cold', 'fit')
OWNERS = ('gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh')


def fixture(monkeypatch, tmp_path, phase='fit', preflight=False, bad=None):
    repo, runs = tmp_path / 'repo', tmp_path / 'runs'
    (repo / 'configs/research').mkdir(parents=True)
    runs.mkdir()
    names = (*OWNERS, 'gx1/audit/native_provider_hlc_learning_v1.py')
    for name in names:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# source-only fixture\n')
    source_owners = {n: hashlib.sha256((repo / n).read_bytes()).hexdigest() for n in names}
    recipe = dict(passes=2, seed=20260911, train_batch_rows=96, eval_batch_rows=256, gradient_clip_norm=1.)
    selection = dict(rows=20727, whole_TRAIN_rows=354550,
        selected_issue_ns_sha256='ca9868f542d290045c2640c320d1e3517329c8a4223c8ec3a58a153d86a283f1',
        selected_source_rows_sha256='84fca333ffd882d679e6339ea9663811b9d16564cb8f8617072753f4f6db5025',
        target_filter=False, top_up=False, quotas=False)
    if bad == 'selection': selection['rows'] += 1
    if bad == 'target_filter': selection['target_filter'] = True
    if bad == 'recipe': recipe['passes'] = 3
    protocol = dict(schema_version='gx1_native_all_tf_provider_hlc_v3_development_protocol_v1',
        experiment_id=NAME, recipe=recipe, selection=selection)
    proto = repo / 'configs/research' / (NAME.replace('_20261127_001', '_PROTOCOL_20261127_001') + '.json')
    proto.write_text(json.dumps(protocol, sort_keys=True))
    protocol_ref = dict(path=str(proto), sha256=hashlib.sha256(proto.read_bytes()).hexdigest())
    if bad == 'protocol_digest': protocol_ref['sha256'] = None
    bridge = dict(producer_commit=g._V3_DEVELOPMENT_PARENT,
        owners={n: dict(original_sha256='e' * 64, current_sha256=source_owners[n]) for n in OWNERS})
    target = dict(schema_version='gx1_native_m5_fully_future_state_dataset_v3_all_tf_provider_hlc',
        contract_sha256='58fd7f15bdf35646680829a718caa2991f2029ed99f48030e3a85c86aae380c5')
    contract = dict(contract_sha256='ded7f2a05417d108ef8bed4d88f31d40ea1de21292e964586598ddaa818010c3',
        target_contract_sha256=target['contract_sha256'], target_metadata_as_model_input=False)
    output = runs / (NAME + '_' + phase + ('_004' if phase == 'fit' else ''))
    plan = dict(schema_version='gx1_native_all_tf_provider_hlc_v3_development_plan_v1', phase=phase,
        stage=(NAME + '_' + phase).lower(), source_commit='a' * 40, output_root=str(output),
        execution=g._native_v3_development_execution(phase), protocol=protocol_ref,
        recipe=recipe, limits={'optimizer_steps':dict(cohort=0, warm=3, cold=1, fit=432)[phase]},
        metadata={'target_contract':target}, model_contract=contract, source_owners=source_owners,
        guard_source_bridge=bridge)
    if bad == 'schema': plan['schema_version'] = 'old'
    if bad == 'model': contract['contract_sha256'] = 'b' * 64
    if bad == 'target': target['schema_version'] = 'v2'
    if bad == 'execution': plan['execution']['tasks_max'] += 1
    if bad == 'updates': plan['limits']['optimizer_steps'] += 1
    if bad == 'owner': source_owners[OWNERS[0]] = 'f' * 64
    if bad == 'bridge': plan['guard_source_bridge'] = {**bridge, 'extra':True}
    if bad == 'old_fit_root': plan['output_root'] = str(runs / (NAME + '_fit'))
    if bad == 'old_fit002_root': plan['output_root'] = str(runs / (NAME + '_fit_002'))
    if bad == 'old_fit003_root': plan['output_root'] = str(runs / (NAME + '_fit_003'))
    path = repo / 'configs/research' / (NAME + '_' + phase.upper() + ('_002' if phase == 'warm' else '_004' if phase == 'fit' else '') + '.json')
    path.write_text(json.dumps(plan, sort_keys=True))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    gate = dict(status='ADMITTED_EXACT_V3_DEVELOPMENT_' + phase.upper(),
        plan=dict(path=str(path), sha256=digest), execution_authorized=True, remaining_invocations=1,
        relaunch_allowed=False, training_authorized=phase != 'cohort', test_authorized=False)
    parent = dict(protocol=protocol_ref, main_fit_authorized=False, test_authorized=False,
        spending_authorized=False, execution_authorized=not preflight,
        active_phase=None if preflight else phase, status='PREPARED' if preflight else gate['status'],
        phase_gates={} if preflight else {phase:gate})
    if bad == 'parent_protocol': parent['protocol'] = {**protocol_ref, 'sha256':'0' * 64}
    if bad == 'parent_phase': parent['active_phase'] = 'wrong'
    if bad == 'spent': gate['remaining_invocations'] = 0
    if bad == 'second_gate': parent['phase_gates']['cohort' if phase != 'cohort' else 'fit'] = gate
    policy = {k:bool(not preflight and phase != 'cohort' and k in ('training_enabled','native_training_enabled'))
        for k in ('training_enabled','native_training_enabled','native_launch_authorized','full_epoch_training_allowed','full_val_allowed')}
    policy['exit_training_and_evaluation_this_wave'] = False
    policy['native_v3_bounded_development_test_20261127_001'] = parent
    if bad == 'flags': policy['full_epoch_training_allowed'] = True
    (repo / 'NEXT_RUN_POLICY.json').write_text(json.dumps(policy))
    if bad == 'existing': output.mkdir()
    if bad == 'symlink':
        original = path.with_suffix('.original'); path.rename(original); path.symlink_to(original)
    real_path = Path
    def adapted(value): return runs if str(value) == '/home/andre2/GX1_RUNS' else real_path(value)
    adapted.cwd = real_path.cwd
    monkeypatch.setattr(g, 'Path', adapted)
    monkeypatch.setattr(g, '__file__', str(repo / OWNERS[0]))
    monkeypatch.chdir(repo)
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **k:'a' * 40 + '\n')
    monkeypatch.setattr(g, 'native_provider_hlc_v3_development_source_bridge', lambda *a, **k:bridge)
    monkeypatch.setattr(sys, 'executable', str(repo / '.venv/bin/python'))
    argv = [str(repo / names[2]), '--v3-development-plan-sha256', digest, '--phase', phase]
    if preflight: argv += ['--preflight']
    if bad == 'argv': argv += ['--unsafe']
    monkeypatch.setattr(sys, 'argv', argv)
    for k in list(os.environ):
        if k.startswith('GX1_NATIVE_'): monkeypatch.delenv(k)
    env = dict(GX1_NATIVE_V3_DEVELOPMENT='true', GX1_NATIVE_V3_DEVELOPMENT_PLAN_SHA256=digest,
        GX1_NATIVE_V3_DEVELOPMENT_PHASE=phase, GX1_POWER_BENCHMARK_SCOPE_JSON='', GX1_POWER_BENCHMARK_SCOPE_SHA256='')
    if phase != 'cohort':
        fields = dict(MAX_WALL_SECONDS=5400 if phase == 'fit' else 1800,
            MODEL_MAX_WALL_SECONDS=5370 if phase == 'fit' else 1770, ATTENDED_STAGE_REQUIRED='false',
            GPU_INDEX=0,GPU_MAX_POWER_LIMIT_W=250,GPU_MAX_POWER_DRAW_W=250,GPU_MAX_CORE_TEMP_C=75,
            GPU_MAX_MEMORY_TEMP_C=80,GPU_MAX_MEMORY_USED_MIB=12288,GPU_MONITOR_INTERVAL_SECONDS=1,
            CPU_MAX_TEMP_C=80,CPU_MAX_PACKAGE_POWER_W=125,HOST_MIN_AVAILABLE_MEMORY_BYTES=8589934592,
            HOST_MIN_COMMIT_HEADROOM_BYTES=8589934592)
        env.update({'GX1_TRAINER_' + k:str(v) for k,v in fields.items()})
    if bad == 'hardware_env': env['GX1_TRAINER_GPU_MAX_POWER_DRAW_W'] = '251'
    if bad == 'unknown_env': env['GX1_NATIVE_V3_DEVELOPMENT_UNSAFE'] = 'false'
    if bad == 'legacy_env': env['GX1_NATIVE_PROVIDER_HLC_ALL_TF_LEARNING'] = 'true'
    for k,v in env.items(): monkeypatch.setenv(k,v)
    called = []
    monkeypatch.setattr(g, 'require_guarded_cuda_trainer_execution', lambda:called.append('cuda') or {'kernel':True})
    monkeypatch.setattr(g, 'require_capped_cpu_research_producer_execution', lambda:called.append('cpu') or {'kernel':True})
    return dict(repo=repo,runs=runs,path=path,digest=digest,output=output,phase=phase,called=called,
        call=lambda:g.require_native_all_tf_provider_hlc_v3_development_execution(plan_sha256=digest,phase=phase))


@pytest.mark.parametrize('phase', PHASES)
@pytest.mark.parametrize('preflight', (False, True))
def test_exact_phase_guard_preserves_resource_only_authority(monkeypatch,tmp_path,phase,preflight):
    f = fixture(monkeypatch,tmp_path,phase,preflight)
    proof = f['call']()
    assert proof['plan']['sha256'] == f['digest'] and proof['phase'] == phase
    assert proof['training_authority_granted'] is proof['optimizer_authority_granted'] is False
    assert f['called'] == ['cpu' if phase == 'cohort' else 'cuda']
    assert not f['output'].exists()


@pytest.mark.parametrize('bad', ('schema','model','target','execution','updates','owner','bridge',
    'selection','target_filter','recipe','protocol_digest','parent_protocol','parent_phase','spent','second_gate','flags',
    'existing','symlink','argv','hardware_env','unknown_env','legacy_env','old_fit_root','old_fit002_root','old_fit003_root'))
def test_exact_scope_rejects_before_kernel_proof(monkeypatch,tmp_path,bad):
    f = fixture(monkeypatch,tmp_path,bad=bad)
    with pytest.raises(RuntimeError): f['call']()
    assert not f['called']


@pytest.mark.parametrize('phase', PHASES)
def test_shell_phase_cli_and_profile_are_fixed(monkeypatch,tmp_path,phase):
    f = fixture(monkeypatch,tmp_path,phase)
    fragments = [x[2] for x in g._V3_DEVELOPMENT_SHELL_ADDITIONS]
    variables,functions,dispatch,_ = fragments
    functions = functions.replace('/home/andre2/GX1_RUNS',str(f['runs']))
    state = dict(REPO_ROOT=str(f['repo']),CANONICAL_TRAINER_PYTHON=str(f['repo'] / '.venv/bin/python'),
        JOB_CLASS='producer' if phase=='cohort' else 'trainer',MEM='12G' if phase=='cohort' else '20G',
        SWAP='512M',ATTENDED_SMOKE='false',CUDA_PRODUCER_GUARD='false',
        POWER_BENCHMARK_SCOPE_JSON='',POWER_BENCHMARK_SCOPE_SHA256='',TRAINER_MAX_WALL_SECONDS='',TRAINER_MODEL_MAX_WALL_SECONDS='',TRAINER_OUT_BUNDLE_DIR='')
    script='set -eu\n'+''.join(k+'='+shlex.quote(v)+'\n'for k,v in state.items())+variables+functions
    script+='validate_target_command() {\n'+dispatch+'return 88\n}\n'
    argv=[state['CANONICAL_TRAINER_PYTHON'],'-m','gx1.audit.native_provider_hlc_learning_v1',
        '--v3-development-plan-sha256',f['digest'],'--phase',phase]
    for wrong in (False,True):
        args=list(argv)
        if wrong: args[2]='gx1.audit.native_provider_hlc_replay_v1'
        tail='validate_target_command '+' '.join(map(shlex.quote,args))+'\nprintf "%s,%s,%s,%s\n" "$NATIVE_V3_DEVELOPMENT_PHASE" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_OUT_BUNDLE_DIR"\n'
        result=subprocess.run(['bash','-c',script+tail],cwd=f['repo'],capture_output=True,text=True)
        assert result.returncode==(75 if wrong else 0),result.stderr
        if not wrong:
            expected=[phase,*(['',''] if phase=='cohort' else ['5400','5370'] if phase=='fit' else ['1800','1770']),str(f['output']) if phase!='cohort' else '']
            assert result.stdout.strip()==','.join(expected)


@pytest.mark.parametrize('phase', ('warm','cold','fit'))
def test_gpu_guard_requests_only_actual_wrapper_exports(monkeypatch,tmp_path,phase):
    f=fixture(monkeypatch,tmp_path,phase,preflight=True)
    wrapper=(Path(__file__).resolve().parents[1]/'scripts/gx1_capped_run.sh').read_text()
    exported=set(re.findall(r'--setenv=([A-Z0-9_]+)=',wrapper))
    assert 'GX1_TRAINER_OUT_BUNDLE_DIR' not in exported
    requested=set()
    class ActualWrapperEnvironment(dict):
        def get(self,key,default=None):
            requested.add(key)
            return super().get(key,default)
    actual={key:value for key,value in os.environ.items() if not key.startswith('GX1_') or key in exported}
    monkeypatch.setattr(os,'environ',ActualWrapperEnvironment(actual))
    f['call']()
    assert requested <= exported
    assert f['called']==['cuda']


@pytest.mark.parametrize('phase,old_suffix', (('warm',''),('fit',''),('fit','_002'),('fit','_003')))
def test_revised_phase_has_no_legacy_plan_alias(monkeypatch,tmp_path,phase,old_suffix):
    f=fixture(monkeypatch,tmp_path,phase,preflight=True)
    assert f['path'].name==NAME+'_'+phase.upper()+('_002' if phase=='warm' else '_004')+'.json'
    old=f['path'].with_name(NAME+'_'+phase.upper()+old_suffix+'.json')
    f['path'].rename(old)
    with pytest.raises(RuntimeError,match='CANONICAL_JSON'): f['call']()
    assert not f['called']


@pytest.mark.parametrize('old_suffix', ('','_002','_003'))
def test_fresh_fit_namespace_preserves_consumed_old_fit(monkeypatch,tmp_path,old_suffix):
    f=fixture(monkeypatch,tmp_path,'fit',preflight=True)
    assert f['output'].name==NAME+'_fit_004'
    old_output=f['runs']/(NAME+'_fit'+old_suffix)
    old_output.mkdir()
    old_claim=old_output/'CLAIM.json'
    old_claim.write_text('{"status":"CONSUMED_OLD_FIT"}\n')
    old_plan=f['path'].with_name(NAME+'_FIT'+old_suffix+'.json')
    value=json.loads(f['path'].read_text())
    value['output_root']=str(old_output)
    old_plan.write_text(json.dumps(value,sort_keys=True))
    before={p:p.read_bytes()for p in (old_claim,old_plan)}
    proof=f['call']()
    assert proof['output_root']==str(f['output'])
    assert not f['output'].exists()
    assert all(p.read_bytes()==raw for p,raw in before.items())


def _actual_owners():
    repo=Path(g.__file__).resolve().parents[2]
    return {n:hashlib.sha256(subprocess.check_output(['git','show',g._V3_DEVELOPMENT_PARENT+':'+n],cwd=repo)).hexdigest()for n in OWNERS}


def test_bridge_proves_exact_addition_and_unchanged_predecessor():
    proof=g.native_provider_hlc_v3_development_source_bridge(_actual_owners())
    assert proof['checked_guard_owners']==2 and proof['old_guard_routes_preserved']
    assert proof['old_numeric_resource_environment_function_unchanged']


@pytest.mark.parametrize('bad',('legacy_resource','new_shell_limit','extra_statement','producer_hash'))
def test_bridge_rejects_unproven_source_delta(monkeypatch,bad):
    owners=_actual_owners()
    if bad=='producer_hash': owners[OWNERS[0]]='0'*64
    else:
        original=Path.read_text;repo=Path(g.__file__).resolve().parents[2]
        def changed(path,*a,**kw):
            text=original(path,*a,**kw)
            if path==repo/OWNERS[0] and bad=='legacy_resource': text=text.replace('_MAX_MEMORY_BYTES = 20 * 1024**3','_MAX_MEMORY_BYTES = 21 * 1024**3',1)
            if path==repo/OWNERS[0] and bad=='extra_statement': text=text.replace('\n# END GX1 V3 DEVELOPMENT GUARDS 20261127 001','\nUNEXPECTED = True\n# END GX1 V3 DEVELOPMENT GUARDS 20261127 001',1)
            if path==repo/OWNERS[1] and bad=='new_shell_limit':
                fragment=g._V3_DEVELOPMENT_SHELL_ADDITIONS[1][2]
                text=text.replace(fragment,fragment.replace('TRAINER_GPU_MAX_POWER_DRAW_W=250','TRAINER_GPU_MAX_POWER_DRAW_W=251'),1)
            return text
        monkeypatch.setattr(Path,'read_text',changed)
    with pytest.raises(RuntimeError):g.native_provider_hlc_v3_development_source_bridge(owners)


RECOVERY_NAME='NATIVE_ALL_TF_PROVIDER_HLC_V3_FIT004_RECOVERY_20261128_002'
RECOVERY_LANES=('reference','cold','recovery')


def recovery_fixture(monkeypatch,tmp_path,lane='recovery',preflight=False,bad=None):
    f=fixture(monkeypatch,tmp_path,'fit' if lane=='recovery' else 'cold' if lane=='cold' else 'warm',preflight)
    repo,runs=f['repo'],f['runs'];old=json.loads(f['path'].read_text())
    original=copy.deepcopy(old)
    original.update(phase='fit',output_root=str(runs/(NAME+'_fit_004')),limits={'optimizer_steps':432})
    original_ref=dict(path=str(repo/'configs/research'/(NAME+'_FIT_004.json')),
        sha256='ebf7222c9fb403cf0adc26738cc682c088035dc2b2dc2fc5aff603a741660437')
    receipt_ref=dict(path=str(runs/(NAME+'_fit_004')/'CURSOR001.json'),
        sha256=g._V3_RECOVERY_CHECKPOINT_RECEIPT_SHA256)
    receipt=dict(original_plan=original_ref,execution_plan=original_ref,logical_cursor=1,optimizer_updates=1)
    trajectory={'original_trajectory':'synthetic_original_only'}
    protocol=dict(schema_version='gx1_native_v3_exact_recovery_protocol_v1',experiment_id=RECOVERY_NAME,
        original_plan=original_ref,checkpoint_receipt=receipt_ref,original_protocol=original['protocol'],
        original_trajectory=trajectory)
    if bad=='protocol':protocol['original_protocol']={}
    if bad=='trajectory':protocol['original_trajectory']={}
    proto=repo/'configs/research'/(RECOVERY_NAME.replace('_20261128_002','_PROTOCOL_20261128_002')+'.json')
    proto.write_text(json.dumps(protocol));proto_ref=dict(path=str(proto),sha256=hashlib.sha256(proto.read_bytes()).hexdigest())
    bridge=dict(producer_commit=g._V3_RECOVERY_PARENT,
        owners={n:dict(original_sha256='f'*64,current_sha256=old['source_owners'][n])for n in OWNERS})
    output=runs/(RECOVERY_NAME+'_'+lane)
    plan=dict(schema_version='gx1_native_all_tf_provider_hlc_v3_development_recovery_plan_v1',
        lane=lane,stage=(RECOVERY_NAME+'_'+lane).lower(),source_commit='a'*40,output_root=str(output),
        execution=g._native_v3_recovery_execution(lane),original_plan=original_ref,checkpoint_receipt=receipt_ref,
        protocol=proto_ref,recipe=original['recipe'],model_contract=original['model_contract'],backbone_gradient_policy=None,
        metadata={'original':original,'receipt':receipt,'trajectory':trajectory},
        limits={'optimizer_steps':dict(reference=2,cold=1,recovery=431)[lane]},
        source_owners=old['source_owners'],guard_source_bridge=bridge)
    if bad=='schema':plan['schema_version']='old'
    if bad=='execution':plan['execution']['GPU_draw_stop_W']+=1
    if bad=='steps':plan['limits']['optimizer_steps']+=1
    if bad=='original_ref':plan['original_plan']={**original_ref,'sha256':'0'*64}
    if bad=='receipt_ref':plan['checkpoint_receipt']={**receipt_ref,'sha256':'0'*64}
    if bad=='metadata_original':plan['metadata']['original']={}
    if bad=='metadata_receipt':plan['metadata']['receipt']={}
    if bad=='recipe':plan['recipe']={**original['recipe'],'passes':3}
    if bad=='model':plan['model_contract']={}
    if bad=='gradient_policy':plan['backbone_gradient_policy']='M5_primary'
    if bad=='owner':plan['source_owners']={**plan['source_owners'],OWNERS[0]:'0'*64}
    if bad=='bridge':plan['guard_source_bridge']={**bridge,'unsafe':True}
    path=repo/'configs/research'/(RECOVERY_NAME+'_'+lane.upper()+'.json')
    path.write_text(json.dumps(plan));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    gate=dict(status='ADMITTED_EXACT_V3_RECOVERY_'+lane.upper(),plan=dict(path=str(path),sha256=digest),
        execution_authorized=True,remaining_invocations=1,relaunch_allowed=False,training_authorized=True,test_authorized=False)
    parent=dict(original_plan=original_ref,checkpoint_receipt=receipt_ref,protocol=proto_ref,
        main_fit_authorized=False,test_authorized=False,spending_authorized=False,
        execution_authorized=not preflight,active_lane=None if preflight else lane,
        status='PREPARED' if preflight else gate['status'],lane_gates={}if preflight else{lane:gate})
    policy=json.loads((repo/'NEXT_RUN_POLICY.json').read_text())
    old_parent=policy['native_v3_bounded_development_test_20261127_001']
    old_parent.update(execution_authorized=False,active_phase=None,phase_gates={'fit':{'execution_authorized':False,'remaining_invocations':0}})
    policy['native_v3_fit004_exact_recovery_20261128_002']=parent
    if bad=='old_live':old_parent['execution_authorized']=True
    if bad=='spent':gate['remaining_invocations']=0
    if bad=='second_lane':parent['lane_gates']['reference'if lane!='reference'else'cold']=gate
    if bad=='active_lane':parent['active_lane']='wrong'
    if bad=='parent_protocol':parent['protocol']={}
    if bad=='flags':policy['native_launch_authorized']=True
    (repo/'NEXT_RUN_POLICY.json').write_text(json.dumps(policy))
    if bad=='existing':output.mkdir()
    if bad=='symlink':
        original_path=path.with_suffix('.original');path.rename(original_path);path.symlink_to(original_path)
    read=g._native_v3_development_read
    def fixture_read(path,digest=None):
        if str(path)==original_ref['path']:
            assert digest==original_ref['sha256'];return original
        if str(path)==receipt_ref['path']:
            assert digest==receipt_ref['sha256'];return receipt
        return read(path,digest)
    monkeypatch.setattr(g,'_native_v3_development_read',fixture_read)
    monkeypatch.setattr(g,'native_provider_hlc_v3_development_recovery_source_bridge',lambda*a,**k:bridge)
    monkeypatch.setattr(sys,'argv',[str(repo/'gx1/audit/native_provider_hlc_learning_v1.py'),
        '--v3-development-recovery-plan-sha256',digest,'--lane',lane]+(['--preflight']if preflight else[])+(['--unsafe']if bad=='argv'else[]))
    for k in list(os.environ):
        if k.startswith('GX1_NATIVE_'):monkeypatch.delenv(k)
    env=dict(GX1_NATIVE_V3_RECOVERY='true',GX1_NATIVE_V3_RECOVERY_PLAN_SHA256=digest,GX1_NATIVE_V3_RECOVERY_LANE=lane,
        GX1_NATIVE_V3_DEVELOPMENT='false')
    if bad=='hardware_env':env['GX1_TRAINER_GPU_MAX_POWER_DRAW_W']='251'
    if bad=='unknown_env':env['GX1_NATIVE_V3_RECOVERY_UNSAFE']='false'
    if bad=='legacy_env':env['GX1_NATIVE_V3_DEVELOPMENT']='true'
    for k,v in env.items():monkeypatch.setenv(k,v)
    return dict(f,plan=plan,path=path,digest=digest,output=output,lane=lane,
        call=lambda:g.require_native_all_tf_provider_hlc_v3_development_recovery_execution(plan_sha256=digest,lane=lane))


@pytest.mark.parametrize('lane',RECOVERY_LANES)
@pytest.mark.parametrize('preflight',(False,True))
def test_exact_original_recovery_resource_guard(monkeypatch,tmp_path,lane,preflight):
    f=recovery_fixture(monkeypatch,tmp_path,lane,preflight);proof=f['call']()
    assert proof['lane']==lane and proof['plan']['sha256']==f['digest']
    assert proof['resource_proof_only']is True
    assert proof['training_authority_granted']is proof['optimizer_authority_granted']is False
    assert f['called']==['cuda'] and not f['output'].exists()


@pytest.mark.parametrize('bad',('schema','execution','steps','original_ref','receipt_ref','metadata_original',
    'metadata_receipt','recipe','model','gradient_policy','owner','bridge','protocol','trajectory','old_live',
    'spent','second_lane','active_lane','parent_protocol','flags','existing','symlink','argv',
    'hardware_env','unknown_env','legacy_env'))
def test_recovery_rejects_before_kernel_or_claim(monkeypatch,tmp_path,bad):
    f=recovery_fixture(monkeypatch,tmp_path,bad=bad)
    with pytest.raises(RuntimeError):f['call']()
    assert not f['called'] and not(f['output']/'CLAIM.json').exists()


@pytest.mark.parametrize('lane',RECOVERY_LANES)
def test_recovery_shell_reuses_exact_profiles_and_rejects_consumed_cli(monkeypatch,tmp_path,lane):
    f=recovery_fixture(monkeypatch,tmp_path,lane)
    variables,functions,dispatch,_=[v[2]for v in g._V3_RECOVERY_SHELL_ADDITIONS]
    functions=functions.replace('/home/andre2/GX1_RUNS',str(f['runs']))
    state=dict(REPO_ROOT=str(f['repo']),CANONICAL_TRAINER_PYTHON=str(f['repo']/'.venv/bin/python'),
        JOB_CLASS='trainer',MEM='20G',SWAP='512M',ATTENDED_SMOKE='false',CUDA_PRODUCER_GUARD='false',
        POWER_BENCHMARK_SCOPE_JSON='',POWER_BENCHMARK_SCOPE_SHA256='')
    script='set -eu\n'+''.join(k+'='+shlex.quote(v)+'\n'for k,v in state.items())+variables+functions
    script+='validate_target_command() {\n'+dispatch+'return 88\n}\n'
    argv=[state['CANONICAL_TRAINER_PYTHON'],'-m','gx1.audit.native_provider_hlc_learning_v1',
        '--v3-development-recovery-plan-sha256',f['digest'],'--lane',lane]
    for wrong in (False,True):
        args=list(argv)
        if wrong:args[3]='--v3-development-plan-sha256'
        tail='validate_target_command '+' '.join(map(shlex.quote,args))+'\nprintf "%s,%s,%s,%s\n" "$NATIVE_V3_RECOVERY_LANE" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_OUT_BUNDLE_DIR"\n'
        result=subprocess.run(['bash','-c',script+tail],cwd=f['repo'],capture_output=True,text=True)
        assert result.returncode==(88 if wrong else 0),result.stderr
        if not wrong:
            execution=g._native_v3_recovery_execution(lane)
            assert result.stdout.strip()==','.join(map(str,[lane,execution['outer_seconds'],execution['internal_seconds'],f['output']]))


def _actual_recovery_owners():
    repo=Path(g.__file__).resolve().parents[2]
    return {n:hashlib.sha256(subprocess.check_output(['git','show',g._V3_RECOVERY_PARENT+':'+n],cwd=repo)).hexdigest()for n in OWNERS}


def test_recovery_bridge_preserves_all_original_guard_bytes():
    proof=g.native_provider_hlc_v3_development_recovery_source_bridge(_actual_recovery_owners())
    assert proof['old_guard_routes_preserved']and proof['old_numeric_resource_environment_function_unchanged']
    assert proof['new_recovery_controlflow_requires_scoped_review']is True
    for n,v in proof['owners'].items():
        assert v['current_sha256']==hashlib.sha256((Path(g.__file__).resolve().parents[2]/n).read_bytes()).hexdigest()


@pytest.mark.parametrize('bad',('legacy_resource','recovery_shell_limit','producer_hash'))
def test_recovery_bridge_rejects_unreviewed_changes(monkeypatch,bad):
    owners=_actual_recovery_owners()
    if bad=='producer_hash':owners[OWNERS[0]]='0'*64
    else:
        original=Path.read_text;repo=Path(g.__file__).resolve().parents[2]
        def changed(path,*a,**kw):
            text=original(path,*a,**kw)
            if path==repo/OWNERS[0]and bad=='legacy_resource':
                text=text.replace('_MAX_MEMORY_BYTES = 20 * 1024**3','_MAX_MEMORY_BYTES = 21 * 1024**3',1)
            if path==repo/OWNERS[1]and bad=='recovery_shell_limit':
                fragment=g._V3_RECOVERY_SHELL_ADDITIONS[1][2]
                text=text.replace(fragment,fragment.replace('TRAINER_GPU_MAX_POWER_DRAW_W=250','TRAINER_GPU_MAX_POWER_DRAW_W=251'),1)
            return text
        monkeypatch.setattr(Path,'read_text',changed)
    with pytest.raises(RuntimeError):g.native_provider_hlc_v3_development_recovery_source_bridge(owners)


@pytest.mark.parametrize('old_suffix',('','_002','_003','_004'))
def test_recovery_rejects_every_consumed_fit_namespace(monkeypatch,tmp_path,old_suffix):
    f=recovery_fixture(monkeypatch,tmp_path,preflight=True)
    old_plan=f['repo']/'configs/research'/(NAME+'_FIT'+old_suffix+'.json')
    f['path'].rename(old_plan)
    with pytest.raises(RuntimeError,match='CANONICAL_JSON'):f['call']()
    assert not f['called']


@pytest.mark.parametrize('lane',RECOVERY_LANES)
def test_recovery_kernel_guard_requests_only_exported_environment(monkeypatch,tmp_path,lane):
    f=recovery_fixture(monkeypatch,tmp_path,lane,preflight=True)
    wrapper=(Path(__file__).resolve().parents[1]/'scripts/gx1_capped_run.sh').read_text()
    exported=set(re.findall(r'--setenv=([A-Z0-9_]+)=',wrapper))
    requested=set()
    class ActualWrapperEnvironment(dict):
        def get(self,key,default=None):
            requested.add(key);return super().get(key,default)
    actual={key:value for key,value in os.environ.items()if not key.startswith('GX1_')or key in exported}
    monkeypatch.setattr(os,'environ',ActualWrapperEnvironment(actual))
    f['call']()
    assert requested<=exported and f['called']==['cuda']


@pytest.mark.parametrize('location',('active','inactive','preflight','old_development'))
def test_recovery_rejects_boolean_invocation_counts(monkeypatch,tmp_path,location):
    f=recovery_fixture(monkeypatch,tmp_path,preflight=location=='preflight')
    policy_path=f['repo']/'NEXT_RUN_POLICY.json';policy=json.loads(policy_path.read_text())
    parent=policy['native_v3_fit004_exact_recovery_20261128_002']
    if location=='active':parent['lane_gates'][f['lane']]['remaining_invocations']=True
    elif location=='inactive':parent['lane_gates']['reference']={'execution_authorized':False,'remaining_invocations':False}
    elif location=='preflight':parent['lane_gates'][f['lane']]={'execution_authorized':False,'remaining_invocations':False}
    else:policy['native_v3_bounded_development_test_20261127_001']['phase_gates']['fit']['remaining_invocations']=False
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(RuntimeError):f['call']()
    assert not f['called']


@pytest.mark.parametrize('lane',RECOVERY_LANES)
def test_recovery002_rejects_consumed001_plan_namespace(monkeypatch,tmp_path,lane):
    f=recovery_fixture(monkeypatch,tmp_path,lane,preflight=True)
    old=f['path'].with_name(f['path'].name.replace('_20261128_002_','_20261128_001_'))
    f['path'].rename(old)
    before=old.read_bytes()
    with pytest.raises(RuntimeError,match='CANONICAL_JSON'):f['call']()
    assert not f['called']and old.read_bytes()==before
    variables,functions,_,_=[v[2]for v in g._V3_RECOVERY_SHELL_ADDITIONS]
    functions=functions.replace('/home/andre2/GX1_RUNS',str(f['runs']))
    state=dict(REPO_ROOT=str(f['repo']),CANONICAL_TRAINER_PYTHON=str(f['repo']/'.venv/bin/python'))
    script='set -eu\n'+''.join(k+'='+shlex.quote(v)+'\n'for k,v in state.items())+variables+functions
    argv=[state['CANONICAL_TRAINER_PYTHON'],'-m','gx1.audit.native_provider_hlc_learning_v1',
        '--v3-development-recovery-plan-sha256',f['digest'],'--lane',lane,'--preflight']
    result=subprocess.run(['bash','-c',script+'matches_bound_native_v3_recovery '+' '.join(map(shlex.quote,argv))],
        cwd=f['repo'],capture_output=True,text=True)
    assert result.returncode==1 and old.read_bytes()==before


@pytest.mark.parametrize('lane',RECOVERY_LANES)
def test_recovery002_rejects_consumed001_output_namespace(monkeypatch,tmp_path,lane):
    f=recovery_fixture(monkeypatch,tmp_path,lane,preflight=True)
    old_output=f['output'].with_name(f['output'].name.replace('_20261128_002_','_20261128_001_'))
    old_output.mkdir();old_claim=old_output/'CLAIM.json';old_claim.write_text('{"status":"CONSUMED_001"}\n')
    before=old_claim.read_bytes()
    f['plan']['output_root']=str(old_output);f['path'].write_text(json.dumps(f['plan']))
    digest=hashlib.sha256(f['path'].read_bytes()).hexdigest()
    argv=list(sys.argv);argv[2]=digest;monkeypatch.setattr(sys,'argv',argv)
    monkeypatch.setenv('GX1_NATIVE_V3_RECOVERY_PLAN_SHA256',digest)
    with pytest.raises(RuntimeError,match='RECOVERY_PLAN_IDENTITY'):
        g.require_native_all_tf_provider_hlc_v3_development_recovery_execution(plan_sha256=digest,lane=lane)
    assert not f['called']and old_claim.read_bytes()==before


DIRECTION_NAME='NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_GRADIENT_20261128_001'


def direction_gradient_fixture(monkeypatch,tmp_path,preflight=False,bad=None):
    f=recovery_fixture(monkeypatch,tmp_path,'reference',preflight=True)
    repo,runs=f['repo'],f['runs'];original=f['plan']['metadata']['original']
    original_ref=f['plan']['original_plan']
    final_root=runs/'NATIVE_ALL_TF_PROVIDER_HLC_V3_FIT004_RECOVERY_20261128_002_recovery'
    receipt=dict(logical_cursor=432,optimizer_updates=432,producer_lane='recovery',original_plan=original_ref,
        checkpoint={'path':str(final_root/'FINAL.pt'),'sha256':'c'*64})
    receipt_ref=dict(path=str(final_root/'FINAL.json'),sha256='a5ac56d7297a75a8157daf507717c51aeffa6e4c61999f9d26711933811a364f')
    design_ref=dict(path=str(runs/'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'/'V3_DIRECTION_GRADIENT_DESIGN_20261128_001.json'),
        sha256='86c1b9675b29ad2f058fb9424afbd07f79ed796ab45a78fb360b9eb670f56e75')
    bridge=dict(producer_commit=g._V3_DIRECTION_GRADIENT_PARENT,
        owners={n:dict(original_sha256='f'*64,current_sha256=f['plan']['source_owners'][n])for n in OWNERS})
    limits={k:0 for k in original['limits']}
    limits.update(model_constructions=1,optimizer_constructions=1,feature_cache_payload_loads=1,
        target_archive_loads=1,checkpoint_loads=1,model_restores=1,model_forwards=3,
        backbone_forwards=3,readout_forwards=3,autograd_calls=30)
    output=runs/DIRECTION_NAME
    plan=dict(schema_version='gx1_native_v3_direction_gradient_plan_v1',stage=DIRECTION_NAME.lower(),
        source_commit='a'*40,output_root=str(output),execution=g._native_v3_development_execution('warm'),
        original_plan=original_ref,checkpoint_receipt=receipt_ref,design=design_ref,
        metadata={'original':original,'receipt':receipt},model_contract=original['model_contract'],
        inputs={'final_checkpoint_receipt':receipt_ref,'final_checkpoint':receipt['checkpoint']},
        limits=limits,panel_indices=[0,107,214],source_owners=f['plan']['source_owners'],guard_source_bridge=bridge)
    if bad=='schema':plan['schema_version']='old'
    if bad=='root':plan['output_root']=str(output).lower()
    if bad=='execution':plan['execution']['GPU_draw_stop_W']=251
    if bad=='design':plan['design']={**design_ref,'sha256':'0'*64}
    if bad=='c1':plan['checkpoint_receipt']=f['plan']['checkpoint_receipt']
    if bad=='model':plan['model_contract']={}
    if bad=='metadata':plan['metadata']['receipt']={}
    if bad=='checkpoint':plan['inputs']['final_checkpoint']={}
    if bad=='owner':plan['source_owners']={**plan['source_owners'],OWNERS[0]:'0'*64}
    if bad=='bridge':plan['guard_source_bridge']={**bridge,'extra':True}
    if bad in ('optimizer_steps','backward_calls','checkpoint_writes','normalization_fits','prior_fits'):
        limits[bad]=1
    if bad in ('model_forwards','autograd_calls'):limits[bad]+=1
    if bad=='bool_budget':limits['model_constructions']=True
    if bad=='panels':plan['panel_indices']=[0,107,215]
    path=repo/'configs/research'/(DIRECTION_NAME+'.json')
    path.write_text(json.dumps(plan));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    policy_path=repo/'NEXT_RUN_POLICY.json';policy=json.loads(policy_path.read_text())
    parent=dict(status='SOURCE_PREPARATION_ONLY'if preflight else'ADMITTED_EXACT_V3_DIRECTION_GRADIENT',
        plan=dict(path=str(path),sha256=digest),design=design_ref,execution_authorized=not preflight,
        remaining_invocations=0 if preflight else 1,relaunch_allowed=False,training_authorized=False,
        test_authorized=False,spending_authorized=False)
    policy[DIRECTION_NAME.lower()]=parent
    policy['native_v3_fit004_exact_recovery_20261128_001']={'execution_authorized':False,'lane_gates':{}}
    if bad=='bool_gate':parent['remaining_invocations']=not preflight
    if bad=='spent':parent['remaining_invocations']=0
    if bad=='training':policy['training_enabled']=True
    if bad=='parent_training':parent['training_authorized']=True
    if bad=='old_route':policy['native_v3_fit004_exact_recovery_20261128_002']['execution_authorized']=True
    if bad=='parent_plan':parent['plan']={}
    policy_path.write_text(json.dumps(policy))
    if bad=='existing':output.mkdir()
    if bad=='symlink':
        old=path.with_suffix('.original');path.rename(old);path.symlink_to(old)
    read=g._native_v3_development_read
    def fixture_read(path,digest=None):
        if str(path)==receipt_ref['path']:
            assert digest==receipt_ref['sha256'];return receipt
        if str(path)==design_ref['path']:
            assert digest==design_ref['sha256'];return {}
        return read(path,digest)
    monkeypatch.setattr(g,'_native_v3_development_read',fixture_read)
    monkeypatch.setattr(g,'native_v3_direction_gradient_source_bridge',lambda*a,**k:bridge)
    argv=[str(repo/'gx1/audit/native_provider_hlc_learning_v1.py'),'--v3-direction-gradient-plan-sha256',digest]
    if preflight:argv+=['--preflight']
    if bad=='argv':argv+=['--lane','reference']
    monkeypatch.setattr(sys,'argv',argv)
    for k in list(os.environ):
        if k.startswith('GX1_NATIVE_'):monkeypatch.delenv(k)
    env=dict(GX1_NATIVE_V3_DIRECTION_GRADIENT='true',GX1_NATIVE_V3_DIRECTION_GRADIENT_PLAN_SHA256=digest)
    if bad=='hardware_env':env['GX1_TRAINER_GPU_MAX_POWER_DRAW_W']='251'
    if bad=='unknown_env':env['GX1_NATIVE_V3_DIRECTION_GRADIENT_TRAINING']='false'
    if bad=='legacy_env':env['GX1_NATIVE_V3_RECOVERY']='true'
    for k,v in env.items():monkeypatch.setenv(k,v)
    return dict(f,path=path,digest=digest,plan=plan,output=output,
        call=lambda:g.require_native_v3_direction_gradient_execution(plan_sha256=digest))


@pytest.mark.parametrize('preflight',(False,True))
def test_direction_gradient_final_only_without_training_authority(monkeypatch,tmp_path,preflight):
    f=direction_gradient_fixture(monkeypatch,tmp_path,preflight);proof=f['call']()
    assert proof['preflight']is preflight and proof['kind']=='provider_hlc_v3_direction_gradient'
    assert proof['training_authority_granted']is proof['optimizer_authority_granted']is False
    assert f['called']==['cuda']and not f['output'].exists()


@pytest.mark.parametrize('bad',('schema','root','execution','design','c1','model','metadata','checkpoint',
    'owner','bridge','optimizer_steps','backward_calls','checkpoint_writes','normalization_fits','prior_fits',
    'model_forwards','autograd_calls','bool_budget','panels','bool_gate','spent','training','parent_training',
    'old_route','parent_plan','existing','symlink','argv','hardware_env','unknown_env','legacy_env'))
def test_direction_gradient_rejects_update_or_mismatched_authority(monkeypatch,tmp_path,bad):
    f=direction_gradient_fixture(monkeypatch,tmp_path,bad=bad)
    with pytest.raises(RuntimeError):f['call']()
    assert not f['called']and not(f['output']/'CLAIM.json').exists()


def test_direction_gradient_shell_exact_cli_and_unchanged_profile(monkeypatch,tmp_path):
    f=direction_gradient_fixture(monkeypatch,tmp_path)
    variables,functions,dispatch,_=[v[2]for v in g._V3_DIRECTION_GRADIENT_SHELL_ADDITIONS]
    functions=functions.replace('/home/andre2/GX1_RUNS',str(f['runs']))
    state=dict(REPO_ROOT=str(f['repo']),CANONICAL_TRAINER_PYTHON=str(f['repo']/'.venv/bin/python'),
        JOB_CLASS='trainer',MEM='20G',SWAP='512M',ATTENDED_SMOKE='false',CUDA_PRODUCER_GUARD='false',
        POWER_BENCHMARK_SCOPE_JSON='',POWER_BENCHMARK_SCOPE_SHA256='')
    script='set -eu\n'+''.join(k+'='+shlex.quote(v)+'\n'for k,v in state.items())+variables+functions
    script+='validate_target_command() {\n'+dispatch+'return 88\n}\n'
    argv=[state['CANONICAL_TRAINER_PYTHON'],'-m','gx1.audit.native_provider_hlc_learning_v1',
        '--v3-direction-gradient-plan-sha256',f['digest']]
    for extra in ([],['--preflight'],['--lane','reference']):
        tail='validate_target_command '+' '.join(map(shlex.quote,argv+extra))
        tail+='\nprintf "%s,%s,%s,%s\n" "$NATIVE_V3_DIRECTION_GRADIENT" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_OUT_BUNDLE_DIR"\n'
        result=subprocess.run(['bash','-c',script+tail],cwd=f['repo'],capture_output=True,text=True)
        assert result.returncode==(75 if extra==['--lane','reference']else 0),result.stderr
        if result.returncode==0:assert result.stdout.strip()=='true,1800,1770,'+str(f['output'])


def test_direction_gradient_requests_only_actual_wrapper_exports(monkeypatch,tmp_path):
    f=direction_gradient_fixture(monkeypatch,tmp_path,preflight=True)
    wrapper=(Path(__file__).resolve().parents[1]/'scripts/gx1_capped_run.sh').read_text()
    exported=set(re.findall(r'--setenv=([A-Z0-9_]+)=',wrapper));requested=set()
    class ActualWrapperEnvironment(dict):
        def get(self,key,default=None):
            requested.add(key);return super().get(key,default)
    actual={k:v for k,v in os.environ.items()if not k.startswith('GX1_')or k in exported}
    monkeypatch.setattr(os,'environ',ActualWrapperEnvironment(actual));f['call']()
    assert requested<=exported and f['called']==['cuda']


def _actual_direction_gradient_owners():
    repo=Path(g.__file__).resolve().parents[2]
    return {n:hashlib.sha256(subprocess.check_output(['git','show',g._V3_DIRECTION_GRADIENT_PARENT+':'+n],cwd=repo)).hexdigest()for n in OWNERS}


def test_direction_gradient_bridge_preserves_all_previous_bytes():
    proof=g.native_v3_direction_gradient_source_bridge(_actual_direction_gradient_owners())
    assert proof['old_guard_routes_preserved']and proof['old_numeric_resource_environment_function_unchanged']
    assert proof['new_diagnostic_controlflow_requires_scoped_review']is True


@pytest.mark.parametrize('bad',('legacy_resource','new_shell_limit','producer_hash'))
def test_direction_gradient_bridge_rejects_unreviewed_delta(monkeypatch,bad):
    owners=_actual_direction_gradient_owners()
    if bad=='producer_hash':owners[OWNERS[0]]='0'*64
    else:
        original=Path.read_text;repo=Path(g.__file__).resolve().parents[2]
        def changed(path,*a,**kw):
            text=original(path,*a,**kw)
            if path==repo/OWNERS[0]and bad=='legacy_resource':
                text=text.replace('_MAX_MEMORY_BYTES = 20 * 1024**3','_MAX_MEMORY_BYTES = 21 * 1024**3',1)
            if path==repo/OWNERS[1]and bad=='new_shell_limit':
                fragment=g._V3_DIRECTION_GRADIENT_SHELL_ADDITIONS[1][2]
                text=text.replace(fragment,fragment.replace('TRAINER_GPU_MAX_POWER_DRAW_W=250','TRAINER_GPU_MAX_POWER_DRAW_W=251'),1)
            return text
        monkeypatch.setattr(Path,'read_text',changed)
    with pytest.raises(RuntimeError):g.native_v3_direction_gradient_source_bridge(owners)


def test_direction_gradient_preflight_rejects_false_boolean_count(monkeypatch,tmp_path):
    f=direction_gradient_fixture(monkeypatch,tmp_path,preflight=True,bad='bool_gate')
    with pytest.raises(RuntimeError,match='DIRECTION_GRADIENT_DIAGNOSTIC_SCOPE'):f['call']()
    assert not f['called']


BLOCK_ORDER_NAME='NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_BLOCK_ORDER_20261128_001'


def direction_block_order_fixture(monkeypatch,tmp_path,preflight=False,bad=None):
    f=direction_gradient_fixture(monkeypatch,tmp_path,preflight=True)
    repo,runs=f['repo'],f['runs'];original=f['plan']['metadata']['original']
    original_ref=f['plan']['original_plan']
    original['initial_reference']={'model_state_sha256':'b'*64,'optimizer_updates':0}
    original['limits']=dict(autograd_calls=9,backbone_forwards=678,backward_calls=432,
        checkpoint_loads=0,checkpoint_prepublication_loads=2,checkpoint_readbacks=2,
        checkpoint_receipt_writes=2,checkpoint_writes=2,diagnostic_skips=0,
        feature_cache_payload_loads=1,logit_archive_readbacks=3,logit_archive_writes=3,
        model_constructions=1,model_forwards=678,model_restores=0,normalization_fits=0,
        optimizer_constructions=1,optimizer_steps=432,prior_fits=0,readout_forwards=678,
        skipped_batches=0,target_archive_loads=1)
    canonical=dict(batch_source_rows_sha256=[hashlib.sha256(('source'+str(i)).encode()).hexdigest()for i in range(216)],
        batch_issue_ns_sha256=[hashlib.sha256(('issue'+str(i)).encode()).hexdigest()for i in range(216)])
    original['schedule']=canonical
    # Synthetic structural permutations; the leaf owns the actual private PCG64 generator.
    permutations=[list(reversed(range(216))),list(range(1,216))+[0]]
    ids=[i for v in permutations for i in v]
    schedule=dict(schema_version='gx1_native_v3_whole_batch_order_schedule_v1',canonical_schedule=canonical,
        permutations=permutations,logical_original_batch_ids=ids,
        ordered_batch_source_rows_sha256=[canonical['batch_source_rows_sha256'][i]for i in ids],
        ordered_batch_issue_ns_sha256=[canonical['batch_issue_ns_sha256'][i]for i in ids],
        rows=20727,batch_rows=96,passes=2,batches_per_pass=216,logical_batches=432,
        optimizer_updates=432,skipped_batches=0,update_flags=[True]*432)
    limits=dict(original['limits']);limits.update(model_forwards=597,backbone_forwards=597,readout_forwards=597,
        logit_archive_writes=2,logit_archive_readbacks=2)
    design_ref=dict(path=str(runs/'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'/'V3_DIRECTION_BLOCK_ORDER_DESIGN_20261128_001.json'),
        sha256='573e747a4ebea2f91a228a5853f4f16eb5a7411280dc62832d46a3e43f250942')
    bridge=dict(producer_commit=g._V3_DIRECTION_BLOCK_ORDER_PARENT,
        owners={n:dict(original_sha256='f'*64,current_sha256=f['plan']['source_owners'][n])for n in OWNERS})
    output=runs/BLOCK_ORDER_NAME
    plan=dict(schema_version='gx1_native_v3_direction_block_order_plan_v1',stage=BLOCK_ORDER_NAME.lower(),
        source_commit='a'*40,output_root=str(output),execution=g._native_v3_development_execution('fit'),
        original_plan=original_ref,design=design_ref,model_contract=original['model_contract'],
        initial_reference=original['initial_reference'],recipe=original['recipe'],schedule=schedule,
        metadata={'original':original,'trajectory':{'schedule':schedule}},limits=limits,
        checkpoint_cursors=[1,432],panel_indices=[0,107,214],
        source_owners=f['plan']['source_owners'],guard_source_bridge=bridge)
    if bad=='schema':plan['schema_version']='old'
    if bad=='execution':plan['execution']['GPU_draw_stop_W']+=1
    if bad=='design':plan['design']={**design_ref,'sha256':'0'*64}
    if bad=='model':plan['model_contract']={}
    if bad=='initializer':plan['initial_reference']={'checkpoint':'FINAL.pt'}
    if bad=='recipe':plan['recipe']={**original['recipe'],'seed':1}
    if bad=='canonical':schedule['canonical_schedule']={}
    if bad=='permutation_duplicate':permutations[0][1]=permutations[0][0]
    if bad=='permutation_length':permutations[0].pop()
    if bad=='permutation_valid_swap':permutations[0][0],permutations[0][1]=permutations[0][1],permutations[0][0]
    if bad=='permutation_bool':permutations[0][permutations[0].index(1)]=True
    if bad=='logical_id_bool':ids[ids.index(1)]=True
    if bad=='source_hash':schedule['ordered_batch_source_rows_sha256'][0]=canonical['batch_source_rows_sha256'][ids[0]^1]
    if bad=='issue_hash':schedule['ordered_batch_issue_ns_sha256'][0]=canonical['batch_issue_ns_sha256'][ids[0]^1]
    if bad=='update_flag':schedule['update_flags'][0]=1
    if bad=='trajectory':plan['metadata']['trajectory']={'schedule':canonical}
    if bad in ('optimizer_steps','model_forwards','autograd_calls','checkpoint_writes','checkpoint_loads'):
        limits[bad]+=1
    if bad=='budget_bool':limits['model_constructions']=True
    if bad=='checkpoint_cursor':plan['checkpoint_cursors']=[1,431]
    if bad=='panels':plan['panel_indices']=[0,107,215]
    if bad=='owner':plan['source_owners']={**plan['source_owners'],OWNERS[0]:'0'*64}
    if bad=='bridge':plan['guard_source_bridge']={**bridge,'extra':True}
    path=repo/'configs/research'/(BLOCK_ORDER_NAME+'.json');path.write_text(json.dumps(plan))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    policy_path=repo/'NEXT_RUN_POLICY.json';policy=json.loads(policy_path.read_text())
    for k in ('training_enabled','native_training_enabled'):policy[k]=not preflight
    parent=dict(status='SOURCE_PREPARATION_ONLY'if preflight else'ADMITTED_EXACT_V3_DIRECTION_BLOCK_ORDER',
        plan=dict(path=str(path),sha256=digest),design=design_ref,execution_authorized=not preflight,
        remaining_invocations=0 if preflight else 1,relaunch_allowed=False,training_authorized=not preflight,
        test_authorized=False,spending_authorized=False)
    policy[BLOCK_ORDER_NAME.lower()]=parent
    if bad=='bool_gate':parent['remaining_invocations']=not preflight
    if bad=='spent':parent['remaining_invocations']=0
    if bad=='flags':policy['full_epoch_training_allowed']=True
    if bad=='parent_training':parent['training_authorized']=preflight
    if bad=='diagnostic_active':policy[DIRECTION_NAME.lower()]['execution_authorized']=True
    if bad=='old_route':policy['native_v3_fit004_exact_recovery_20261128_002']['execution_authorized']=True
    if bad=='parent_plan':parent['plan']={}
    policy_path.write_text(json.dumps(policy))
    if bad=='existing':output.mkdir()
    if bad=='symlink':
        old=path.with_suffix('.original');path.rename(old);path.symlink_to(old)
    read=g._native_v3_development_read
    def fixture_read(path,digest=None):
        if str(path)==design_ref['path']:
            assert digest==design_ref['sha256'];return {}
        return read(path,digest)
    monkeypatch.setattr(g,'_native_v3_development_read',fixture_read)
    monkeypatch.setattr(g,'native_v3_direction_block_order_source_bridge',lambda*a,**k:bridge)
    argv=[str(repo/'gx1/audit/native_provider_hlc_learning_v1.py'),'--v3-direction-block-order-plan-sha256',digest]
    if preflight:argv+=['--preflight']
    if bad=='argv':argv+=['--lane','reference']
    monkeypatch.setattr(sys,'argv',argv)
    for k in list(os.environ):
        if k.startswith('GX1_NATIVE_'):monkeypatch.delenv(k)
    env=dict(GX1_NATIVE_V3_DIRECTION_BLOCK_ORDER='true',GX1_NATIVE_V3_DIRECTION_BLOCK_ORDER_PLAN_SHA256=digest,
        GX1_TRAINER_MAX_WALL_SECONDS='5400',GX1_TRAINER_MODEL_MAX_WALL_SECONDS='5370')
    if bad=='hardware_env':env['GX1_TRAINER_GPU_MAX_POWER_DRAW_W']='251'
    if bad=='unknown_env':env['GX1_NATIVE_V3_DIRECTION_BLOCK_ORDER_UNSAFE']='false'
    if bad=='legacy_env':env['GX1_NATIVE_V3_RECOVERY']='true'
    for k,v in env.items():monkeypatch.setenv(k,v)
    return dict(f,path=path,digest=digest,plan=plan,output=output,
        call=lambda:g.require_native_v3_direction_block_order_execution(plan_sha256=digest))


@pytest.mark.parametrize('preflight',(False,True))
def test_direction_block_order_exact_scoped_fit_guard(monkeypatch,tmp_path,preflight):
    f=direction_block_order_fixture(monkeypatch,tmp_path,preflight);proof=f['call']()
    assert proof['preflight']is preflight and proof['kind']=='provider_hlc_v3_direction_block_order'
    assert proof['training_authority_granted']is proof['optimizer_authority_granted']is False
    assert f['called']==['cuda']and not f['output'].exists()


@pytest.mark.parametrize('bad',('schema','execution','design','model','initializer','recipe','canonical',
    'permutation_duplicate','permutation_length','permutation_valid_swap','permutation_bool','logical_id_bool','source_hash','issue_hash',
    'update_flag','trajectory','optimizer_steps','model_forwards','autograd_calls','checkpoint_writes',
    'checkpoint_loads','budget_bool','checkpoint_cursor','panels','owner','bridge','bool_gate','spent','flags',
    'parent_training','diagnostic_active','old_route','parent_plan','existing','symlink','argv',
    'hardware_env','unknown_env','legacy_env'))
def test_direction_block_order_rejects_wrong_order_or_admission(monkeypatch,tmp_path,bad):
    f=direction_block_order_fixture(monkeypatch,tmp_path,bad=bad)
    with pytest.raises(RuntimeError):f['call']()
    assert not f['called']and not(f['output']/'CLAIM.json').exists()


def test_direction_block_order_preflight_rejects_false_boolean_count(monkeypatch,tmp_path):
    f=direction_block_order_fixture(monkeypatch,tmp_path,preflight=True,bad='bool_gate')
    with pytest.raises(RuntimeError,match='DIRECTION_BLOCK_ORDER_FIT_SCOPE'):f['call']()
    assert not f['called']


def test_direction_block_order_shell_exact_cli_and_fullfit_profile(monkeypatch,tmp_path):
    f=direction_block_order_fixture(monkeypatch,tmp_path)
    variables,functions,dispatch,_=[v[2]for v in g._V3_DIRECTION_BLOCK_ORDER_SHELL_ADDITIONS]
    functions=functions.replace('/home/andre2/GX1_RUNS',str(f['runs']))
    state=dict(REPO_ROOT=str(f['repo']),CANONICAL_TRAINER_PYTHON=str(f['repo']/'.venv/bin/python'),
        JOB_CLASS='trainer',MEM='20G',SWAP='512M',ATTENDED_SMOKE='false',CUDA_PRODUCER_GUARD='false',
        POWER_BENCHMARK_SCOPE_JSON='',POWER_BENCHMARK_SCOPE_SHA256='')
    script='set -eu\n'+''.join(k+'='+shlex.quote(v)+'\n'for k,v in state.items())+variables+functions
    script+='validate_target_command() {\n'+dispatch+'return 88\n}\n'
    argv=[state['CANONICAL_TRAINER_PYTHON'],'-m','gx1.audit.native_provider_hlc_learning_v1',
        '--v3-direction-block-order-plan-sha256',f['digest']]
    for extra in ([],['--preflight'],['--lane','reference']):
        tail='validate_target_command '+' '.join(map(shlex.quote,argv+extra))
        tail+='\nprintf "%s,%s,%s,%s\n" "$NATIVE_V3_DIRECTION_BLOCK_ORDER" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_OUT_BUNDLE_DIR"\n'
        result=subprocess.run(['bash','-c',script+tail],cwd=f['repo'],capture_output=True,text=True)
        assert result.returncode==(75 if extra==['--lane','reference']else 0),result.stderr
        if result.returncode==0:assert result.stdout.strip()=='true,5400,5370,'+str(f['output'])


def test_direction_block_order_requests_only_wrapper_exports(monkeypatch,tmp_path):
    f=direction_block_order_fixture(monkeypatch,tmp_path,preflight=True)
    wrapper=(Path(__file__).resolve().parents[1]/'scripts/gx1_capped_run.sh').read_text()
    exported=set(re.findall(r'--setenv=([A-Z0-9_]+)=',wrapper));requested=set()
    class ActualWrapperEnvironment(dict):
        def get(self,key,default=None):
            requested.add(key);return super().get(key,default)
    actual={k:v for k,v in os.environ.items()if not k.startswith('GX1_')or k in exported}
    monkeypatch.setattr(os,'environ',ActualWrapperEnvironment(actual));f['call']()
    assert requested<=exported and f['called']==['cuda']


def _actual_direction_block_order_owners():
    repo=Path(g.__file__).resolve().parents[2]
    return {n:hashlib.sha256(subprocess.check_output(['git','show',g._V3_DIRECTION_BLOCK_ORDER_PARENT+':'+n],cwd=repo)).hexdigest()for n in OWNERS}


def test_direction_block_order_bridge_preserves_all_previous_bytes():
    proof=g.native_v3_direction_block_order_source_bridge(_actual_direction_block_order_owners())
    assert proof['old_guard_routes_preserved']and proof['old_numeric_resource_environment_function_unchanged']
    assert proof['new_order_controlflow_requires_scoped_review']is True


@pytest.mark.parametrize('bad',('legacy_resource','new_shell_limit','producer_hash'))
def test_direction_block_order_bridge_rejects_unreviewed_delta(monkeypatch,bad):
    owners=_actual_direction_block_order_owners()
    if bad=='producer_hash':owners[OWNERS[0]]='0'*64
    else:
        original=Path.read_text;repo=Path(g.__file__).resolve().parents[2]
        def changed(path,*a,**kw):
            text=original(path,*a,**kw)
            if path==repo/OWNERS[0]and bad=='legacy_resource':
                text=text.replace('_MAX_MEMORY_BYTES = 20 * 1024**3','_MAX_MEMORY_BYTES = 21 * 1024**3',1)
            if path==repo/OWNERS[1]and bad=='new_shell_limit':
                fragment=g._V3_DIRECTION_BLOCK_ORDER_SHELL_ADDITIONS[1][2]
                text=text.replace(fragment,fragment.replace('TRAINER_GPU_MAX_POWER_DRAW_W=250','TRAINER_GPU_MAX_POWER_DRAW_W=251'),1)
            return text
        monkeypatch.setattr(Path,'read_text',changed)
    with pytest.raises(RuntimeError):g.native_v3_direction_block_order_source_bridge(owners)


ORDER_RECOVERY_NAME='NATIVE_ALL_TF_PROVIDER_HLC_V3_BLOCK_ORDER_RECOVERY_20261128_001'


def block_order_recovery_fixture(monkeypatch,tmp_path,lane='recovery',preflight=False,bad=None):
    f=direction_block_order_fixture(monkeypatch,tmp_path,preflight=True)
    repo,runs=f['repo'],f['runs'];original=f['plan']
    original['schedule']['contract_sha256']='dba16db2b9c04c1fd988a79d3fd713b9719bbfc3514b76b38174b92ccfef19de'
    original['metadata']['trajectory']['contract_sha256']='214fceb7d316c33edd928936612fa0ad3fc2fd3f9843396cea35e41fcbd3a0b6'
    original_ref=dict(path=str(f['path']),sha256='15a59b3840d61d57d6e217967312a6b76e4e89d256ea902b619ec3fa8099a008')
    receipt_ref=dict(path=str(runs/BLOCK_ORDER_NAME/'CURSOR001.json'),
        sha256='9d6f3ec4908f2cd2c60f1a6c5e8c9980f59bd8b2933dd0500e50bf8b3548a7f7')
    receipt=dict(logical_cursor=1,optimizer_updates=1,producer_lane='block_order',
        original_plan=original_ref,execution_plan=original_ref,trajectory_sha256=original['metadata']['trajectory']['contract_sha256'])
    if bad=='producer_lane':receipt['producer_lane']='fit'
    design_ref=dict(path=str(runs/'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'/'V3_BLOCK_ORDER_EXACT_RECOVERY_DESIGN_20261128_001.json'),
        sha256='737dd510ca83958d1f1bb192ef22ca40e072c0aa9b5c757cc3787bb464ebb830')
    protocol=dict(schema_version='gx1_v3_block_order_recovery_protocol_v1',experiment_id=ORDER_RECOVERY_NAME,
        original_plan=original_ref,checkpoint_receipt=receipt_ref,
        original_trajectory=original['metadata']['trajectory'],original_schedule=original['schedule'])
    if bad=='protocol_schedule':protocol['original_schedule']={}
    if bad=='protocol_trajectory':protocol['original_trajectory']={}
    proto=repo/'configs/research'/(ORDER_RECOVERY_NAME.replace('_20261128_001','_PROTOCOL_20261128_001')+'.json')
    proto.write_text(json.dumps(protocol));protocol_ref=dict(path=str(proto),sha256=hashlib.sha256(proto.read_bytes()).hexdigest())
    limits=dict.fromkeys(original['limits'],0);updates=dict(reference=2,cold=1,recovery=431)[lane];fw=dict(reference=2,cold=1,recovery=596)[lane]
    limits.update(model_constructions=1,optimizer_constructions=1,feature_cache_payload_loads=1,
        target_archive_loads=1,model_forwards=fw,backbone_forwards=fw,readout_forwards=fw,
        optimizer_steps=updates,backward_calls=updates,checkpoint_writes=1,checkpoint_readbacks=1,
        checkpoint_receipt_writes=1,checkpoint_prepublication_loads=1)
    if lane!='reference':limits.update(checkpoint_loads=1,model_restores=1)
    if lane=='recovery':limits.update(autograd_calls=9,logit_archive_writes=2,logit_archive_readbacks=2)
    bridge=dict(producer_commit=g._V3_BLOCK_ORDER_RECOVERY_PARENT,
        owners={n:dict(original_sha256='f'*64,current_sha256=original['source_owners'][n])for n in OWNERS})
    output=runs/(ORDER_RECOVERY_NAME+'_'+lane)
    plan=dict(schema_version='gx1_native_v3_block_order_recovery_plan_v1',stage=ORDER_RECOVERY_NAME.lower(),
        lane=lane,source_commit='a'*40,output_root=str(output),execution=g._native_v3_recovery_execution(lane),
        original_plan=original_ref,checkpoint_receipt=receipt_ref,design=design_ref,protocol=protocol_ref,
        metadata={'original':original,'receipt':receipt,'trajectory':original['metadata']['trajectory']},
        schedule=original['schedule'],recipe=original['recipe'],model_contract=original['model_contract'],
        initial_reference=original['initial_reference'],limits=limits,
        source_owners=original['source_owners'],guard_source_bridge=bridge)
    if bad=='schema':plan['schema_version']='old'
    if bad=='execution':plan['execution']['GPU_draw_stop_W']+=1
    if bad=='chronological_original':plan['original_plan']=original['original_plan']
    if bad=='chronological_receipt':plan['checkpoint_receipt']={'path':str(runs/(NAME+'_fit_004')/'CURSOR001.json'),'sha256':'adcc8717242aa5bac0505332fbe360dc3578315bb49f5e6230ebefe15e871529'}
    if bad=='design':plan['design']={**design_ref,'sha256':'0'*64}
    if bad=='metadata_receipt':plan['metadata']['receipt']={}
    if bad=='trajectory':plan['metadata']['trajectory']={}
    if bad=='schedule':plan['schedule']={}
    if bad=='model':plan['model_contract']={}
    if bad=='initializer':plan['initial_reference']={'checkpoint':'technical_C2.pt'}
    if bad=='recipe':plan['recipe']={**original['recipe'],'seed':1}
    if bad in ('optimizer_steps','model_forwards','checkpoint_writes','checkpoint_loads'):
        limits[bad]+=1
    if bad=='budget_bool':limits['model_constructions']=True
    if bad=='owner':plan['source_owners']={**plan['source_owners'],OWNERS[0]:'0'*64}
    if bad=='bridge':plan['guard_source_bridge']={**bridge,'extra':True}
    path=repo/'configs/research'/(ORDER_RECOVERY_NAME+'_'+lane.upper()+'.json')
    path.write_text(json.dumps(plan));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    gate=dict(status='ADMITTED_EXACT_V3_BLOCK_ORDER_RECOVERY_'+lane.upper(),
        plan=dict(path=str(path),sha256=digest),execution_authorized=True,remaining_invocations=1,
        relaunch_allowed=False,training_authorized=True,test_authorized=False)
    parent=dict(original_plan=original_ref,checkpoint_receipt=receipt_ref,design=design_ref,protocol=protocol_ref,
        main_fit_authorized=False,test_authorized=False,spending_authorized=False,
        execution_authorized=not preflight,active_lane=None if preflight else lane,
        status='SOURCE_PREPARATION_ONLY'if preflight else gate['status'],lane_gates={}if preflight else{lane:gate})
    policy_path=repo/'NEXT_RUN_POLICY.json';policy=json.loads(policy_path.read_text())
    for k in ('training_enabled','native_training_enabled'):policy[k]=not preflight
    policy[ORDER_RECOVERY_NAME.lower()]=parent
    if bad=='bool_gate':gate['remaining_invocations']=True
    if bad=='spent':gate['remaining_invocations']=0
    if bad=='inactive_bool':
        parent['lane_gates']['reference'if lane!='reference'else'cold']={'execution_authorized':False,'remaining_invocations':False,
            'training_authorized':False,'test_authorized':False,'relaunch_allowed':False}
    if bad=='second_lane':parent['lane_gates']['reference'if lane!='reference'else'cold']=gate
    if bad=='parent_protocol':parent['protocol']={}
    if bad=='flags':policy['full_val_allowed']=True
    if bad=='old_fit':policy[BLOCK_ORDER_NAME.lower()]['execution_authorized']=True
    if bad=='diagnostic':policy[DIRECTION_NAME.lower()]['execution_authorized']=True
    policy_path.write_text(json.dumps(policy))
    if bad=='existing':output.mkdir()
    if bad=='symlink':
        old=path.with_suffix('.original');path.rename(old);path.symlink_to(old)
    read=g._native_v3_development_read
    def fixture_read(path,digest=None):
        if str(path)==original_ref['path']:
            assert digest==original_ref['sha256'];return original
        if str(path)==receipt_ref['path']:
            assert digest==receipt_ref['sha256'];return receipt
        if str(path)==design_ref['path']:
            assert digest==design_ref['sha256'];return {}
        return read(path,digest)
    monkeypatch.setattr(g,'_native_v3_development_read',fixture_read)
    monkeypatch.setattr(g,'native_v3_block_order_recovery_source_bridge',lambda*a,**k:bridge)
    argv=[str(repo/'gx1/audit/native_provider_hlc_learning_v1.py'),
        '--v3-block-order-recovery-plan-sha256',digest,'--lane',lane]
    if preflight:argv+=['--preflight']
    if bad=='argv':argv+=['--phase','fit']
    monkeypatch.setattr(sys,'argv',argv)
    for k in list(os.environ):
        if k.startswith('GX1_NATIVE_'):monkeypatch.delenv(k)
    env=dict(GX1_NATIVE_V3_BLOCK_ORDER_RECOVERY='true',GX1_NATIVE_V3_BLOCK_ORDER_RECOVERY_PLAN_SHA256=digest,
        GX1_NATIVE_V3_BLOCK_ORDER_RECOVERY_LANE=lane,GX1_TRAINER_MAX_WALL_SECONDS='5400'if lane=='recovery'else'1800',
        GX1_TRAINER_MODEL_MAX_WALL_SECONDS='5370'if lane=='recovery'else'1770')
    if bad=='hardware_env':env['GX1_TRAINER_GPU_MAX_POWER_DRAW_W']='251'
    if bad=='unknown_env':env['GX1_NATIVE_V3_BLOCK_ORDER_RECOVERY_UNSAFE']='false'
    if bad=='legacy_env':env['GX1_NATIVE_V3_DIRECTION_BLOCK_ORDER']='true'
    for k,v in env.items():monkeypatch.setenv(k,v)
    return dict(f,path=path,digest=digest,plan=plan,output=output,lane=lane,
        call=lambda:g.require_native_v3_block_order_recovery_execution(plan_sha256=digest,lane=lane))


@pytest.mark.parametrize('lane',RECOVERY_LANES)
@pytest.mark.parametrize('preflight',(False,True))
def test_block_order_recovery_exact_original_order_and_scoped_roles(monkeypatch,tmp_path,lane,preflight):
    f=block_order_recovery_fixture(monkeypatch,tmp_path,lane,preflight);proof=f['call']()
    assert proof['lane']==lane and proof['preflight']is preflight
    assert proof['training_authority_granted']is proof['optimizer_authority_granted']is False
    assert f['called']==['cuda']and not f['output'].exists()


@pytest.mark.parametrize('bad',('schema','execution','chronological_original','chronological_receipt','design',
    'metadata_receipt','producer_lane','trajectory','schedule','model','initializer','recipe','protocol_schedule',
    'protocol_trajectory','optimizer_steps','model_forwards','checkpoint_writes','checkpoint_loads','budget_bool',
    'owner','bridge','bool_gate','inactive_bool','spent','second_lane','parent_protocol','flags','old_fit','diagnostic',
    'existing','symlink','argv','hardware_env','unknown_env','legacy_env'))
def test_block_order_recovery_rejects_wrong_lineage_or_scope(monkeypatch,tmp_path,bad):
    f=block_order_recovery_fixture(monkeypatch,tmp_path,bad=bad)
    with pytest.raises(RuntimeError):f['call']()
    assert not f['called']and not(f['output']/'CLAIM.json').exists()


def test_block_order_recovery_preflight_rejects_false_boolean_count(monkeypatch,tmp_path):
    f=block_order_recovery_fixture(monkeypatch,tmp_path,preflight=True)
    policy_path=f['repo']/'NEXT_RUN_POLICY.json';policy=json.loads(policy_path.read_text())
    policy[ORDER_RECOVERY_NAME.lower()]['lane_gates'][f['lane']]={'execution_authorized':False,'remaining_invocations':False,
        'training_authorized':False,'test_authorized':False,'relaunch_allowed':False}
    policy_path.write_text(json.dumps(policy))
    with pytest.raises(RuntimeError,match='BLOCK_ORDER_RECOVERY_TYPED_LANES'):f['call']()
    assert not f['called']


@pytest.mark.parametrize('lane',RECOVERY_LANES)
def test_block_order_recovery_shell_exact_lane_profiles(monkeypatch,tmp_path,lane):
    f=block_order_recovery_fixture(monkeypatch,tmp_path,lane)
    variables,functions,dispatch,_=[v[2]for v in g._V3_BLOCK_ORDER_RECOVERY_SHELL_ADDITIONS]
    functions=functions.replace('/home/andre2/GX1_RUNS',str(f['runs']))
    state=dict(REPO_ROOT=str(f['repo']),CANONICAL_TRAINER_PYTHON=str(f['repo']/'.venv/bin/python'),
        JOB_CLASS='trainer',MEM='20G',SWAP='512M',ATTENDED_SMOKE='false',CUDA_PRODUCER_GUARD='false',
        POWER_BENCHMARK_SCOPE_JSON='',POWER_BENCHMARK_SCOPE_SHA256='')
    script='set -eu\n'+''.join(k+'='+shlex.quote(v)+'\n'for k,v in state.items())+variables+functions
    script+='validate_target_command() {\n'+dispatch+'return 88\n}\n'
    argv=[state['CANONICAL_TRAINER_PYTHON'],'-m','gx1.audit.native_provider_hlc_learning_v1',
        '--v3-block-order-recovery-plan-sha256',f['digest'],'--lane',lane]
    for extra in ([],['--preflight'],['--phase','fit']):
        tail='validate_target_command '+' '.join(map(shlex.quote,argv+extra))
        tail+='\nprintf "%s,%s,%s,%s\n" "$NATIVE_V3_BLOCK_ORDER_RECOVERY_LANE" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_OUT_BUNDLE_DIR"\n'
        result=subprocess.run(['bash','-c',script+tail],cwd=f['repo'],capture_output=True,text=True)
        assert result.returncode==(75 if extra==['--phase','fit']else 0),result.stderr
        if result.returncode==0:
            timing=['5400','5370']if lane=='recovery'else['1800','1770']
            assert result.stdout.strip()==','.join([lane,*timing,str(f['output'])])


@pytest.mark.parametrize('lane',RECOVERY_LANES)
def test_block_order_recovery_requests_only_wrapper_exports(monkeypatch,tmp_path,lane):
    f=block_order_recovery_fixture(monkeypatch,tmp_path,lane,preflight=True)
    wrapper=(Path(__file__).resolve().parents[1]/'scripts/gx1_capped_run.sh').read_text()
    exported=set(re.findall(r'--setenv=([A-Z0-9_]+)=',wrapper));requested=set()
    class ActualWrapperEnvironment(dict):
        def get(self,key,default=None):
            requested.add(key);return super().get(key,default)
    actual={k:v for k,v in os.environ.items()if not k.startswith('GX1_')or k in exported}
    monkeypatch.setattr(os,'environ',ActualWrapperEnvironment(actual));f['call']()
    assert requested<=exported and f['called']==['cuda']


def _actual_block_order_recovery_owners():
    repo=Path(g.__file__).resolve().parents[2]
    return {n:hashlib.sha256(subprocess.check_output(['git','show',g._V3_BLOCK_ORDER_RECOVERY_PARENT+':'+n],cwd=repo)).hexdigest()for n in OWNERS}


def test_block_order_recovery_bridge_preserves_all_previous_bytes():
    proof=g.native_v3_block_order_recovery_source_bridge(_actual_block_order_recovery_owners())
    assert proof['old_guard_routes_preserved']and proof['old_numeric_resource_environment_function_unchanged']
    assert proof['new_order_recovery_controlflow_requires_scoped_review']is True


@pytest.mark.parametrize('bad',('legacy_resource','new_shell_limit','producer_hash'))
def test_block_order_recovery_bridge_rejects_unreviewed_delta(monkeypatch,bad):
    owners=_actual_block_order_recovery_owners()
    if bad=='producer_hash':owners[OWNERS[0]]='0'*64
    else:
        original=Path.read_text;repo=Path(g.__file__).resolve().parents[2]
        def changed(path,*a,**kw):
            text=original(path,*a,**kw)
            if path==repo/OWNERS[0]and bad=='legacy_resource':
                text=text.replace('_MAX_MEMORY_BYTES = 20 * 1024**3','_MAX_MEMORY_BYTES = 21 * 1024**3',1)
            if path==repo/OWNERS[1]and bad=='new_shell_limit':
                fragment=g._V3_BLOCK_ORDER_RECOVERY_SHELL_ADDITIONS[1][2]
                text=text.replace(fragment,fragment.replace('TRAINER_GPU_MAX_POWER_DRAW_W=250','TRAINER_GPU_MAX_POWER_DRAW_W=251'),1)
            return text
        monkeypatch.setattr(Path,'read_text',changed)
    with pytest.raises(RuntimeError):g.native_v3_block_order_recovery_source_bridge(owners)


ISSUE_MIX_NAME = 'NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_ISSUE_MIX_20261128_001'
ISSUE_MIX_LANES = ('reference', 'cold', 'fit')


@pytest.mark.parametrize('lane', ISSUE_MIX_LANES)
def test_issue_mix_reuses_exact_existing_technical_and_fit_resources(lane):
    phase = 'fit' if lane == 'fit' else 'cold' if lane == 'cold' else 'warm'
    assert g._native_v3_issue_mix_execution(lane) == g._native_v3_development_execution(phase)
    assert g._V3_ISSUE_MIX_PROFILE == 'gx1_native_provider_hlc_v3_issue_mix_trainer20g_v1'


def _actual_issue_mix_owners():
    repo = Path(g.__file__).resolve().parents[2]
    return {n: hashlib.sha256(subprocess.check_output(
        ['git', 'show', g._V3_ISSUE_MIX_PARENT + ':' + n], cwd=repo)).hexdigest() for n in OWNERS}


def test_issue_mix_bridge_preserves_all_previous_guard_and_shell_bytes():
    proof = g.native_v3_issue_mix_source_bridge(_actual_issue_mix_owners())
    assert proof['checked_guard_owners'] == 2 and proof['old_guard_routes_preserved']
    assert proof['old_numeric_resource_environment_function_unchanged']
    assert proof['new_issue_mix_controlflow_requires_scoped_review'] is True


@pytest.mark.parametrize('bad', ('legacy_resource', 'new_shell_limit', 'extra_statement', 'producer_hash'))
def test_issue_mix_bridge_rejects_unreviewed_predecessor_or_shell_delta(monkeypatch, bad):
    owners = _actual_issue_mix_owners()
    if bad == 'producer_hash':
        owners[OWNERS[0]] = '0' * 64
    else:
        original = Path.read_text; repo = Path(g.__file__).resolve().parents[2]
        def changed(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            if path == repo / OWNERS[0] and bad == 'legacy_resource':
                text = text.replace('_MAX_MEMORY_BYTES = 20 * 1024**3', '_MAX_MEMORY_BYTES = 21 * 1024**3', 1)
            if path == repo / OWNERS[0] and bad == 'extra_statement':
                text = text.replace('def _main() -> int:', 'UNREVIEWED_ISSUE_MIX_ADDITION = True\n\ndef _main() -> int:', 1)
            if path == repo / OWNERS[1] and bad == 'new_shell_limit':
                fragment = g._V3_ISSUE_MIX_SHELL_ADDITIONS[1][2]
                text = text.replace(fragment, fragment.replace('TRAINER_GPU_MAX_POWER_DRAW_W=250',
                    'TRAINER_GPU_MAX_POWER_DRAW_W=251'), 1)
            return text
        monkeypatch.setattr(Path, 'read_text', changed)
    with pytest.raises(RuntimeError):
        g.native_v3_issue_mix_source_bridge(owners)


def issue_mix_fixture(monkeypatch, tmp_path, lane='fit', preflight=False, bad=None):
    f = block_order_recovery_fixture(monkeypatch, tmp_path, preflight=True)
    repo, runs = f['repo'], f['runs']; control = f['plan']['metadata']['original']
    original = control['metadata']['original']
    original['schedule'].update(rows=20727, source_rows_sha256='1' * 64, issue_ns_sha256='2' * 64)
    control['schedule']['original_batch_rows'] = [96] * 215 + [87]
    control['initial_reuse'] = {'INITIAL_score': {'path': str(runs / 'INITIAL_SCORE.json'), 'sha256': '3' * 64}}
    control_ref = dict(path=str(repo / 'configs/research' / (BLOCK_ORDER_NAME + '.json')),
        sha256='15a59b3840d61d57d6e217967312a6b76e4e89d256ea902b619ec3fa8099a008')
    evidence = runs / 'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'; evidence.mkdir()
    design_ref = dict(path=str(evidence / 'V3_DIRECTION_ISSUE_MIX_DESIGN_20261128_001.json'),
        sha256='93a59021614b48f67a0b71e66c29926e3fc302d2f4bcf6187e9f8bdf7c0d75d2')
    sizes = [control['schedule']['original_batch_rows'][i] for i in control['schedule']['logical_original_batch_ids']]
    # Structural synthetic partitions only; the leaf tests own canonical private-PCG64 generation.
    takes = []
    for p in range(2):
        offset = 0
        for size in sizes[p * 216:(p + 1) * 216]:
            takes.append(list(range(offset, offset + size))); offset += size
    ordering = dict(generator='private_numpy_Generator_PCG64', logical_step_rows=sizes,
        control_schedule_sha256=control['schedule']['contract_sha256'],
        raw_permutation_state_sha256_by_pass=['4' * 64, '5' * 64],
        actual_sorted_partition_stream_state_sha256_by_pass=['6' * 64, '7' * 64],
        actual_cohort_take_state_sha256_by_logical_batch=['8' * 64] * 432)
    design = dict(ordering=ordering, single_changed_variable='TRAIN minibatch membership only')
    schedule = dict(schema_version='gx1_native_v3_issue_mix_schedule_v1', rows=20727, passes=2,
        batch_rows=96, batches_per_pass=216, logical_batches=432, optimizer_updates=432, skipped_batches=0,
        seed=20260911, drop_last=False, shuffle=True, generator=ordering['generator'], update_flags=[True] * 432,
        logical_step_rows=sizes, cohort_takes=takes, control_schedule_sha256=ordering['control_schedule_sha256'],
        raw_permutation_state_sha256_by_pass=ordering['raw_permutation_state_sha256_by_pass'],
        actual_sorted_partition_stream_state_sha256_by_pass=ordering['actual_sorted_partition_stream_state_sha256_by_pass'],
        cohort_take_state_sha256_by_logical_batch=ordering['actual_cohort_take_state_sha256_by_logical_batch'],
        source_rows_sha256=original['schedule']['source_rows_sha256'], issue_ns_sha256=original['schedule']['issue_ns_sha256'])
    trajectory = dict(schedule=schedule, control_plan=control_ref, issue_mix_design=design_ref)
    limits = {}
    for role in ISSUE_MIX_LANES:
        values = dict.fromkeys(original['limits'], 0)
        updates, fw, writes = dict(reference=(2, 2, 2), cold=(1, 1, 1), fit=(432, 597, 2))[role]
        values.update(model_constructions=1, optimizer_constructions=1, feature_cache_payload_loads=1,
            target_archive_loads=1, model_forwards=fw, backbone_forwards=fw, readout_forwards=fw,
            optimizer_steps=updates, backward_calls=updates, checkpoint_writes=writes, checkpoint_readbacks=writes,
            checkpoint_receipt_writes=writes, checkpoint_prepublication_loads=writes)
        if role == 'cold': values.update(checkpoint_loads=1, model_restores=1)
        if role == 'fit': values.update(autograd_calls=9, logit_archive_writes=2, logit_archive_readbacks=2)
        limits[role] = values
    scope = dict(candidate_phase='PASS2', single_changed_variable=design['single_changed_variable'],
        only_all5_and_original_stability_criteria=True, technical_weights_never_initializer=True,
        TEST_VAL_M1_GC_trade_spending=False)
    protocol = dict(schema_version='gx1_v3_direction_issue_mix_protocol_v1', experiment_id=ISSUE_MIX_NAME,
        design=design_ref, original_plan=control_ref, original_schedule=control['schedule'],
        original_trajectory=trajectory, schedule=schedule, recipe=original['recipe'],
        initial_reuse=control['initial_reuse'], physical_input_identity={'binding': 'unchanged'}, limits=limits, scope=scope)
    owners = original['source_owners']
    bridge = dict(producer_commit=g._V3_ISSUE_MIX_PARENT,
        owners={n: dict(original_sha256='f' * 64, current_sha256=owners[n]) for n in OWNERS})
    if bad == 'unsorted_take': takes[0].reverse()
    if bad == 'boolean_take': takes[0][1] = True
    if bad == 'duplicate_take': takes[0][1] = takes[0][0]
    if bad == 'cross_batch_duplicate': takes[1][0] = takes[0][0]; takes[1].sort()
    if bad == 'boolean_geometry': schedule['skipped_batches'] = False
    if bad == 'integer_update_flag': schedule['update_flags'][0] = 1
    if bad == 'source_hash': schedule['source_rows_sha256'] = '9' * 64
    if bad == 'design_take_hash': schedule['cohort_take_state_sha256_by_logical_batch'] = ['9' * 64] * 432
    if bad == 'budget_bool': limits[lane]['model_constructions'] = True
    if bad == 'optimizer_steps': limits[lane]['optimizer_steps'] += 1
    if bad == 'checkpoint_loads': limits[lane]['checkpoint_loads'] += 1
    if bad == 'absolute_criteria': scope['only_all5_and_original_stability_criteria'] = False
    proto = repo / 'configs/research/NATIVE_ALL_TF_PROVIDER_HLC_V3_DIRECTION_ISSUE_MIX_PROTOCOL_20261128_001.json'
    proto.write_text(json.dumps(protocol)); protocol_ref = dict(path=str(proto), sha256=hashlib.sha256(proto.read_bytes()).hexdigest())
    if bad == 'valid_plan_membership_swap':
        # A valid bijection in the role plan must still match the frozen common protocol.
        takes[0][0], takes[1][0] = takes[1][0], takes[0][0]; takes[0].sort(); takes[1].sort()
    parents = {}
    for role in (() if lane == 'reference' else ('reference',) if lane == 'cold' else ('reference', 'cold')):
        previous = dict(lane=role, protocol=protocol_ref, source_commit='c' * 40 if bad == 'parent_source' else 'a' * 40,
            source_owners=owners, schedule=schedule, metadata={'trajectory': trajectory})
        parent_path = repo / 'configs/research' / (ISSUE_MIX_NAME + '_' + role.upper() + '.json')
        parent_path.write_text(json.dumps(previous)); parents[role] = {'plan': dict(path=str(parent_path), sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest())}
    output = runs / (ISSUE_MIX_NAME + '_' + lane)
    plan = dict(schema_version='gx1_native_v3_direction_issue_mix_plan_v1', stage=ISSUE_MIX_NAME.lower(),
        lane=lane, source_commit='a' * 40, source_owners=owners, output_root=str(output),
        execution_profile=g._V3_ISSUE_MIX_PROFILE, execution=g._native_v3_issue_mix_execution(lane),
        original_plan=control_ref, design=design_ref, protocol=protocol_ref,
        metadata={'original': original, 'control': control, 'trajectory': trajectory},
        recipe=original['recipe'], model_contract=original['model_contract'], initial_reference=original['initial_reference'],
        initial_reuse=control['initial_reuse'], physical_input_identity=protocol['physical_input_identity'],
        scope=scope, schedule=schedule, limits=limits[lane], guard_source_bridge=bridge, parents=parents,
        backbone_gradient_policy=None, starting_cursor=1 if lane == 'cold' else 0, ending_cursor=432 if lane == 'fit' else 2,
        checkpoint_cursors=[1, 432] if lane == 'fit' else [1, 2] if lane == 'reference' else [2], panel_indices=[0, 107, 214])
    if bad == 'schema': plan['schema_version'] = 'old'
    if bad == 'profile': plan['execution_profile'] = 'generic_trainer'
    if bad == 'execution': plan['execution']['GPU_draw_stop_W'] = 251
    if bad == 'original_plan': plan['original_plan'] = original['original_plan'] if 'original_plan' in original else {}
    if bad == 'model': plan['model_contract'] = {}
    if bad == 'initializer': plan['initial_reference'] = {'checkpoint': 'FINAL.pt'}
    if bad == 'backbone_policy': plan['backbone_gradient_policy'] = 'side_only'
    if bad == 'trajectory': plan['metadata']['trajectory'] = {}
    if bad == 'checkpoint_cursor': plan['checkpoint_cursors'] = [1, 431]
    if bad == 'panels': plan['panel_indices'] = [0, 107, 215]
    if bad == 'owner': plan['source_owners'] = {**owners, OWNERS[0]: '0' * 64}
    if bad == 'bridge': plan['guard_source_bridge'] = {**bridge, 'extra': True}
    path = repo / 'configs/research' / (ISSUE_MIX_NAME + '_' + lane.upper() + '.json')
    path.write_text(json.dumps(plan)); digest = hashlib.sha256(path.read_bytes()).hexdigest()
    gate = dict(status='PREPARED_CLOSED_NOT_ADMITTED' if preflight else 'ADMITTED_EXACT_V3_ISSUE_MIX_' + lane.upper(),
        plan=dict(path=str(path), sha256=digest), execution_authorized=not preflight,
        remaining_invocations=int(not preflight), relaunch_allowed=False, training_authorized=not preflight, test_authorized=False)
    closed = lambda binding: dict(status='CLOSED_CONSUMED_COMPLETE', plan=binding, execution_authorized=False,
        remaining_invocations=0, relaunch_allowed=False, training_authorized=False, test_authorized=False)
    gates = {role: closed(value['plan']) for role, value in parents.items()}; gates[lane] = gate
    parent = dict(original_plan=control_ref, design=design_ref, protocol=protocol_ref, main_fit_authorized=False,
        test_authorized=False, spending_authorized=False, status='SOURCE_PREPARATION_ONLY' if preflight else gate['status'],
        execution_authorized=not preflight, active_lane=None if preflight else lane, lane_gates=gates)
    policy_path = repo / 'NEXT_RUN_POLICY.json'; policy = json.loads(policy_path.read_text())
    for key in ('training_enabled', 'native_training_enabled'): policy[key] = not preflight
    policy[ISSUE_MIX_NAME.lower()] = parent
    if bad == 'boolean_gate': gate['remaining_invocations'] = not preflight
    if bad == 'inactive_boolean_gate':
        extra = 'cold' if lane == 'reference' else 'reference'; gates[extra] = closed({}); gates[extra]['remaining_invocations'] = False
    if bad == 'second_lane': gates['cold' if lane == 'reference' else 'reference'] = gate
    if bad == 'flags': policy['full_val_allowed'] = True
    if bad == 'old_route': policy[ORDER_RECOVERY_NAME.lower()]['execution_authorized'] = True
    if bad == 'parent_protocol': parent['protocol'] = {}
    if bad == 'parent_lane': parent['active_lane'] = 'wrong'
    policy_path.write_text(json.dumps(policy))
    snapshot = evidence / ('V3_DIRECTION_ISSUE_MIX_' + lane.upper() + '_ADMITTED_POLICY_20261128_001.json')
    if not preflight: snapshot.write_bytes(b'{}' if bad == 'snapshot' else policy_path.read_bytes())
    if bad == 'existing': output.mkdir()
    if bad == 'symlink':
        other = path.with_suffix('.original'); path.rename(other); path.symlink_to(other)
    read = g._native_v3_development_read
    def fixture_read(candidate, candidate_digest=None):
        if str(candidate) == control_ref['path']:
            assert candidate_digest == control_ref['sha256']; return control
        if str(candidate) == design_ref['path']:
            assert candidate_digest == design_ref['sha256']; return design
        return read(candidate, candidate_digest)
    monkeypatch.setattr(g, '_native_v3_development_read', fixture_read)
    monkeypatch.setattr(g, 'native_v3_issue_mix_source_bridge', lambda *a, **k: bridge)
    argv = [str(repo / 'gx1/audit/native_provider_hlc_learning_v1.py'), '--v3-issue-mix-plan-sha256', digest, '--lane', lane]
    if preflight: argv += ['--preflight']
    if bad == 'argv': argv += ['--phase', 'fit']
    if bad == 'python': monkeypatch.setattr(sys, 'executable', '/usr/bin/python3')
    monkeypatch.setattr(sys, 'argv', argv)
    for key in list(os.environ):
        if key.startswith('GX1_NATIVE_'): monkeypatch.delenv(key)
    execution = g._native_v3_issue_mix_execution(lane)
    env = dict(GX1_NATIVE_V3_ISSUE_MIX='true', GX1_NATIVE_V3_ISSUE_MIX_PLAN_SHA256=digest,
        GX1_NATIVE_V3_ISSUE_MIX_LANE=lane, GX1_TRAINER_MAX_WALL_SECONDS=str(execution['outer_seconds']),
        GX1_TRAINER_MODEL_MAX_WALL_SECONDS=str(execution['internal_seconds']))
    if bad == 'hardware_env': env['GX1_TRAINER_GPU_MAX_POWER_DRAW_W'] = '251'
    if bad == 'unknown_env': env['GX1_NATIVE_V3_ISSUE_MIX_UNSAFE'] = 'false'
    if bad == 'legacy_env': env['GX1_NATIVE_V3_BLOCK_ORDER_RECOVERY'] = 'true'
    for key, value in env.items(): monkeypatch.setenv(key, value)
    return dict(f, path=path, digest=digest, plan=plan, output=output, lane=lane,
        call=lambda: g.require_native_v3_issue_mix_execution(plan_sha256=digest, lane=lane))


@pytest.mark.parametrize('lane', ISSUE_MIX_LANES)
@pytest.mark.parametrize('preflight', (False, True))
def test_issue_mix_exact_role_guard_has_only_resource_authority(monkeypatch, tmp_path, lane, preflight):
    f = issue_mix_fixture(monkeypatch, tmp_path, lane, preflight)
    proof = f['call']()
    assert proof['lane'] == lane and proof['preflight'] is preflight and f['called'] == ['cuda']
    assert proof['training_authority_granted'] is proof['optimizer_authority_granted'] is False
    assert not f['output'].exists()


@pytest.mark.parametrize('bad', ('schema', 'profile', 'execution', 'original_plan', 'model', 'initializer',
    'backbone_policy', 'trajectory', 'checkpoint_cursor', 'panels', 'owner', 'bridge', 'unsorted_take',
    'boolean_take', 'duplicate_take', 'cross_batch_duplicate', 'boolean_geometry', 'integer_update_flag',
    'source_hash', 'design_take_hash', 'valid_plan_membership_swap', 'budget_bool', 'optimizer_steps',
    'checkpoint_loads', 'absolute_criteria', 'parent_source', 'boolean_gate', 'inactive_boolean_gate',
    'second_lane', 'flags', 'old_route', 'parent_protocol', 'parent_lane', 'snapshot', 'existing', 'symlink',
    'argv', 'python', 'hardware_env', 'unknown_env', 'legacy_env'))
def test_issue_mix_rejects_mismatched_membership_source_or_admission(monkeypatch, tmp_path, bad):
    f = issue_mix_fixture(monkeypatch, tmp_path, bad=bad)
    with pytest.raises(RuntimeError): f['call']()
    assert not f['called']


def test_issue_mix_preflight_rejects_false_boolean_invocation_count(monkeypatch, tmp_path):
    f = issue_mix_fixture(monkeypatch, tmp_path, lane='reference', preflight=True, bad='boolean_gate')
    with pytest.raises(RuntimeError, match='ISSUE_MIX_TYPED_LANES'): f['call']()
    assert not f['called']


@pytest.mark.parametrize('lane', ISSUE_MIX_LANES)
def test_issue_mix_guard_requests_only_actual_wrapper_exports(monkeypatch, tmp_path, lane):
    f = issue_mix_fixture(monkeypatch, tmp_path, lane, preflight=True)
    wrapper = (Path(__file__).resolve().parents[1] / 'scripts/gx1_capped_run.sh').read_text()
    exported = set(re.findall(r'--setenv=([A-Z0-9_]+)=', wrapper)); requested = set()
    class ActualWrapperEnvironment(dict):
        def get(self, key, default=None):
            requested.add(key); return super().get(key, default)
    actual = {k: v for k, v in os.environ.items() if not k.startswith('GX1_') or k in exported}
    monkeypatch.setattr(os, 'environ', ActualWrapperEnvironment(actual)); f['call']()
    assert requested <= exported and f['called'] == ['cuda']


@pytest.mark.parametrize('lane', ISSUE_MIX_LANES)
def test_issue_mix_shell_exact_roles_and_unchanged_profiles(monkeypatch, tmp_path, lane):
    f = issue_mix_fixture(monkeypatch, tmp_path, lane)
    variables, functions, dispatch, _ = [v[2] for v in g._V3_ISSUE_MIX_SHELL_ADDITIONS]
    functions = functions.replace('/home/andre2/GX1_RUNS', str(f['runs']))
    state = dict(REPO_ROOT=str(f['repo']), CANONICAL_TRAINER_PYTHON=str(f['repo'] / '.venv/bin/python'),
        JOB_CLASS='trainer', MEM='20G', SWAP='512M', ATTENDED_SMOKE='false', CUDA_PRODUCER_GUARD='false',
        POWER_BENCHMARK_SCOPE_JSON='', POWER_BENCHMARK_SCOPE_SHA256='')
    script = 'set -eu\n' + ''.join(k + '=' + shlex.quote(v) + '\n' for k, v in state.items()) + variables + functions
    script += 'validate_target_command() {\n' + dispatch + 'return 88\n}\n'
    argv = [state['CANONICAL_TRAINER_PYTHON'], '-m', 'gx1.audit.native_provider_hlc_learning_v1',
        '--v3-issue-mix-plan-sha256', f['digest'], '--lane', lane]
    for extra in ([], ['--preflight'], ['--phase', 'fit']):
        tail = 'validate_target_command ' + ' '.join(map(shlex.quote, argv + extra))
        tail += '\nprintf "%s,%s,%s,%s\\n" "$NATIVE_V3_ISSUE_MIX_LANE" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_OUT_BUNDLE_DIR"\n'
        result = subprocess.run(['bash', '-c', script + tail], cwd=f['repo'], capture_output=True, text=True)
        assert result.returncode == (75 if extra == ['--phase', 'fit'] else 0), result.stderr
        if result.returncode == 0:
            timing = ['5400', '5370'] if lane == 'fit' else ['1800', '1770']
            assert result.stdout.strip() == ','.join([lane, *timing, str(f['output'])])


TRAIN_EVAL_GAP_NAME = 'NATIVE_ALL_TF_PROVIDER_HLC_V3_TRAIN_EVAL_GAP_20261128_001'


def train_eval_gap_fixture(monkeypatch, tmp_path, preflight=False, bad=None):
    f = issue_mix_fixture(monkeypatch, tmp_path, preflight=True)
    repo, runs = f['repo'], f['runs']; producer = copy.deepcopy(f['plan'])
    original_ref = dict(path=str(repo / 'configs/research' / (ISSUE_MIX_NAME + '_FIT.json')),
        sha256='25ca4cc9d780f5d16376a765ec58f6d84c58a1f16f55d48033d11168fda2d4a8')
    final_root = runs / (ISSUE_MIX_NAME + '_fit')
    receipt_ref = dict(path=str(final_root / 'FINAL.json'),
        sha256='1b67d9e51e211480c168b0eae05e5c1677e804e4e4d58e7b9bb1fb619271884d')
    checkpoint = dict(path=str(final_root / 'FINAL.pt'),
        sha256='577b5b37f5c4cfa98744cfeefb06e8156231488a7adc038f33cdfd685df86632')
    evidence = runs / 'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'
    design_ref = dict(path=str(evidence / 'V3_TRAIN_EVAL_GAP_DESIGN_20261128_003.json'),
        sha256='eb0f66e2e1454f3f0f56e10016be039d2cef56f421556a2363f5db461f34a000')
    logits = dict(path=str(final_root / 'PASS2_LOGITS.npz'),
        sha256='e8dd65bc8dc21c83f6d894634a3c16f3464d3426099fb9d3cf81adaa1177048d')
    score = dict(path=str(final_root / 'PASS2_SCORE.json'),
        sha256='d6e7c600244bb28867dab7dc7ba39af23591bb264e91390b8e641f0712a2d21e')
    receipt = dict(logical_cursor=432, optimizer_updates=432, producer_lane='fit',
        original_plan=original_ref, execution_plan=original_ref, checkpoint=checkpoint)
    design = dict(producer_fit_plan=original_ref, frozen_FINAL=checkpoint, frozen_FINAL_receipt=receipt_ref,
        saved_eval_logits=logits, saved_eval_score=score)
    limits = dict.fromkeys(producer['limits'], 0)
    limits.update(model_constructions=1, optimizer_constructions=1, feature_cache_payload_loads=1,
        target_archive_loads=1, checkpoint_loads=1, model_restores=1, model_forwards=405,
        backbone_forwards=405, readout_forwards=405, logit_archive_writes=5, logit_archive_readbacks=5,
        saved_eval_logit_loads=1)
    bridge = dict(producer_commit=g._V3_TRAIN_EVAL_GAP_PARENT,
        owners={n: dict(original_sha256='f' * 64, current_sha256=producer['source_owners'][n]) for n in OWNERS})
    output = runs / TRAIN_EVAL_GAP_NAME
    plan = dict(schema_version='gx1_native_v3_train_eval_gap_plan_v1', stage=TRAIN_EVAL_GAP_NAME.lower(),
        source_commit='a' * 40, output_root=str(output), execution=g._native_v3_development_execution('warm'),
        execution_profile=g._V3_TRAIN_EVAL_GAP_PROFILE, original_plan=original_ref,
        checkpoint_receipt=receipt_ref, design=design_ref, metadata=dict(original=producer['metadata']['original'],
        producer=producer, receipt=receipt, trajectory=producer['metadata']['trajectory']),
        model_contract=producer['model_contract'], recipe=producer['recipe'],
        physical_input_identity=producer['physical_input_identity'],
        inputs=dict(design=design_ref, final_checkpoint_receipt=receipt_ref, final_checkpoint=checkpoint,
            saved_eval_logits=logits, saved_eval_score=score), limits=limits, source_owners=producer['source_owners'],
        guard_source_bridge=bridge, passes=[dict(label='EVAL_PARITY', training=False, equal_tf=True)] + [
            dict(label='TRAIN_DRAW_' + str(i).zfill(2), training=True, equal_tf=False) for i in range(1, 5)])
    if bad == 'schema': plan['schema_version'] = 'old'
    if bad == 'root': plan['output_root'] = str(output).lower()
    if bad == 'profile': plan['execution_profile'] = 'generic_trainer'
    if bad == 'execution': plan['execution']['GPU_draw_stop_W'] = 251
    if bad == 'design': plan['design'] = {**design_ref, 'sha256': '0' * 64}
    if bad == 'superseded_design': plan['design'] = dict(path=str(evidence / 'V3_TRAIN_EVAL_GAP_DESIGN_20261128_002.json'), sha256='78ef8219ea16005691736721c7a26cc629868011cd30d2e2d5433d4b2206403c')
    if bad == 'producer': plan['original_plan'] = {}
    if bad == 'receipt': plan['checkpoint_receipt'] = {}
    if bad == 'model': plan['model_contract'] = {}
    if bad == 'metadata': plan['metadata']['producer'] = {}
    if bad == 'trajectory': plan['metadata']['trajectory'] = {}
    if bad == 'checkpoint': plan['inputs']['final_checkpoint'] = {}
    if bad == 'saved_logits': plan['inputs']['saved_eval_logits'] = {}
    if bad == 'saved_score': plan['inputs']['saved_eval_score'] = {}
    if bad == 'identity': plan['physical_input_identity'] = {}
    if bad == 'owner': plan['source_owners'] = {**producer['source_owners'], OWNERS[0]: '0' * 64}
    if bad == 'bridge': plan['guard_source_bridge'] = {**bridge, 'extra': True}
    if bad in ('optimizer_steps', 'backward_calls', 'autograd_calls', 'checkpoint_writes', 'normalization_fits', 'prior_fits'):
        limits[bad] = 1
    if bad in ('model_forwards', 'saved_eval_logit_loads'): limits[bad] += 1
    if bad == 'bool_budget': limits['model_constructions'] = True
    if bad == 'extra_draw': plan['passes'].append(plan['passes'][-1])
    if bad == 'eval_train': plan['passes'][0]['training'] = True
    if bad == 'integer_mode': plan['passes'][1]['training'] = 1
    path = repo / 'configs/research' / (TRAIN_EVAL_GAP_NAME + '.json')
    path.write_text(json.dumps(plan)); digest = hashlib.sha256(path.read_bytes()).hexdigest()
    policy_path = repo / 'NEXT_RUN_POLICY.json'; policy = json.loads(policy_path.read_text())
    parent = dict(status='PREPARED_CLOSED_NOT_ADMITTED' if preflight else 'ADMITTED_EXACT_V3_TRAIN_EVAL_GAP',
        plan=dict(path=str(path), sha256=digest), design=design_ref, execution_authorized=not preflight,
        remaining_invocations=int(not preflight), relaunch_allowed=False, training_authorized=False,
        test_authorized=False, spending_authorized=False)
    policy[TRAIN_EVAL_GAP_NAME.lower()] = parent
    grouped = ('native_v3_bounded_development_test_20261127_001', 'native_v3_fit004_exact_recovery_20261128_001',
        'native_v3_fit004_exact_recovery_20261128_002', ORDER_RECOVERY_NAME.lower(), ISSUE_MIX_NAME.lower())
    for scope in grouped:
        policy[scope] = dict(execution_authorized=False, active_phase=None, active_lane=None,
            lane_gates={'old': dict(execution_authorized=False, remaining_invocations=0,
                relaunch_allowed=False, training_authorized=False, test_authorized=False)})
    for scope in (DIRECTION_NAME.lower(), BLOCK_ORDER_NAME.lower()):
        policy[scope] = dict(execution_authorized=False, remaining_invocations=0,
            relaunch_allowed=False, training_authorized=False)
    if bad == 'bool_gate': parent['remaining_invocations'] = not preflight
    if bad == 'spent': parent['remaining_invocations'] = 0
    if bad == 'parent_training': parent['training_authorized'] = True
    if bad == 'parent_spending': parent['spending_authorized'] = True
    if bad == 'parent_plan': parent['plan'] = {}
    if bad == 'old_route': policy[grouped[-1]]['execution_authorized'] = True
    if bad == 'old_gate': policy[grouped[-1]]['lane_gates']['old']['remaining_invocations'] = 1
    if bad == 'old_boolean_gate': policy[grouped[-1]]['lane_gates']['old']['remaining_invocations'] = False
    if bad == 'old_training': policy[grouped[-1]]['lane_gates']['old']['training_authorized'] = True
    if bad == 'old_flat': policy[DIRECTION_NAME.lower()]['execution_authorized'] = True
    policy_path.write_text(json.dumps(policy))
    if not preflight:
        snapshot = evidence / 'V3_TRAIN_EVAL_GAP_ADMITTED_POLICY_20261128_001.json'
        snapshot.write_bytes(b'{}' if bad == 'snapshot' else policy_path.read_bytes())
    if bad == 'existing': output.mkdir()
    if bad == 'symlink':
        other = path.with_suffix('.original'); path.rename(other); path.symlink_to(other)
    read = g._native_v3_development_read
    def fixture_read(candidate, candidate_digest=None):
        for binding, value in ((original_ref, producer), (receipt_ref, receipt), (design_ref, design)):
            if str(candidate) == binding['path']:
                assert candidate_digest == binding['sha256']; return value
        return read(candidate, candidate_digest)
    monkeypatch.setattr(g, '_native_v3_development_read', fixture_read)
    monkeypatch.setattr(g, 'native_v3_train_eval_gap_source_bridge', lambda *a, **k: bridge)
    argv = [str(repo / 'gx1/audit/native_provider_hlc_learning_v1.py'), '--v3-train-eval-gap-plan-sha256', digest]
    if preflight: argv += ['--preflight']
    if bad == 'argv': argv += ['--lane', 'fit']
    if bad == 'python': monkeypatch.setattr(sys, 'executable', '/usr/bin/python3')
    monkeypatch.setattr(sys, 'argv', argv)
    for key in list(os.environ):
        if key.startswith('GX1_NATIVE_'): monkeypatch.delenv(key)
    env = dict(GX1_NATIVE_V3_TRAIN_EVAL_GAP='true', GX1_NATIVE_V3_TRAIN_EVAL_GAP_PLAN_SHA256=digest,
        GX1_TRAINER_MAX_WALL_SECONDS='1800', GX1_TRAINER_MODEL_MAX_WALL_SECONDS='1770')
    if bad == 'hardware_env': env['GX1_TRAINER_GPU_MAX_POWER_DRAW_W'] = '251'
    if bad == 'unknown_env': env['GX1_NATIVE_V3_TRAIN_EVAL_GAP_TRAINING'] = 'false'
    if bad == 'legacy_env': env['GX1_NATIVE_V3_ISSUE_MIX'] = 'true'
    for key, value in env.items(): monkeypatch.setenv(key, value)
    return dict(f, path=path, digest=digest, plan=plan, output=output,
        call=lambda: g.require_native_v3_train_eval_gap_execution(plan_sha256=digest))


@pytest.mark.parametrize('preflight', (False, True))
def test_train_eval_gap_frozen_final_without_training_authority(monkeypatch, tmp_path, preflight):
    f = train_eval_gap_fixture(monkeypatch, tmp_path, preflight); proof = f['call']()
    assert proof['preflight'] is preflight and proof['kind'] == 'provider_hlc_v3_train_eval_gap'
    assert proof['training_authority_granted'] is proof['optimizer_authority_granted'] is False
    assert f['called'] == ['cuda'] and not f['output'].exists()


@pytest.mark.parametrize('bad', ('schema', 'root', 'profile', 'execution', 'design', 'superseded_design', 'producer', 'receipt', 'model',
    'metadata', 'trajectory', 'checkpoint', 'saved_logits', 'saved_score', 'identity', 'owner', 'bridge',
    'optimizer_steps', 'backward_calls', 'autograd_calls', 'checkpoint_writes', 'normalization_fits', 'prior_fits',
    'model_forwards', 'saved_eval_logit_loads', 'bool_budget', 'extra_draw', 'eval_train', 'integer_mode',
    'bool_gate', 'spent', 'parent_training', 'parent_spending', 'parent_plan', 'old_route', 'old_gate',
    'old_boolean_gate', 'old_training', 'old_flat', 'snapshot', 'existing', 'symlink', 'argv', 'python',
    'hardware_env', 'unknown_env', 'legacy_env'))
def test_train_eval_gap_rejects_updates_or_mismatched_authority(monkeypatch, tmp_path, bad):
    f = train_eval_gap_fixture(monkeypatch, tmp_path, bad=bad)
    with pytest.raises(RuntimeError): f['call']()
    assert not f['called'] and not (f['output'] / 'CLAIM.json').exists()


@pytest.mark.parametrize('flag', ('training_enabled', 'native_training_enabled', 'native_launch_authorized',
    'full_epoch_training_allowed', 'full_val_allowed', 'exit_training_and_evaluation_this_wave'))
def test_train_eval_gap_keeps_every_global_authority_flag_false(monkeypatch, tmp_path, flag):
    f = train_eval_gap_fixture(monkeypatch, tmp_path, preflight=True)
    path = f['repo'] / 'NEXT_RUN_POLICY.json'; policy = json.loads(path.read_text()); policy[flag] = True
    path.write_text(json.dumps(policy))
    with pytest.raises(RuntimeError, match='TRAIN_EVAL_GAP_ALL_TRAINING_CLOSED'): f['call']()
    assert not f['called']


def test_train_eval_gap_preflight_rejects_boolean_zero_gate(monkeypatch, tmp_path):
    f = train_eval_gap_fixture(monkeypatch, tmp_path, preflight=True, bad='bool_gate')
    with pytest.raises(RuntimeError, match='TRAIN_EVAL_GAP_ONE_USE_GATE'): f['call']()
    assert not f['called']


def test_train_eval_gap_reuses_exact_existing_technical_resources():
    assert g._V3_TRAIN_EVAL_GAP_PROFILE == 'gx1_native_provider_hlc_v3_train_eval_gap_trainer20g_v1'
    assert g._native_v3_development_execution('warm')['outer_seconds'] == 1800


def test_train_eval_gap_shell_exact_cli_and_unchanged_profile(monkeypatch, tmp_path):
    f = train_eval_gap_fixture(monkeypatch, tmp_path)
    variables, functions, dispatch, _ = [v[2] for v in g._V3_TRAIN_EVAL_GAP_SHELL_ADDITIONS]
    functions = functions.replace('/home/andre2/GX1_RUNS', str(f['runs']))
    state = dict(REPO_ROOT=str(f['repo']), CANONICAL_TRAINER_PYTHON=str(f['repo'] / '.venv/bin/python'),
        JOB_CLASS='trainer', MEM='20G', SWAP='512M', ATTENDED_SMOKE='false', CUDA_PRODUCER_GUARD='false',
        POWER_BENCHMARK_SCOPE_JSON='', POWER_BENCHMARK_SCOPE_SHA256='')
    script = 'set -eu\n' + ''.join(k + '=' + shlex.quote(v) + '\n' for k, v in state.items()) + variables + functions
    script += 'validate_target_command() {\n' + dispatch + 'return 88\n}\n'
    argv = [state['CANONICAL_TRAINER_PYTHON'], '-m', 'gx1.audit.native_provider_hlc_learning_v1',
        '--v3-train-eval-gap-plan-sha256', f['digest']]
    for extra in ([], ['--preflight'], ['--lane', 'fit'], ['--preflight', '--preflight']):
        tail = 'validate_target_command ' + ' '.join(map(shlex.quote, argv + extra))
        tail += '\nprintf "%s,%s,%s,%s\\n" "$NATIVE_V3_TRAIN_EVAL_GAP" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_OUT_BUNDLE_DIR"\n'
        result = subprocess.run(['bash', '-c', script + tail], cwd=f['repo'], capture_output=True, text=True)
        assert result.returncode == (0 if extra in ([], ['--preflight']) else 75), result.stderr
        if result.returncode == 0: assert result.stdout.strip() == 'true,1800,1770,' + str(f['output'])


def test_train_eval_gap_requests_only_actual_wrapper_exports(monkeypatch, tmp_path):
    f = train_eval_gap_fixture(monkeypatch, tmp_path, preflight=True)
    wrapper = (Path(__file__).resolve().parents[1] / 'scripts/gx1_capped_run.sh').read_text()
    exported = set(re.findall(r'--setenv=([A-Z0-9_]+)=', wrapper)); requested = set()
    class ActualWrapperEnvironment(dict):
        def get(self, key, default=None):
            requested.add(key); return super().get(key, default)
    actual = {k: v for k, v in os.environ.items() if not k.startswith('GX1_') or k in exported}
    monkeypatch.setattr(os, 'environ', ActualWrapperEnvironment(actual)); f['call']()
    assert requested <= exported and f['called'] == ['cuda']


def _actual_train_eval_gap_owners():
    repo = Path(g.__file__).resolve().parents[2]
    return {n: hashlib.sha256(subprocess.check_output(['git', 'show', g._V3_TRAIN_EVAL_GAP_PARENT + ':' + n], cwd=repo)).hexdigest() for n in OWNERS}


def test_train_eval_gap_bridge_preserves_every_predecessor_guard_and_shell_byte():
    proof = g.native_v3_train_eval_gap_source_bridge(_actual_train_eval_gap_owners())
    assert proof['old_guard_routes_preserved'] and proof['old_numeric_resource_environment_function_unchanged']
    assert proof['new_diagnostic_controlflow_requires_scoped_review'] is True


@pytest.mark.parametrize('bad', ('legacy_resource', 'new_shell_limit', 'producer_hash'))
def test_train_eval_gap_bridge_rejects_unreviewed_delta(monkeypatch, bad):
    owners = _actual_train_eval_gap_owners()
    if bad == 'producer_hash': owners[OWNERS[0]] = '0' * 64
    else:
        original = Path.read_text; repo = Path(g.__file__).resolve().parents[2]
        def changed(path, *a, **kw):
            text = original(path, *a, **kw)
            if path == repo / OWNERS[0] and bad == 'legacy_resource':
                text = text.replace('_MAX_MEMORY_BYTES = 20 * 1024**3', '_MAX_MEMORY_BYTES = 21 * 1024**3', 1)
            if path == repo / OWNERS[1] and bad == 'new_shell_limit':
                fragment = g._V3_TRAIN_EVAL_GAP_SHELL_ADDITIONS[1][2]
                text = text.replace(fragment, fragment.replace('TRAINER_GPU_MAX_POWER_DRAW_W=250', 'TRAINER_GPU_MAX_POWER_DRAW_W=251'), 1)
            return text
        monkeypatch.setattr(Path, 'read_text', changed)
    with pytest.raises(RuntimeError): g.native_v3_train_eval_gap_source_bridge(owners)


def _actual_step_observer_original_core():
    repo = Path(g.__file__).resolve().parents[2]
    return subprocess.check_output(['git', 'show', g._V3_ACTUAL_STEP_OBSERVER_PARENT + ':' + g._V3_ACTUAL_STEP_OBSERVER_CORE], cwd=repo)


def _actual_step_observer_additions(original):
    text = original.decode()
    old, new = g._V3_ACTUAL_STEP_OBSERVER_CORE_SIGNATURE
    assert text.count(old) == 1
    text = text.replace(old, new, 1)
    for anchor, label, call in g._V3_ACTUAL_STEP_OBSERVER_CORE_ADDITIONS:
        fragment = '    # BEGIN GX1 V3 ACTUAL STEP OBSERVER ' + label + ' 20261128\n' + call + \
            '    # END GX1 V3 ACTUAL STEP OBSERVER ' + label + ' 20261128\n'
        # The core may contain other backward/step calls; bind this exact step owner.
        start = text.index(new); first, tail = text[:start], text[start:]
        assert anchor in tail
        text = first + tail.replace(anchor, anchor + fragment, 1)
    return text


def test_actual_step_observer_source_projection_recovers_authentic_core_bytes_and_AST():
    original = _actual_step_observer_original_core()
    observed = _actual_step_observer_additions(original)
    assert g._native_v3_actual_step_observer_core_projection(observed).encode() == original


def test_actual_step_observer_source_bridge_preserves_original_numerical_step():
    original = _actual_step_observer_original_core()
    proof = g.native_v3_actual_step_observer_source_bridge({g._V3_ACTUAL_STEP_OBSERVER_CORE: hashlib.sha256(original).hexdigest()})
    assert proof['observer_off_projection_byte_exact'] and proof['observer_off_projection_AST_exact']
    assert proof['old_numeric_step_unchanged'] and proof['additive_observer_phases'] == 4
    assert proof['training_authority_granted'] is proof['execution_authority_granted'] is False


@pytest.mark.parametrize('bad', ('phase_call', 'inside_operation', 'outside_operation', 'outside_comment',
    'signature_default', 'positional_observer', 'duplicate_marker', 'misplaced_phase', 'missing_phase',
    'producer_hash', 'producer_commit'))
def test_actual_step_observer_source_bridge_rejects_every_unreviewed_delta(monkeypatch, bad):
    original = _actual_step_observer_original_core(); observed = _actual_step_observer_additions(original)
    hashes = {g._V3_ACTUAL_STEP_OBSERVER_CORE: hashlib.sha256(original).hexdigest()}
    commit = g._V3_ACTUAL_STEP_OBSERVER_PARENT
    if bad == 'producer_hash': hashes[g._V3_ACTUAL_STEP_OBSERVER_CORE] = '0' * 64
    if bad == 'producer_commit': commit = '0' * 40
    if bad == 'phase_call': observed = observed.replace("observer('backward',", "observer('step',", 1)
    if bad == 'inside_operation': observed = observed.replace("observer('backward',", "optimizer.step();observer('backward',", 1)
    if bad == 'outside_operation': observed = observed.replace("    optimizer.step();counts['optimizer_steps']+=1\n", "    optimizer.step();optimizer.step();counts['optimizer_steps']+=1\n", 1)
    if bad == 'outside_comment': observed += '\n# Unreviewed extra source delta\n'
    if bad == 'signature_default': observed = observed.replace('*,observer=None):', '*,observer=True):', 1)
    if bad == 'positional_observer': observed = observed.replace('*,observer=None):', 'observer=None):', 1)
    if bad == 'duplicate_marker': observed += '    # BEGIN GX1 V3 ACTUAL STEP OBSERVER BACKWARD 20261128\n'
    anchor, label, call = g._V3_ACTUAL_STEP_OBSERVER_CORE_ADDITIONS[1]
    fragment = '    # BEGIN GX1 V3 ACTUAL STEP OBSERVER ' + label + ' 20261128\n' + call + \
        '    # END GX1 V3 ACTUAL STEP OBSERVER ' + label + ' 20261128\n'
    if bad == 'misplaced_phase': observed = observed.replace(anchor + fragment, fragment + anchor, 1)
    if bad == 'missing_phase': observed = observed.replace(fragment, '', 1)
    original_read = Path.read_bytes; core_path = Path(g.__file__).resolve().parents[2] / g._V3_ACTUAL_STEP_OBSERVER_CORE
    def changed(path, *args, **kwargs):
        return observed.encode() if path == core_path else original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_bytes', changed)
    with pytest.raises(RuntimeError):
        g.native_v3_actual_step_observer_source_bridge(hashes, producer_commit=commit)


def actual_step_observer_fixture(monkeypatch, tmp_path, lane='reference', preflight=False, bad=None):
    f = train_eval_gap_fixture(monkeypatch, tmp_path, preflight=True)
    repo, runs = f['repo'], f['runs']; producer = copy.deepcopy(f['plan']['metadata']['producer'])
    names = ('gx1/audit/native_future_state_step_observer_qualify_v1.py',
        'gx1/audit/native_future_state_step_observer_v1.py', g._V3_ACTUAL_STEP_OBSERVER_CORE)
    owners = dict(producer['source_owners'])
    for name in names:
        path = repo / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# source-only observer fixture\n')
        owners[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    producer['source_owners'][g._V3_ACTUAL_STEP_OBSERVER_CORE] = owners[g._V3_ACTUAL_STEP_OBSERVER_CORE]
    producer_ref = f['plan']['original_plan']; receipt_ref = f['plan']['checkpoint_receipt']
    receipt = f['plan']['metadata']['receipt']; checkpoint_ref = receipt['checkpoint']
    evidence = runs / 'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'
    design_ref = dict(path=str(evidence / 'V3_ACTUAL_STEP_OBSERVER_DESIGN_20261128_001.json'),
        sha256='faf07246d9fd7fdf92811a10668dc0431772efe90f086b2f967ecd2db23a3297')
    protocol_ref = dict(path=str(evidence / 'V3_ACTUAL_STEP_OBSERVER_QUALIFICATION_PROTOCOL_20261128_001.json'),
        sha256='c225d94342af1bf4b787373f1f3cdc5a5a4a08a09464e5c53ed06571492aae40')
    count = 2 if lane == 'reference' else 1; limits = dict.fromkeys(producer['limits'], 0)
    limits.update(model_constructions=1, optimizer_constructions=1, feature_cache_payload_loads=1,
        target_archive_loads=1, checkpoint_loads=1, model_restores=1, model_forwards=count,
        backbone_forwards=count, readout_forwards=count, backward_calls=count, optimizer_steps=count,
        checkpoint_writes=count, checkpoint_readbacks=count, checkpoint_prepublication_loads=count,
        checkpoint_receipt_writes=count, observer_records=0 if lane == 'reference' else 1)
    bridge = dict(producer_commit=g._V3_ACTUAL_STEP_OBSERVER_PARENT,
        owners={n: dict(original_sha256='f' * 64, current_sha256=owners[n]) for n in OWNERS})
    core_bridge = {'exact': 'core projection'}; trajectory = {'logical_batches': 434, 'technical_takes': [0, 1]}
    parents = {}
    if lane == 'cold':
        previous = dict(lane='reference', source_commit='a' * 40, source_owners=owners,
            protocol=protocol_ref, metadata={'trajectory': trajectory})
        previous_path = repo / 'configs/research' / (g._V3_ACTUAL_STEP_OBSERVER_NAME + '_REFERENCE.json')
        previous_path.write_text(json.dumps(previous)); parents['reference'] = {'plan': dict(path=str(previous_path), sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest())}
    output = runs / (g._V3_ACTUAL_STEP_OBSERVER_NAME + '_' + lane)
    plan = dict(schema_version='gx1_native_v3_actual_step_observer_qualification_plan_v1',
        stage=g._V3_ACTUAL_STEP_OBSERVER_STAGE, lane=lane, source_commit='a' * 40, output_root=str(output),
        execution=g._native_v3_development_execution('warm'), execution_profile=g._V3_ACTUAL_STEP_OBSERVER_PROFILE,
        design=design_ref, protocol=protocol_ref, producer_plan=producer_ref, authentic_final_receipt=receipt_ref,
        authentic_final_checkpoint=checkpoint_ref, metadata=dict(original=producer['metadata']['original'],
            producer=producer, trajectory=trajectory), model_contract=producer['model_contract'], recipe=producer['recipe'],
        starting_cursor=432 if lane == 'reference' else 433, ending_cursor=434,
        checkpoint_cursors=[433, 434] if lane == 'reference' else [434], observer_enabled=lane == 'cold',
        technical_weights_never_initializer=True, limits=limits, source_owners=owners,
        core_source_bridge=core_bridge, guard_source_bridge=bridge, parents=parents)
    if bad == 'schema': plan['schema_version'] = 'old'
    if bad == 'profile': plan['execution_profile'] = 'generic'
    if bad == 'execution': plan['execution']['GPU_draw_stop_W'] = 251
    if bad == 'final': plan['authentic_final_checkpoint'] = {}
    if bad == 'producer': plan['producer_plan'] = {}
    if bad == 'metadata': plan['metadata']['producer'] = {}
    if bad == 'model': plan['model_contract'] = {}
    if bad == 'cursor': plan['starting_cursor'] = 0
    if bad == 'boolean_cursor': plan['starting_cursor'] = True
    if bad == 'observer': plan['observer_enabled'] = lane != 'cold'
    if bad == 'initializer': plan['technical_weights_never_initializer'] = False
    if bad == 'checkpoint_cursors': plan['checkpoint_cursors'] = [434, 435]
    if bad == 'updates': limits['optimizer_steps'] += 1
    if bad == 'autograd': limits['autograd_calls'] = 1
    if bad == 'records': limits['observer_records'] += 1
    if bad == 'bool_budget': limits['model_constructions'] = True
    if bad == 'core_bridge': plan['core_source_bridge'] = {}
    if bad == 'guard_bridge': plan['guard_source_bridge'] = {**bridge, 'extra': True}
    if bad == 'owner': plan['source_owners'] = {**owners, names[0]: '0' * 64}
    if bad == 'parents': plan['parents'] = {'extra': {}}
    path = repo / 'configs/research' / (g._V3_ACTUAL_STEP_OBSERVER_NAME + '_' + lane.upper() + '.json')
    path.write_text(json.dumps(plan)); digest = hashlib.sha256(path.read_bytes()).hexdigest()
    policy_path = repo / 'NEXT_RUN_POLICY.json'; policy = json.loads(policy_path.read_text())
    for flag in ('training_enabled', 'native_training_enabled'): policy[flag] = not preflight
    closed = dict(execution_authorized=False, remaining_invocations=0, relaunch_allowed=False, training_authorized=False)
    for scope in ('native_v3_frozen_batch_residual_audit_20261128_001', 'native_v3_frozen_batch_residual_audit_20261128_002'):
        policy[scope] = dict(closed)
    policy[TRAIN_EVAL_GAP_NAME.lower()]['aggregation_completion'] = dict(closed)
    status = 'PREPARED_CLOSED_NOT_ADMITTED' if preflight else 'ADMITTED_EXACT_V3_ACTUAL_STEP_OBSERVER_' + lane.upper()
    gate = dict(status=status, plan=dict(path=str(path), sha256=digest), execution_authorized=not preflight,
        remaining_invocations=int(not preflight), relaunch_allowed=False, training_authorized=not preflight, test_authorized=False)
    lanes = {role: dict(closed, test_authorized=False, status='CLOSED_CONSUMED_COMPLETE', plan=value['plan']) for role, value in parents.items()}
    lanes[lane] = gate
    parent = dict(source_preparation_authorized=True, design=design_ref, protocol=protocol_ref,
        producer_plan=producer_ref, authentic_final_receipt=receipt_ref, authentic_final_checkpoint=checkpoint_ref,
        execution_authorized=not preflight, status='SOURCE_PREPARATION_ONLY' if preflight else status,
        active_lane=None if preflight else lane, lane_gates=lanes)
    policy[g._V3_ACTUAL_STEP_OBSERVER_STAGE] = parent
    if bad == 'bool_gate': gate['remaining_invocations'] = not preflight
    if bad == 'spent': gate['remaining_invocations'] = 0
    if bad == 'flags': policy['native_launch_authorized'] = True
    if bad == 'old_route': policy[TRAIN_EVAL_GAP_NAME.lower()]['execution_authorized'] = True
    if bad == 'old_cpu_route': policy['native_v3_frozen_batch_residual_audit_20261128_001']['remaining_invocations'] = 1
    if bad == 'parent_lane': parent['active_lane'] = 'other'
    if bad == 'parent_protocol': parent['protocol'] = {}
    policy_path.write_text(json.dumps(policy))
    if not preflight:
        snapshot = evidence / ('V3_ACTUAL_STEP_OBSERVER_' + lane.upper() + '_ADMITTED_POLICY_20261128_001.json')
        snapshot.write_bytes(b'{}' if bad == 'snapshot' else policy_path.read_bytes())
    if bad == 'existing': output.mkdir()
    if bad == 'symlink':
        other = path.with_suffix('.original'); path.rename(other); path.symlink_to(other)
    read = g._native_v3_development_read
    def fixture_read(candidate, candidate_digest=None):
        for binding, value in ((producer_ref, producer), (receipt_ref, receipt), (design_ref, {}),
                (protocol_ref, {'schema_version': 'gx1_v3_actual_step_observer_qualification_protocol_v1'})):
            if str(candidate) == binding['path']:
                assert candidate_digest == binding['sha256']; return value
        return read(candidate, candidate_digest)
    monkeypatch.setattr(g, '_native_v3_development_read', fixture_read)
    monkeypatch.setattr(g, 'native_v3_actual_step_observer_source_bridge', lambda *a, **k: core_bridge)
    monkeypatch.setattr(g, 'native_v3_actual_step_observer_guard_source_bridge', lambda *a, **k: bridge)
    argv = [str(repo / names[0]), '--v3-actual-step-observer-plan-sha256', digest] + (['--preflight'] if preflight else [])
    if bad == 'argv': argv += ['--lane', lane]
    monkeypatch.setattr(sys, 'argv', argv)
    if bad == 'python': monkeypatch.setattr(sys, 'executable', '/usr/bin/python3')
    for key in list(os.environ):
        if key.startswith('GX1_NATIVE_'): monkeypatch.delenv(key)
    env = dict(GX1_NATIVE_V3_ACTUAL_STEP_OBSERVER='true', GX1_NATIVE_V3_ACTUAL_STEP_OBSERVER_PLAN_SHA256=digest,
        GX1_NATIVE_V3_ACTUAL_STEP_OBSERVER_LANE=lane)
    if bad == 'hardware_env': env['GX1_TRAINER_GPU_MAX_POWER_DRAW_W'] = '251'
    if bad == 'unknown_env': env['GX1_NATIVE_V3_ACTUAL_STEP_OBSERVER_EXTRA'] = 'false'
    if bad == 'legacy_env': env['GX1_NATIVE_V3_ISSUE_MIX'] = 'true'
    for key, value in env.items(): monkeypatch.setenv(key, value)
    return dict(f, path=path, digest=digest, plan=plan, output=output, lane=lane,
        call=lambda: g.require_native_v3_actual_step_observer_execution(plan_sha256=digest))


@pytest.mark.parametrize('lane', ('reference', 'cold'))
@pytest.mark.parametrize('preflight', (False, True))
def test_actual_step_observer_exact_role_guard_has_resource_authority_only(monkeypatch, tmp_path, lane, preflight):
    f = actual_step_observer_fixture(monkeypatch, tmp_path, lane, preflight); proof = f['call']()
    assert proof['lane'] == lane and proof['preflight'] is preflight
    assert proof['training_authority_granted'] is proof['optimizer_authority_granted'] is False
    assert f['called'] == ['cuda'] and not f['output'].exists()


@pytest.mark.parametrize('bad', ('schema', 'profile', 'execution', 'final', 'producer', 'metadata', 'model',
    'cursor', 'boolean_cursor', 'observer', 'initializer', 'checkpoint_cursors', 'updates', 'autograd',
    'records', 'bool_budget', 'core_bridge', 'guard_bridge', 'owner', 'parents', 'bool_gate', 'spent',
    'flags', 'old_route', 'old_cpu_route', 'parent_lane', 'parent_protocol', 'snapshot', 'existing',
    'symlink', 'argv', 'python', 'hardware_env', 'unknown_env', 'legacy_env'))
def test_actual_step_observer_guard_rejects_unqualified_or_mixed_execution(monkeypatch, tmp_path, bad):
    f = actual_step_observer_fixture(monkeypatch, tmp_path, bad=bad)
    with pytest.raises(RuntimeError): f['call']()
    assert not f['called'] and not (f['output'] / 'CLAIM.json').exists()


def test_actual_step_observer_preflight_rejects_boolean_zero_gate(monkeypatch, tmp_path):
    f = actual_step_observer_fixture(monkeypatch, tmp_path, preflight=True, bad='bool_gate')
    with pytest.raises(RuntimeError, match='ACTUAL_STEP_OBSERVER_TYPED_LANES'): f['call']()
    assert not f['called']


@pytest.mark.parametrize('lane', ('reference', 'cold'))
def test_actual_step_observer_guard_requests_only_actual_wrapper_exports(monkeypatch, tmp_path, lane):
    f = actual_step_observer_fixture(monkeypatch, tmp_path, lane, preflight=True)
    wrapper = (Path(__file__).resolve().parents[1] / 'scripts/gx1_capped_run.sh').read_text()
    exported = set(re.findall(r'--setenv=([A-Z0-9_]+)=', wrapper)); requested = set()
    class ActualWrapperEnvironment(dict):
        def get(self, key, default=None):
            requested.add(key); return super().get(key, default)
    actual = {k: v for k, v in os.environ.items() if not k.startswith('GX1_') or k in exported}
    monkeypatch.setattr(os, 'environ', ActualWrapperEnvironment(actual)); f['call']()
    assert requested <= exported and f['called'] == ['cuda']


@pytest.mark.parametrize('lane', ('reference', 'cold'))
def test_actual_step_observer_shell_binds_exact_two_role_plans_and_existing_caps(monkeypatch, tmp_path, lane):
    f = actual_step_observer_fixture(monkeypatch, tmp_path, lane)
    variables, functions, dispatch, _ = [v[2] for v in g._V3_ACTUAL_STEP_OBSERVER_SHELL_ADDITIONS]
    functions = functions.replace('/home/andre2/GX1_RUNS', str(f['runs']))
    state = dict(REPO_ROOT=str(f['repo']), CANONICAL_TRAINER_PYTHON=str(f['repo'] / '.venv/bin/python'),
        JOB_CLASS='trainer', MEM='20G', SWAP='512M', ATTENDED_SMOKE='false', CUDA_PRODUCER_GUARD='false',
        POWER_BENCHMARK_SCOPE_JSON='', POWER_BENCHMARK_SCOPE_SHA256='')
    script = 'set -eu\n' + ''.join(k + '=' + shlex.quote(v) + '\n' for k, v in state.items()) + variables + functions
    script += 'validate_target_command() {\n' + dispatch + 'return 88\n}\n'
    argv = [state['CANONICAL_TRAINER_PYTHON'], '-m', 'gx1.audit.native_future_state_step_observer_qualify_v1',
        '--v3-actual-step-observer-plan-sha256', f['digest']]
    for extra in ([], ['--preflight'], ['--lane', lane]):
        tail = 'validate_target_command ' + ' '.join(map(shlex.quote, argv + extra))
        tail += '\nprintf "%s,%s,%s,%s,%s\\n" "$NATIVE_V3_ACTUAL_STEP_OBSERVER" "$NATIVE_V3_ACTUAL_STEP_OBSERVER_LANE" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_OUT_BUNDLE_DIR"\n'
        result = subprocess.run(['bash', '-c', script + tail], cwd=f['repo'], capture_output=True, text=True)
        assert result.returncode == (75 if extra == ['--lane', lane] else 0), result.stderr
        if result.returncode == 0: assert result.stdout.strip() == 'true,' + lane + ',1800,1770,' + str(f['output'])


def _actual_step_observer_guard_owners():
    repo = Path(g.__file__).resolve().parents[2]
    return {n: hashlib.sha256(subprocess.check_output(['git', 'show', g._V3_ACTUAL_STEP_OBSERVER_PARENT + ':' + n], cwd=repo)).hexdigest() for n in OWNERS}


def test_actual_step_observer_guard_bridge_preserves_all_original_guard_and_shell_bytes():
    proof = g.native_v3_actual_step_observer_guard_source_bridge(_actual_step_observer_guard_owners())
    assert proof['old_guard_routes_preserved'] and proof['old_numeric_resource_environment_function_unchanged']


@pytest.mark.parametrize('bad', ('legacy_resource', 'new_shell_limit', 'producer_hash'))
def test_actual_step_observer_guard_bridge_rejects_unreviewed_delta(monkeypatch, bad):
    owners = _actual_step_observer_guard_owners()
    if bad == 'producer_hash': owners[OWNERS[0]] = '0' * 64
    else:
        original = Path.read_text; repo = Path(g.__file__).resolve().parents[2]
        def changed(path, *a, **kw):
            text = original(path, *a, **kw)
            if path == repo / OWNERS[0] and bad == 'legacy_resource':
                text = text.replace('_MAX_MEMORY_BYTES = 20 * 1024**3', '_MAX_MEMORY_BYTES = 21 * 1024**3', 1)
            if path == repo / OWNERS[1] and bad == 'new_shell_limit':
                fragment = g._V3_ACTUAL_STEP_OBSERVER_SHELL_ADDITIONS[1][2]
                text = text.replace(fragment, fragment.replace('TRAINER_GPU_MAX_POWER_DRAW_W=250', 'TRAINER_GPU_MAX_POWER_DRAW_W=251'), 1)
            return text
        monkeypatch.setattr(Path, 'read_text', changed)
    with pytest.raises(RuntimeError): g.native_v3_actual_step_observer_guard_source_bridge(owners)


def actual_step_observer_cold_recovery_fixture(monkeypatch, tmp_path, preflight=False, bad=None):
    f = actual_step_observer_fixture(monkeypatch, tmp_path, 'cold', preflight=True)
    repo, runs = f['repo'], f['runs']; cold = copy.deepcopy(f['plan'])
    evidence = runs / 'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'
    cold_ref = dict(path=str(repo / 'configs/research' / (g._V3_ACTUAL_STEP_OBSERVER_NAME + '_COLD.json')),
        sha256='cb14e5d4cdc9d65b76a689bf9994cf5528388571771318b496362922d45d23ac')
    reference_ref = dict(path=str(repo / 'configs/research' / (g._V3_ACTUAL_STEP_OBSERVER_NAME + '_REFERENCE.json')),
        sha256='c578814baea374e9ab11e7b9ea6dbd6ce0f76f1f550e4321c249f5ee117aa78f')
    cold['parents']['reference']['plan'] = reference_ref
    root = runs / (g._V3_ACTUAL_STEP_OBSERVER_NAME + '_reference')
    c433 = dict(path=str(root / 'CURSOR433.json'), sha256='2549dc79fbe7d4312f2e10a5b687053e0dcca1d9bd9bde2275021d6f8cf6412e')
    c434 = dict(path=str(root / 'CURSOR434.json'), sha256='dd20636570a9d00d0e2dee04ef71ac36b7db7da722624e99e80afb2a753bd050')
    cold['inputs'] = dict(reference_cursor1_receipt=c433, reference_cursor2_receipt=c434,
        reference_checkpoint=dict(path=str(root / 'CURSOR433.pt'), sha256='1' * 64))
    design_ref = dict(path=str(evidence / 'V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_DESIGN_20261128_001.json'),
        sha256='a4e157293c9339b2afbc4ef2bbf34718c832d932bead198398fc0a636b32b173')
    design = dict(original_failed_COLD={'plan': cold_ref}, reference={'plan': reference_ref},
        reference_C433_receipt=c433, reference_C434_receipt=c434)
    output = runs / g._V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_NAME
    plan = copy.deepcopy(cold); plan['metadata']['original_cold'] = cold
    adapter = 'gx1/audit/native_future_state_step_observer_cold_recovery_v1.py'
    (repo / adapter).write_text('# source-only recovery fixture\n')
    plan['source_owners'][adapter] = hashlib.sha256((repo / adapter).read_bytes()).hexdigest()
    bridge = dict(producer_commit=g._V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_PARENT,
        owners={n: dict(original_sha256='f' * 64, current_sha256=plan['source_owners'][n]) for n in OWNERS})
    plan.update(schema_version='gx1_native_v3_actual_step_observer_cold_recovery_plan_v1',
        stage=g._V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_STAGE, output_root=str(output), source_commit='a' * 40,
        execution_profile=g._V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_PROFILE,
        recovery_design=design_ref, original_cold_plan=cold_ref, reference_plan=reference_ref,
        guard_source_bridge=bridge)
    if bad == 'schema': plan['schema_version'] = 'old'
    if bad == 'profile': plan['execution_profile'] = 'generic'
    if bad == 'execution': plan['execution']['GPU_draw_stop_W'] = 251
    if bad == 'old_cold_plan': plan['original_cold_plan'] = {}
    if bad == 'reference_plan': plan['reference_plan'] = {}
    if bad == 'design': plan['recovery_design'] = {}
    if bad == 'old_metadata': plan['metadata']['original_cold'] = {}
    if bad == 'trajectory': plan['metadata']['trajectory'] = {}
    if bad == 'recipe': plan['recipe'] = {}
    if bad == 'parents': plan['parents'] = {}
    if bad == 'cursor': plan['starting_cursor'] = 432
    if bad == 'boolean_cursor': plan['starting_cursor'] = True
    if bad == 'observer': plan['observer_enabled'] = False
    if bad == 'initializer': plan['technical_weights_never_initializer'] = False
    if bad == 'updates': plan['limits']['optimizer_steps'] = 2
    if bad == 'autograd': plan['limits']['autograd_calls'] = 1
    if bad == 'records': plan['limits']['observer_records'] = 2
    if bad == 'bool_budget': plan['limits']['model_constructions'] = True
    if bad == 'checkpoint': plan['inputs']['reference_checkpoint'] = {}
    if bad == 'C434': plan['inputs']['reference_cursor2_receipt'] = {}
    if bad == 'owner': plan['source_owners'][adapter] = '0' * 64
    if bad == 'math_owner': plan['source_owners'][g._V3_ACTUAL_STEP_OBSERVER_CORE] = '0' * 64
    if bad == 'bridge': plan['guard_source_bridge'] = {**bridge, 'extra': True}
    path = repo / 'configs/research' / (g._V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_NAME + '.json')
    path.write_text(json.dumps(plan)); digest = hashlib.sha256(path.read_bytes()).hexdigest()
    policy_path = repo / 'NEXT_RUN_POLICY.json'; policy = json.loads(policy_path.read_text())
    for flag in ('training_enabled', 'native_training_enabled'): policy[flag] = not preflight
    old = policy[g._V3_ACTUAL_STEP_OBSERVER_STAGE]
    old.update(execution_authorized=False, training_authorized=False, remaining_invocations=0, relaunch_allowed=False, active_lane=None)
    old['lane_gates'] = {lane: dict(status='CLOSED_CONSUMED_COMPLETE' if lane == 'reference' else 'CLOSED_CONSUMED_FAILED_CONSUMED',
        plan=reference_ref if lane == 'reference' else cold_ref, execution_authorized=False,
        training_authorized=False, test_authorized=False, relaunch_allowed=False, remaining_invocations=0) for lane in ('reference', 'cold')}
    parent = dict(status='PREPARED_CLOSED_NOT_ADMITTED' if preflight else 'ADMITTED_EXACT_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY',
        source_preparation_authorized=True, design=design_ref, plan=dict(path=str(path), sha256=digest),
        execution_authorized=not preflight, remaining_invocations=int(not preflight), relaunch_allowed=False,
        training_authorized=not preflight, test_authorized=False, spending_authorized=False)
    policy[g._V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_STAGE] = parent
    if bad == 'bool_gate': parent['remaining_invocations'] = not preflight
    if bad == 'spent': parent['remaining_invocations'] = 0
    if bad == 'flags': policy['native_launch_authorized'] = True
    if bad == 'old_cold_open': old['lane_gates']['cold']['execution_authorized'] = True
    if bad == 'old_cold_status': old['lane_gates']['cold']['status'] = 'CLOSED_CONSUMED_COMPLETE'
    if bad == 'old_reference_status': old['lane_gates']['reference']['status'] = 'CLOSED_CONSUMED_FAILED_CONSUMED'
    if bad == 'old_boolean_gate': old['lane_gates']['cold']['remaining_invocations'] = False
    if bad == 'old_aggregation': policy[TRAIN_EVAL_GAP_NAME.lower()]['aggregation_completion']['remaining_invocations'] = 1
    policy_path.write_text(json.dumps(policy))
    if not preflight:
        snapshot = evidence / 'V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_ADMITTED_POLICY_20261128_001.json'
        snapshot.write_bytes(b'{}' if bad == 'snapshot' else policy_path.read_bytes())
    if bad == 'existing': output.mkdir()
    if bad == 'symlink':
        other = path.with_suffix('.original'); path.rename(other); path.symlink_to(other)
    read = g._native_v3_development_read
    def fixture_read(candidate, candidate_digest=None):
        for binding, value in ((cold_ref, cold), (design_ref, design)):
            if str(candidate) == binding['path']:
                assert candidate_digest == binding['sha256']; return value
        return read(candidate, candidate_digest)
    monkeypatch.setattr(g, '_native_v3_development_read', fixture_read)
    monkeypatch.setattr(g, 'native_v3_actual_step_observer_cold_recovery_guard_source_bridge', lambda *a, **k: bridge)
    argv = [str(repo / adapter), '--v3-actual-step-observer-cold-recovery-plan-sha256', digest] + (['--preflight'] if preflight else [])
    if bad == 'argv': argv += ['--lane', 'cold']
    monkeypatch.setattr(sys, 'argv', argv)
    if bad == 'python': monkeypatch.setattr(sys, 'executable', '/usr/bin/python3')
    for key in list(os.environ):
        if key.startswith('GX1_NATIVE_'): monkeypatch.delenv(key)
    env = dict(GX1_NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY='true',
        GX1_NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_PLAN_SHA256=digest)
    if bad == 'hardware_env': env['GX1_TRAINER_GPU_MAX_POWER_DRAW_W'] = '251'
    if bad == 'legacy_env': env['GX1_NATIVE_V3_ACTUAL_STEP_OBSERVER'] = 'true'
    if bad == 'unknown_env': env['GX1_NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_EXTRA'] = 'false'
    for key, value in env.items(): monkeypatch.setenv(key, value)
    return dict(f, path=path, digest=digest, plan=plan, output=output,
        call=lambda: g.require_native_v3_actual_step_observer_cold_recovery_execution(plan_sha256=digest))


@pytest.mark.parametrize('preflight', (False, True))
def test_actual_step_observer_cold_recovery_has_only_exact_resource_authority(monkeypatch, tmp_path, preflight):
    f = actual_step_observer_cold_recovery_fixture(monkeypatch, tmp_path, preflight); proof = f['call']()
    assert proof['preflight'] is preflight and proof['lane'] == 'cold'
    assert proof['training_authority_granted'] is proof['optimizer_authority_granted'] is False
    assert f['called'] == ['cuda'] and not f['output'].exists()


@pytest.mark.parametrize('bad', ('schema', 'profile', 'execution', 'old_cold_plan', 'reference_plan', 'design',
    'old_metadata', 'trajectory', 'recipe', 'parents', 'cursor', 'boolean_cursor', 'observer', 'initializer',
    'updates', 'autograd', 'records', 'bool_budget', 'checkpoint', 'C434', 'owner', 'math_owner', 'bridge',
    'bool_gate', 'spent', 'flags', 'old_cold_open', 'old_cold_status', 'old_reference_status', 'old_boolean_gate',
    'old_aggregation', 'snapshot', 'existing', 'symlink', 'argv', 'python', 'hardware_env', 'legacy_env', 'unknown_env'))
def test_actual_step_observer_cold_recovery_rejects_scope_or_source_relaxation(monkeypatch, tmp_path, bad):
    f = actual_step_observer_cold_recovery_fixture(monkeypatch, tmp_path, bad=bad)
    with pytest.raises(RuntimeError): f['call']()
    assert not f['called'] and not (f['output'] / 'CLAIM.json').exists()


def test_actual_step_observer_cold_recovery_preflight_rejects_boolean_zero_gate(monkeypatch, tmp_path):
    f = actual_step_observer_cold_recovery_fixture(monkeypatch, tmp_path, preflight=True, bad='bool_gate')
    with pytest.raises(RuntimeError, match='OBSERVER_COLD_RECOVERY_EXACT_ONE_USE_GATE'): f['call']()
    assert not f['called']


def test_actual_step_observer_cold_recovery_shell_exact_cli_and_unchanged_caps(monkeypatch, tmp_path):
    f = actual_step_observer_cold_recovery_fixture(monkeypatch, tmp_path)
    variables, functions, dispatch, _ = [v[2] for v in g._V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_SHELL_ADDITIONS]
    functions = functions.replace('/home/andre2/GX1_RUNS', str(f['runs']))
    state = dict(REPO_ROOT=str(f['repo']), CANONICAL_TRAINER_PYTHON=str(f['repo'] / '.venv/bin/python'),
        JOB_CLASS='trainer', MEM='20G', SWAP='512M', ATTENDED_SMOKE='false', CUDA_PRODUCER_GUARD='false',
        POWER_BENCHMARK_SCOPE_JSON='', POWER_BENCHMARK_SCOPE_SHA256='')
    script = 'set -eu\n' + ''.join(k + '=' + shlex.quote(v) + '\n' for k, v in state.items()) + variables + functions
    script += 'validate_target_command() {\n' + dispatch + 'return 88\n}\n'
    argv = [state['CANONICAL_TRAINER_PYTHON'], '-m', 'gx1.audit.native_future_state_step_observer_cold_recovery_v1',
        '--v3-actual-step-observer-cold-recovery-plan-sha256', f['digest']]
    for extra in ([], ['--preflight'], ['--lane', 'cold']):
        tail = 'validate_target_command ' + ' '.join(map(shlex.quote, argv + extra))
        tail += '\nprintf "%s,%s,%s,%s\\n" "$NATIVE_V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_OUT_BUNDLE_DIR"\n'
        result = subprocess.run(['bash', '-c', script + tail], cwd=f['repo'], capture_output=True, text=True)
        assert result.returncode == (75 if extra == ['--lane', 'cold'] else 0), result.stderr
        if result.returncode == 0: assert result.stdout.strip() == 'true,1800,1770,' + str(f['output'])


def test_actual_step_observer_cold_recovery_guard_requests_only_actual_wrapper_exports(monkeypatch, tmp_path):
    f = actual_step_observer_cold_recovery_fixture(monkeypatch, tmp_path, preflight=True)
    wrapper = (Path(__file__).resolve().parents[1] / 'scripts/gx1_capped_run.sh').read_text()
    exported = set(re.findall(r'--setenv=([A-Z0-9_]+)=', wrapper)); requested = set()
    class ActualWrapperEnvironment(dict):
        def get(self, key, default=None):
            requested.add(key); return super().get(key, default)
    actual = {k: v for k, v in os.environ.items() if not k.startswith('GX1_') or k in exported}
    monkeypatch.setattr(os, 'environ', ActualWrapperEnvironment(actual)); f['call']()
    assert requested <= exported and f['called'] == ['cuda']


def _actual_step_observer_cold_recovery_guard_owners():
    repo = Path(g.__file__).resolve().parents[2]
    return {n: hashlib.sha256(subprocess.check_output(['git', 'show', g._V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_PARENT + ':' + n], cwd=repo)).hexdigest() for n in OWNERS}


def test_actual_step_observer_cold_recovery_bridge_preserves_qualifier_math_and_old_guard_bytes():
    proof = g.native_v3_actual_step_observer_cold_recovery_guard_source_bridge(_actual_step_observer_cold_recovery_guard_owners())
    assert proof['old_guard_routes_preserved'] and proof['old_numeric_resource_environment_function_unchanged']
    assert len(proof['old_qualifier_and_math_byte_unchanged']) == 6


@pytest.mark.parametrize('bad', ('legacy_resource', 'new_shell_limit', 'producer_hash', 'math_owner'))
def test_actual_step_observer_cold_recovery_bridge_rejects_unreviewed_delta(monkeypatch, bad):
    owners = _actual_step_observer_cold_recovery_guard_owners(); repo = Path(g.__file__).resolve().parents[2]
    if bad == 'producer_hash': owners[OWNERS[0]] = '0' * 64
    elif bad == 'math_owner':
        original = Path.read_bytes
        monkeypatch.setattr(Path, 'read_bytes', lambda p, *a, **k: original(p, *a, **k) + (b'\n# mutation\n' if p == repo / g._V3_ACTUAL_STEP_OBSERVER_CORE else b''))
    else:
        original = Path.read_text
        def changed(path, *a, **kw):
            text = original(path, *a, **kw)
            if path == repo / OWNERS[0] and bad == 'legacy_resource':
                text = text.replace('_MAX_MEMORY_BYTES = 20 * 1024**3', '_MAX_MEMORY_BYTES = 21 * 1024**3', 1)
            if path == repo / OWNERS[1] and bad == 'new_shell_limit':
                fragment = g._V3_ACTUAL_STEP_OBSERVER_COLD_RECOVERY_SHELL_ADDITIONS[1][2]
                text = text.replace(fragment, fragment.replace('TRAINER_GPU_MAX_POWER_DRAW_W=250', 'TRAINER_GPU_MAX_POWER_DRAW_W=251'), 1)
            return text
        monkeypatch.setattr(Path, 'read_text', changed)
    with pytest.raises(RuntimeError): g.native_v3_actual_step_observer_cold_recovery_guard_source_bridge(owners)
