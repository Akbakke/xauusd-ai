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
