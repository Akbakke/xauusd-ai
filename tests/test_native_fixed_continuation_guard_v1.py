"""Reusable one-run binding tests; synthetic JSON/source only, no payload/NN."""
import copy
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from gx1.contracts import gx1_capped_execution_v1 as g

_spec = importlib.util.spec_from_file_location('gx1_passing_frozen864_guard_fixtures',
    Path(__file__).with_name('test_native_frozen_final864_mode_gap_guard_v1.py'))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
OWNERS = _base.OWNERS
NEW = ('gx1/audit/native_future_state_fixed_continuation_v1.py',
    'tests/test_native_future_state_fixed_continuation_v1.py', 'tests/test_native_fixed_continuation_guard_v1.py')
RUN = 'NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_003'


def fixture(monkeypatch, tmp_path, preflight=False, bad=None, run_id=RUN):
    f = _base.fixture(monkeypatch, tmp_path, preflight=True)
    repo, runs = f['repo'], f['runs']; source_plan = copy.deepcopy(f['plan'])
    source_ref = dict(path=str(repo / 'configs/research' / (g._V3_FROZEN_FINAL864_MODE_GAP_NAME + '.json')),
        sha256='0739ca250b1629a5dac9400aff0a4ce1ee16626a23e1a6ccd4863e1ebd9cd388')
    parent_ref = source_plan['producer_plan']; original_ref = source_plan['original_producer_plan']
    parent_plan = copy.deepcopy(g._native_v3_development_read(parent_ref['path'], parent_ref['sha256']))
    original = copy.deepcopy(g._native_v3_development_read(original_ref['path'], original_ref['sha256']))
    receipt_ref = source_plan['authentic_final_receipt']; checkpoint_ref = source_plan['authentic_final_checkpoint']
    receipt = copy.deepcopy(g._native_v3_development_read(receipt_ref['path'], receipt_ref['sha256']))
    owners = dict(source_plan['source_owners']); anchored = {n: (repo / n).read_bytes() for n in owners}
    for name in NEW:
        p = repo / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# synthetic fixed execution fixture\n')
        owners[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    def command(argv, **kwargs):
        return anchored[argv[2].split(':', 1)[1]] if argv[:2] == ['git', 'show'] else 'a' * 40 + '\n'
    monkeypatch.setattr(subprocess, 'check_output', command)
    date, sequence = run_id.rsplit('_', 2)[1:]; stage = run_id.lower()
    evidence = runs / 'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'
    design_ref = dict(path=str(evidence / ('V3_OBSERVED_TRAIN_CONTINUATION_DESIGN_' + date + '_' + sequence + '.json')),
        sha256='6d68837c399baa06732f281cd7b7ad20aaed069cafa842cf1413538eb74dd70b')
    alias = dict(actual_before='BASELINE864', actual_candidate='FINAL1080', numeric_owner_phase='PASS2')
    scope = dict(starting_global_cursor=864, locked_final_global_cursor=1080, new_optimizer_updates=216,
        training_forwards=216, fixed_terminal_EVAL_forwards=81, total_model_forwards=297, observer_records=216,
        selected_TRAIN_rows=20727, source_day_buckets=1377, batch_rows=96, evaluation_batch_rows=256,
        extra_observer_forwards_or_backwards=0, new_autograd_diagnostics=0, new_prior_or_normalization_fits=0,
        genuine_families=8, timeframes=['M5','M15','H1','H4','D1'], years=[2021,2022,2023,2024,2025],
        D1_macro_preserved=True, TEST='SEALED', backbone_gradient_policy=None)
    coverage = dict(producer_plan=parent_ref, producer_result=dict(path=str(evidence / 'RESULT.json'), sha256='1'*64),
        producer_terminal=dict(path=str(evidence / 'TERMINAL.json'), sha256='2'*64),
        producer_acceptance=dict(path=str(evidence / 'ACCEPTANCE.json'), sha256='3'*64),
        authentic_start_FINAL864_receipt=receipt_ref, authentic_start_FINAL864_checkpoint=checkpoint_ref,
        BASELINE864_score=dict(path=str(evidence / 'SCORE.json'), sha256='4'*64),
        BASELINE864_numeric_owner_score=dict(path=str(evidence / 'NUMERIC.json'), sha256='5'*64),
        BASELINE864_logits=dict(path=str(evidence / 'LOGITS.npz'), sha256='6'*64))
    design = dict(schema_version='gx1_observed_train_continuation_source_design_v3', experiment_id=run_id, stage=stage,
        execution_owner=dict(module='gx1.audit.native_future_state_fixed_continuation_v1',
            CLI=['--v3-fixed-continuation-run-id', run_id, '--v3-fixed-continuation-plan-sha256', 'EXACT_PLAN_SHA256']),
        scope=scope, checkpoint_cursors=[865,1080], phase_alias=alias, limits=copy.deepcopy(parent_plan['limits']),
        original_producer_plan=original_ref, **coverage)
    source = parent_plan['schedule']; old = original['schedule']; schedule = copy.deepcopy(source); schedule.pop('contract_sha256')
    schedule.update(schema_version='gx1_native_v3_fixed_continuation_schedule_v1', logical_batches=1080,
        optimizer_updates=1080, passes=source['passes']+1, skipped_batches=0, original_prefix_schedule_sha256=source['contract_sha256'],
        original_take_indices_appended=list(range(216,432)), order_mode='repeat_locked_original_second_pass_once')
    for key in ('cohort_takes','logical_step_rows','cohort_take_state_sha256_by_logical_batch','update_flags'):
        schedule[key] = copy.deepcopy(source[key] + old[key][216:432])
    for key in ('raw_permutation_state_sha256_by_pass','actual_sorted_partition_stream_state_sha256_by_pass'):
        schedule[key] = source[key] + [old[key][1]]
    schedule['contract_sha256'] = '9'*64
    prefix = 'fixed_continuation_' + date + '_' + sequence + '_'
    inputs = copy.deepcopy(parent_plan['inputs']); inputs[prefix+'design'] = design_ref
    bridge = dict(producer_commit=g._V3_FIXED_CONTINUATION_PARENT,
        owners={n:dict(original_sha256='f'*64,current_sha256=owners[n])for n in OWNERS})
    path = repo / 'configs/research' / (run_id+'.json'); output = runs / run_id
    plan = dict(schema_version='gx1_native_v3_fixed_continuation_plan_v1', run_id=run_id, stage=stage, source_commit='a'*40,
        output_root=str(output), execution=g._native_v3_development_execution('warm'), execution_profile=g._V3_FIXED_CONTINUATION_PROFILE,
        design=design_ref, producer_plan=parent_ref, original_producer_plan=original_ref,
        authentic_final_receipt=receipt_ref, authentic_final_checkpoint=checkpoint_ref,
        metadata=dict(parent_plan=parent_ref,original_plan=original_ref,receipt=copy.deepcopy(receipt),
            trajectory=dict(schedule=schedule,original_prefix_trajectory_sha256=parent_plan['metadata']['trajectory']['contract_sha256'])),
        recipe=copy.deepcopy(parent_plan['recipe']),model_contract=copy.deepcopy(parent_plan['model_contract']),
        physical_input_identity=copy.deepcopy(parent_plan['physical_input_identity']),backbone_gradient_policy=None,
        starting_cursor=864,ending_cursor=1080,checkpoint_cursors=[865,1080],observer_enabled=True,phase_alias=alias,
        schedule=schedule,limits=copy.deepcopy(parent_plan['limits']),inputs=inputs,input_prefix=prefix,
        source_reference_plan=source_ref,source_owners=owners,guard_source_bridge=bridge)
    if bad=='schema':plan['schema_version']='old'
    if bad=='profile':plan['execution_profile']='generic'
    if bad=='execution':plan['execution']['GPU_draw_stop_W']=251
    if bad=='run_id':plan['run_id']=run_id+'_other'
    if bad=='arbitrary_root':plan['output_root']='/tmp/other'
    if bad=='design_id':design['experiment_id']=run_id+'_other'
    if bad=='design_CLI':design['execution_owner']['CLI']+=['--epochs','2']
    if bad=='cohort':scope['selected_TRAIN_rows']=20728
    if bad=='adaptive_end':scope['locked_final_global_cursor']=1081
    if bad=='old648_initializer':plan['authentic_final_checkpoint']=parent_plan['authentic_final_checkpoint']
    if bad=='old_input':inputs['recovery_design']=design_ref
    if bad=='extra_input':inputs['arbitrary_new_role']=design_ref
    if bad=='prefix':plan['input_prefix']='other_'
    if bad=='order':schedule['cohort_takes'][864][0]+=1
    if bad=='bool_take':schedule['cohort_takes'][864][0]=False
    if bad=='extra_update':plan['limits']['optimizer_steps']+=1
    if bad=='extra_forward':plan['limits']['model_forwards']+=1
    if bad=='prior_fit':plan['limits']['prior_fits']=1
    if bad=='phase':plan['phase_alias']=dict(alias,actual_candidate='PASS2')
    if bad=='bridge':plan['guard_source_bridge']={**bridge,'extra':True}
    if bad=='source':owners[NEW[0]]='0'*64
    if bad=='source_reference':plan['source_reference_plan']={}
    path.write_text(json.dumps(plan));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path=evidence/'SYNTHETIC_FIXED_PARENT_MANIFEST.json'
    members=list(coverage.values());members=members[:-1]if bad=='backup_coverage'else members
    manifest_path.write_text(json.dumps({'files':members}));manifest_ref=dict(path=str(manifest_path),sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    backup_path=evidence/'SYNTHETIC_FIXED_PARENT_BACKUP.json'
    backup_path.write_text(json.dumps(dict(F_FULLFSYNC=bad!='backup_unverified',separate_mac_disk=True,
        independent_member_readback=True,manifest=manifest_ref,manifest_sha256=manifest_ref['sha256'])))
    backup_ref=dict(path=str(backup_path),sha256=hashlib.sha256(backup_path.read_bytes()).hexdigest())
    pp=repo/'NEXT_RUN_POLICY.json';policy=json.loads(pp.read_text())
    for flag in _base.FLAGS:policy[flag]=not preflight and flag in _base.FLAGS[:2]
    previous=policy[parent_name:=Path(parent_ref['path']).stem.lower()]
    previous.update(status='CLOSED_CONSUMED_COMPLETE',plan=parent_ref,execution_authorized=False,training_authorized=False,
        remaining_invocations=0,relaunch_allowed=False,final_backup=backup_ref)
    diagnostic=policy[g._V3_FROZEN_FINAL864_MODE_GAP_STAGE]
    diagnostic.update(status='CLOSED_CONSUMED_COMPLETE',plan=source_ref,execution_authorized=False,training_authorized=False,
        remaining_invocations=0,relaunch_allowed=False)
    if bad=='old_open':previous['execution_authorized']=True
    if bad=='old_boolean':previous['remaining_invocations']=False
    if bad=='diagnostic_failed':diagnostic['status']='FAILED_CONSUMED'
    if bad=='backup_missing':previous.pop('final_backup')
    parent=dict(status='PREPARED_CLOSED_NOT_ADMITTED'if preflight else'ADMITTED_EXACT_V3_FIXED_CONTINUATION',
        source_preparation_authorized=True,design=design_ref,plan=dict(path=str(path),sha256=digest),
        execution_authorized=not preflight,training_authorized=not preflight,remaining_invocations=int(not preflight),
        relaunch_allowed=False,test_authorized=False,spending_authorized=False)
    policy[stage]=parent
    if bad=='flags':policy['native_launch_authorized']=True
    if bad=='bool_gate':parent['remaining_invocations']=not preflight
    if bad=='spent':parent['remaining_invocations']=0
    pp.write_text(json.dumps(policy))
    if not preflight:
        (evidence/('V3_OBSERVED_TRAIN_CONTINUATION_ADMITTED_POLICY_'+date+'_'+sequence+'.json')).write_bytes(b'{}'if bad=='snapshot'else pp.read_bytes())
    if bad=='existing':output.mkdir()
    read=g._native_v3_development_read
    def adapted(candidate,candidate_digest=None):
        for binding,value in((source_ref,source_plan),(parent_ref,parent_plan),(original_ref,original),(receipt_ref,receipt),(design_ref,design)):
            if str(candidate)==binding['path']:
                assert candidate_digest==binding['sha256'];return value
        return read(candidate,candidate_digest)
    monkeypatch.setattr(g,'_native_v3_development_read',adapted)
    monkeypatch.setattr(g,'native_v3_fixed_continuation_guard_source_bridge',lambda*a,**k:bridge)
    argv=[str(repo/NEW[0]),'--v3-fixed-continuation-run-id',run_id,'--v3-fixed-continuation-plan-sha256',digest]+(['--preflight']if preflight else[])
    if bad=='argv':argv+=['--epochs','2']
    monkeypatch.setattr(sys,'argv',argv)
    if bad=='python':monkeypatch.setattr(sys,'executable','/usr/bin/python3')
    for key in list(os.environ):
        if key.startswith('GX1_NATIVE_'):monkeypatch.delenv(key)
    env=dict(GX1_NATIVE_V3_FIXED_CONTINUATION='true',GX1_NATIVE_V3_FIXED_CONTINUATION_RUN_ID=run_id,GX1_NATIVE_V3_FIXED_CONTINUATION_PLAN_SHA256=digest)
    if bad=='hardware_env':env['GX1_TRAINER_GPU_MAX_POWER_DRAW_W']='251'
    if bad=='legacy_env':env['GX1_NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_002']='true'
    if bad=='unknown_env':env['GX1_NATIVE_V3_FIXED_CONTINUATION_EXTRA']='false'
    for key,value in env.items():monkeypatch.setenv(key,value)
    return dict(f,path=path,digest=digest,plan=plan,output=output,run_id=run_id,
        call=lambda:g.require_native_v3_fixed_continuation_execution(run_id=run_id,plan_sha256=digest))


@pytest.mark.parametrize('preflight',(False,True))
@pytest.mark.parametrize('run_id',(RUN,'NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261201_004'))
def test_fixed_run_requires_its_own_explicit_canonical_design_and_one_use_gate(monkeypatch,tmp_path,preflight,run_id):
    f=fixture(monkeypatch,tmp_path,preflight,run_id=run_id);proof=f['call']()
    assert proof['run_id']==run_id and proof['preflight']is preflight
    assert proof['training_authority_granted']is proof['optimizer_authority_granted']is False
    assert f['called']==['cuda']and not f['output'].exists()


@pytest.mark.parametrize('bad',('schema','profile','execution','run_id','arbitrary_root','design_id','design_CLI','cohort',
    'adaptive_end','old648_initializer','old_input','extra_input','prefix','order','bool_take','extra_update','extra_forward',
    'prior_fit','phase','bridge','source','source_reference','old_open','old_boolean','diagnostic_failed','backup_missing',
    'backup_coverage','backup_unverified','flags','bool_gate','spent','snapshot','existing','argv','python','hardware_env','legacy_env','unknown_env'))
def test_fixed_run_rejects_unbound_options_paths_scope_or_authority(monkeypatch,tmp_path,bad):
    f=fixture(monkeypatch,tmp_path,bad=bad)
    with pytest.raises(RuntimeError):f['call']()
    assert not f['called']and not(f['output']/'CLAIM.json').exists()


@pytest.mark.parametrize('run_id',('../escape','NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20260230_003',
    'NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_000','native_v3_observed_train_continuation_20261128_003'))
def test_fixed_run_id_is_calendar_valid_canonical_and_not_a_path(run_id):
    with pytest.raises(RuntimeError):g.native_fixed_continuation_run_paths(run_id)


def test_fixed_shell_uses_only_two_exact_binding_flags_and_actual_module(monkeypatch,tmp_path):
    f=fixture(monkeypatch,tmp_path)
    variables,functions,dispatch,_=[v[2]for v in g._V3_FIXED_CONTINUATION_SHELL_ADDITIONS]
    functions=functions.replace('/home/andre2/GX1_RUNS',str(f['runs']))
    state=dict(REPO_ROOT=str(f['repo']),CANONICAL_TRAINER_PYTHON=str(f['repo']/'.venv/bin/python'),
        JOB_CLASS='trainer',MEM='20G',SWAP='512M',ATTENDED_SMOKE='false',CUDA_PRODUCER_GUARD='false',
        POWER_BENCHMARK_SCOPE_JSON='',POWER_BENCHMARK_SCOPE_SHA256='')
    script='set -eu\n'+''.join(k+'='+shlex.quote(v)+'\n'for k,v in state.items())+variables+functions
    script+='validate_target_command() {\n'+dispatch+'return 88\n}\n'
    for module,extra,expected in(('native_future_state_fixed_continuation_v1',[],0),
        ('native_future_state_fixed_continuation_v1',['--preflight'],0),('native_future_state_observed_train_continuation_002_v1',[],75),
        ('native_future_state_fixed_continuation_v1',['--epochs','2'],75)):
        argv=[state['CANONICAL_TRAINER_PYTHON'],'-m','gx1.audit.'+module,'--v3-fixed-continuation-run-id',RUN,
            '--v3-fixed-continuation-plan-sha256',f['digest']]+extra
        tail='validate_target_command '+' '.join(map(shlex.quote,argv))
        tail+='\nprintf "%s,%s,%s,%s,%s\\n" "$NATIVE_V3_FIXED_CONTINUATION" "$NATIVE_V3_FIXED_CONTINUATION_RUN_ID" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_OUT_BUNDLE_DIR"\n'
        result=subprocess.run(['bash','-c',script+tail],cwd=f['repo'],capture_output=True,text=True)
        assert result.returncode==expected,result.stderr
        if expected==0:assert result.stdout.strip()=='true,'+RUN+',1800,1770,'+str(f['output'])


def _guard_hashes():
    repo=Path(g.__file__).resolve().parents[2]
    return{n:hashlib.sha256(subprocess.check_output(['git','show',g._V3_FIXED_CONTINUATION_PARENT+':'+n],cwd=repo)).hexdigest()for n in OWNERS}


def test_fixed_bridge_preserves_all92_old_source_owners_and_caps():
    proof=g.native_v3_fixed_continuation_guard_source_bridge(_guard_hashes())
    assert proof['old_guard_routes_preserved']and proof['old_numeric_resource_environment_function_unchanged']
    assert len(proof['old_92_source_owners_byte_unchanged'])==92


@pytest.mark.parametrize('bad',('producer_hash','legacy_limit','route_limit','old_adapter'))
def test_fixed_bridge_rejects_unreviewed_old_source_changes(monkeypatch,bad):
    hashes=_guard_hashes();repo=Path(g.__file__).resolve().parents[2]
    if bad=='producer_hash':hashes[OWNERS[0]]='0'*64
    elif bad=='old_adapter':
        read=Path.read_bytes;target=repo/'gx1/audit/native_future_state_frozen_final864_mode_gap_v1.py'
        monkeypatch.setattr(Path,'read_bytes',lambda p,*a,**k:read(p,*a,**k)+(b'\n# mutation\n'if p==target else b''))
    else:
        read=Path.read_text
        def changed(path,*a,**kw):
            text=read(path,*a,**kw)
            if path==repo/OWNERS[0]and bad=='legacy_limit':text=text.replace('_MAX_MEMORY_BYTES = 20 * 1024**3','_MAX_MEMORY_BYTES = 21 * 1024**3',1)
            if path==repo/OWNERS[1]and bad=='route_limit':
                fragment=g._V3_FIXED_CONTINUATION_SHELL_ADDITIONS[1][2]
                text=text.replace(fragment,fragment.replace('TRAINER_GPU_MAX_POWER_DRAW_W=250','TRAINER_GPU_MAX_POWER_DRAW_W=251'),1)
            return text
        monkeypatch.setattr(Path,'read_text',changed)
    with pytest.raises(RuntimeError):g.native_v3_fixed_continuation_guard_source_bridge(hashes)
