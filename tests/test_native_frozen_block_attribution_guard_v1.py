"""Frozen2x2 exact read-only gate tests: synthetic metadata, no payload/NN."""
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

_spec = importlib.util.spec_from_file_location('gx1_passing_fixed_continuation_guard_fixtures',
    Path(__file__).with_name('test_native_fixed_continuation_guard_v1.py'))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
OWNERS = _base.OWNERS
FLAGS = _base._base.FLAGS
NEW = ('gx1/audit/native_future_state_frozen_block_attribution_v1.py',
    'tests/test_native_future_state_frozen_block_attribution_v1.py',
    'tests/test_native_frozen_block_attribution_guard_v1.py')


def fixture(monkeypatch, tmp_path, preflight=False, bad=None):
    f = _base.fixture(monkeypatch, tmp_path, preflight=True)
    repo, runs = f['repo'], f['runs']; source = copy.deepcopy(f['plan'])
    evidence = runs / 'GX1_NATIVE_TWENTY_STEP_QUALIFICATION_20261123_001'
    source_ref = dict(path=str(repo / 'configs/research/NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_004.json'),
        sha256='20ce68f848c0e48a5c81d3e1684ee2ab4dadd7c13c439ef904615c83d5085734')
    original_ref = source['original_producer_plan']
    original = copy.deepcopy(g._native_v3_development_read(original_ref['path'], original_ref['sha256']))
    parent_refs = {'864': source['producer_plan'], '1080': source_ref}
    parents = {'864': copy.deepcopy(g._native_v3_development_read(parent_refs['864']['path'], parent_refs['864']['sha256'])), '1080': source}
    source['metadata']['trajectory']['contract_sha256'] = 'd' * 64
    receipt_refs = {'864': source['authentic_final_receipt'], '1080': dict(
        path=str(runs / 'NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_004/FINAL1080.json'),
        sha256='f4928b08a1ccd890079902ef00bda818a054f7a3aba5fb1a5fcb899099ae593c')}
    checkpoint_refs = {'864': source['authentic_final_checkpoint'], '1080': dict(
        path=str(runs / 'NATIVE_V3_OBSERVED_TRAIN_CONTINUATION_20261128_004/FINAL1080.pt'),
        sha256='dde0148c0b7bde965aa6cb47f516a2445d7c7a0936333e8c10bb45c6ae25be44')}
    proof_keys = ('model_state','optimizer_parameter_names','optimizer_state','position','rng_state',
        'scheduler_state','scores','steps','training','trajectory')
    receipts = {}
    for cursor in (864,1080):
        key = str(cursor)
        receipts[key] = dict(logical_cursor=cursor,optimizer_updates=cursor,producer_lane='continuation',
            original_plan=parent_refs[key],execution_plan=parent_refs[key],checkpoint=checkpoint_refs[key],
            trajectory_sha256=parents[key]['metadata']['trajectory']['contract_sha256'],
            proof=dict.fromkeys(proof_keys,'b'*64))
    owners = dict(source['source_owners']); anchored = {n:(repo/n).read_bytes()for n in owners}
    for name in NEW:
        p=repo/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# synthetic frozen block fixture\n')
        owners[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    def command(argv,**kwargs):
        return anchored[argv[2].split(':',1)[1]]if argv[:2]==['git','show']else'a'*40+'\n'
    monkeypatch.setattr(subprocess,'check_output',command)
    design_ref=dict(path=str(evidence/'V3_FROZEN_864_1080_BLOCK_ATTRIBUTION_DESIGN_20261128_001.json'),
        sha256='9578cee76b7d0676529bf78a48f90c9d8a6f68bd5e13a914ce9c08c47578d2dd')
    scope=dict(rows=20727,source_day_buckets=1377,batch_rows=256,passes=4,forwards_per_pass=81,total_model_forwards=324,
        optimizer_steps=0,backward_calls=0,autograd_calls=0,checkpoint_writes=0,new_input_builds=0,
        new_prior_or_normalization_fits=0,genuine_families=8,timeframes=['M5','M15','H1','H4','D1'],
        years=[2021,2022,2023,2024,2025],EVAL_only=True,no_grad=True,D1_macro_preserved=True,TEST='SEALED',backbone_gradient_policy=None)
    arms=[dict(name='SELF864',backbone_cursor=864,readout_cursor=864),dict(name='SELF1080',backbone_cursor=1080,readout_cursor=1080),
        dict(name='BACKBONE864_READOUT1080',backbone_cursor=864,readout_cursor=1080),
        dict(name='BACKBONE1080_READOUT864',backbone_cursor=1080,readout_cursor=864)]
    design=dict(schema_version='gx1_native_frozen_block_attribution_source_design_v1',experiment_id=g._V3_FROZEN_BLOCK_ATTRIBUTION_NAME,
        stage=g._V3_FROZEN_BLOCK_ATTRIBUTION_STAGE,scope=scope,arms_in_fixed_order=copy.deepcopy(arms),source_reference_plan=source_ref)
    for cursor in (864,1080):
        key=str(cursor);design['producer'+key+'_plan']=parent_refs[key]
        design['authentic'+key+'_receipt']=receipt_refs[key];design['authentic'+key+'_checkpoint']=checkpoint_refs[key]
        for i,role in enumerate(('result','terminal','acceptance'),1):
            design['producer'+key+'_'+role]=dict(path=str(evidence/(key+'_'+role+'.json')),sha256=str(i)*64)
        for i,role in enumerate(('score','numeric_owner_score','logits'),4):
            design['authentic'+key+'_'+role]=dict(path=str(evidence/(key+'_'+role+'.json')),sha256=str(i)*64)
    limits=dict.fromkeys(source['limits'],0)
    limits.update(model_constructions=1,optimizer_constructions=1,feature_cache_payload_loads=1,target_archive_loads=1,
        checkpoint_loads=2,model_restores=3,model_forwards=324,backbone_forwards=324,readout_forwards=324,
        logit_archive_writes=4,logit_archive_readbacks=4,saved_eval_logit_loads=2,authentic_checkpoint_prechecks=2,block_composite_copies=2)
    inputs=copy.deepcopy(source['inputs']);inputs.update({'frozen_block_attribution_'+k:v for k,v in design.items()
        if type(v)is dict and set(v)=={'path','sha256'}})
    bridge=dict(producer_commit=g._V3_FROZEN_BLOCK_ATTRIBUTION_PARENT,
        owners={n:dict(original_sha256='f'*64,current_sha256=owners[n])for n in OWNERS})
    path=repo/'configs/research'/(g._V3_FROZEN_BLOCK_ATTRIBUTION_NAME+'.json');output=runs/g._V3_FROZEN_BLOCK_ATTRIBUTION_NAME
    plan=dict(schema_version='gx1_native_v3_frozen_block_attribution_plan_v1',stage=g._V3_FROZEN_BLOCK_ATTRIBUTION_STAGE,
        source_commit='a'*40,output_root=str(output),execution=g._native_v3_development_execution('warm'),
        execution_profile=g._V3_FROZEN_BLOCK_ATTRIBUTION_PROFILE,design=design_ref,source_reference_plan=source_ref,
        original_producer_plan=original_ref,producer_plans=parent_refs,authentic_receipts=receipt_refs,authentic_checkpoints=checkpoint_refs,
        metadata=dict(original_plan=original_ref,endpoints={key:dict(parent_plan=parent_refs[key],receipt=copy.deepcopy(receipts[key]),scores={})
            for key in ('864','1080')}),recipe=copy.deepcopy(source['recipe']),model_contract=copy.deepcopy(source['model_contract']),
        physical_input_identity=copy.deepcopy(source['physical_input_identity']),backbone_gradient_policy=None,
        scope=copy.deepcopy(scope),arms=arms,numeric_owner_phase='PASS2',limits=limits,inputs=inputs,source_owners=owners,guard_source_bridge=bridge)
    if bad=='schema':plan['schema_version']='old'
    if bad=='profile':plan['execution_profile']='generic'
    if bad=='execution':plan['execution']['GPU_draw_stop_W']=251
    if bad=='wrong_root':plan['output_root']='/tmp/other'
    if bad=='design':plan['design']={}
    if bad=='cohort':plan['scope']['rows']=20728
    if bad=='bool_scope':plan['scope']['new_input_builds']=False
    if bad=='training_arm':plan['arms'][0]['training']=True
    if bad=='arm_order':plan['arms'].reverse()
    if bad=='float_cursor':plan['arms'][0]['backbone_cursor']=864.0
    if bad=='phase':plan['numeric_owner_phase']='FINAL1080'
    if bad=='old_initializer':plan['authentic_checkpoints']=dict(checkpoint_refs,**{'1080':checkpoint_refs['864']})
    if bad=='receipt_cursor':receipts['1080']['logical_cursor']=864
    if bad=='receipt_producer':receipts['864']['execution_plan']=parent_refs['1080']
    if bad=='receipt_full10':receipts['864']['proof'].pop('rng_state')
    if bad=='receipt_trajectory':receipts['1080']['trajectory_sha256']='0'*64
    if bad=='metadata_receipt':plan['metadata']['endpoints']['864']['receipt']={}
    if bad=='metadata_parent':plan['metadata']['endpoints']['1080']['parent_plan']=parent_refs['864']
    if bad=='model':plan['model_contract']={}
    if bad=='recipe':plan['recipe']={}
    if bad in ('optimizer_steps','backward_calls','autograd_calls','checkpoint_writes','normalization_fits','prior_fits','observer_records'):
        limits[bad]=1
    if bad=='extra_forward':limits['model_forwards']+=1
    if bad=='extra_checkpoint_load':limits['checkpoint_loads']+=1
    if bad=='missing_precheck':limits['authentic_checkpoint_prechecks']=1
    if bad=='extra_copy':limits['block_composite_copies']=3
    if bad=='extra_restore':limits['model_restores']=4
    if bad=='bool_budget':limits['model_constructions']=True
    if bad=='old_input':inputs['recovery_design']=design_ref
    if bad=='extra_input':inputs['unbound_new_role']=design_ref
    if bad=='saved_logits':inputs['frozen_block_attribution_authentic864_logits']={}
    if bad=='bridge':plan['guard_source_bridge']={**bridge,'extra':True}
    if bad=='source':owners[NEW[0]]='0'*64
    if bad=='extra_owner':owners['unowned.py']='0'*64
    if bad=='source_reference':plan['source_reference_plan']={}
    path.write_text(json.dumps(plan));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    pp=repo/'NEXT_RUN_POLICY.json';policy=json.loads(pp.read_text())
    for flag in FLAGS:policy[flag]=False
    # Close the previously prepared synthetic003 route; it has no authority here.
    policy[_base.RUN.lower()].update(execution_authorized=False,training_authorized=False,remaining_invocations=0,relaunch_allowed=False)
    for cursor in (864,1080):
        key=str(cursor);members=[parent_refs[key],receipt_refs[key],checkpoint_refs[key]]+[
            design['producer'+key+'_'+role]for role in ('result','terminal','acceptance')]+[
            design['authentic'+key+'_'+role]for role in ('score','numeric_owner_score','logits')]
        if bad=='backup_coverage'+key:members=members[:-1]
        manifest_path=evidence/('SYNTHETIC_BLOCK_'+key+'_MANIFEST.json');manifest_path.write_text(json.dumps({'files':members}))
        manifest_ref=dict(path=str(manifest_path),sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        backup_path=evidence/('SYNTHETIC_BLOCK_'+key+'_BACKUP.json');backup_path.write_text(json.dumps(dict(
            F_FULLFSYNC=bad!='backup_unverified'+key,separate_mac_disk=True,independent_member_readback=True,
            manifest=manifest_ref,manifest_sha256=manifest_ref['sha256'])))
        previous=policy.setdefault(Path(parent_refs[key]['path']).stem.lower(),{})
        previous.update(status='CLOSED_CONSUMED_COMPLETE',plan=parent_refs[key],execution_authorized=False,training_authorized=False,
            remaining_invocations=0,relaunch_allowed=False,final_backup=dict(path=str(backup_path),sha256=hashlib.sha256(backup_path.read_bytes()).hexdigest()))
        if bad=='old_open'+key:previous['execution_authorized']=True
        if bad=='old_failed'+key:previous['status']='FAILED_CONSUMED'
        if bad=='old_boolean'+key:previous['remaining_invocations']=False
        if bad=='backup_missing'+key:previous.pop('final_backup')
    parent=dict(status='PREPARED_CLOSED_NOT_ADMITTED'if preflight else'ADMITTED_EXACT_V3_FROZEN_BLOCK_ATTRIBUTION',
        source_preparation_authorized=True,design=design_ref,plan=dict(path=str(path),sha256=digest),
        execution_authorized=not preflight,training_authorized=False,remaining_invocations=int(not preflight),
        relaunch_allowed=False,test_authorized=False,spending_authorized=False)
    policy[g._V3_FROZEN_BLOCK_ATTRIBUTION_STAGE]=parent
    if bad=='training_authority':parent['training_authorized']=True
    if bad=='bool_gate':parent['remaining_invocations']=not preflight
    if bad=='spent':parent['remaining_invocations']=0
    pp.write_text(json.dumps(policy))
    if not preflight:
        (evidence/'V3_FROZEN_864_1080_BLOCK_ATTRIBUTION_ADMITTED_POLICY_20261128_001.json').write_bytes(b'{}'if bad=='snapshot'else pp.read_bytes())
    if bad=='existing':output.mkdir()
    read=g._native_v3_development_read
    def adapted(candidate,candidate_digest=None):
        bindings=[(source_ref,source),(original_ref,original),(design_ref,design)]
        bindings+=[(parent_refs[key],parents[key])for key in ('864','1080')]
        bindings+=[(receipt_refs[key],receipts[key])for key in ('864','1080')]
        for binding,value in bindings:
            if str(candidate)==binding['path']:
                assert candidate_digest==binding['sha256'];return value
        return read(candidate,candidate_digest)
    monkeypatch.setattr(g,'_native_v3_development_read',adapted)
    monkeypatch.setattr(g,'native_v3_frozen_block_attribution_guard_source_bridge',lambda*a,**k:bridge)
    argv=[str(repo/NEW[0]),'--v3-frozen-block-attribution-plan-sha256',digest]+(['--preflight']if preflight else[])
    if bad=='argv':argv+=['--mode','TRAIN']
    if bad=='duplicate_argv':argv+=['--v3-frozen-block-attribution-plan-sha256',digest]
    monkeypatch.setattr(sys,'argv',argv)
    if bad=='python':monkeypatch.setattr(sys,'executable','/usr/bin/python3')
    for key in list(os.environ):
        if key.startswith('GX1_NATIVE_'):monkeypatch.delenv(key)
    env=dict(GX1_NATIVE_V3_FROZEN_BLOCK_ATTRIBUTION='true',GX1_NATIVE_V3_FROZEN_BLOCK_ATTRIBUTION_PLAN_SHA256=digest)
    if bad=='hardware_env':env['GX1_TRAINER_GPU_MAX_POWER_DRAW_W']='251'
    if bad=='legacy_env':env['GX1_NATIVE_V3_FIXED_CONTINUATION']='true'
    if bad=='unknown_env':env['GX1_NATIVE_V3_FROZEN_BLOCK_ATTRIBUTION_EXTRA']='false'
    for key,value in env.items():monkeypatch.setenv(key,value)
    return dict(f,path=path,digest=digest,plan=plan,output=output,call=lambda:g.require_native_v3_frozen_block_attribution_execution(plan_sha256=digest))


@pytest.mark.parametrize('preflight',(False,True))
def test_frozen_block_gate_has_no_training_authority_even_when_admitted(monkeypatch,tmp_path,preflight):
    f=fixture(monkeypatch,tmp_path,preflight);proof=f['call']()
    assert proof['preflight']is preflight
    assert proof['training_authority_granted']is proof['optimizer_authority_granted']is False
    assert f['called']==['cuda']and not f['output'].exists()


@pytest.mark.parametrize('preflight',(False,True))
@pytest.mark.parametrize('flag',FLAGS)
def test_frozen_block_every_global_flag_stays_false(monkeypatch,tmp_path,preflight,flag):
    f=fixture(monkeypatch,tmp_path,preflight);pp=f['repo']/'NEXT_RUN_POLICY.json';policy=json.loads(pp.read_text());policy[flag]=True
    pp.write_text(json.dumps(policy))
    with pytest.raises(RuntimeError,match='ALL_SIX_GLOBAL_FLAGS_FALSE'):f['call']()
    assert not f['called']


@pytest.mark.parametrize('bad',('schema','profile','execution','wrong_root','design','cohort','bool_scope','training_arm','arm_order',
    'float_cursor','phase','old_initializer','receipt_cursor','receipt_producer','receipt_full10','receipt_trajectory',
    'metadata_receipt','metadata_parent','model','recipe','optimizer_steps','backward_calls','autograd_calls','checkpoint_writes',
    'normalization_fits','prior_fits','observer_records','extra_forward','extra_checkpoint_load','missing_precheck','extra_copy',
    'extra_restore','bool_budget','old_input','extra_input','saved_logits','bridge','source','extra_owner','source_reference',
    'old_open864','old_open1080','old_failed864','old_failed1080','old_boolean864','old_boolean1080',
    'backup_missing864','backup_missing1080','backup_coverage864','backup_coverage1080','backup_unverified864','backup_unverified1080',
    'training_authority','bool_gate','spent','snapshot','existing','argv','duplicate_argv','python','hardware_env','legacy_env','unknown_env'))
def test_frozen_block_rejects_changed_scope_lineage_or_authority(monkeypatch,tmp_path,bad):
    f=fixture(monkeypatch,tmp_path,bad=bad)
    with pytest.raises(RuntimeError):f['call']()
    assert not f['called']and not(f['output']/'CLAIM.json').exists()


def test_frozen_block_shell_exact_module_flags_and_caps(monkeypatch,tmp_path):
    f=fixture(monkeypatch,tmp_path)
    variables,functions,dispatch,_=[v[2]for v in g._V3_FROZEN_BLOCK_ATTRIBUTION_SHELL_ADDITIONS]
    functions=functions.replace('/home/andre2/GX1_RUNS',str(f['runs']))
    state=dict(REPO_ROOT=str(f['repo']),CANONICAL_TRAINER_PYTHON=str(f['repo']/'.venv/bin/python'),JOB_CLASS='trainer',
        MEM='20G',SWAP='512M',ATTENDED_SMOKE='false',CUDA_PRODUCER_GUARD='false',POWER_BENCHMARK_SCOPE_JSON='',POWER_BENCHMARK_SCOPE_SHA256='')
    script='set -eu\n'+''.join(k+'='+shlex.quote(v)+'\n'for k,v in state.items())+variables+functions
    script+='validate_target_command() {\n'+dispatch+'return 88\n}\n'
    for python,module,extra,expected in ((state['CANONICAL_TRAINER_PYTHON'],'native_future_state_frozen_block_attribution_v1',[],0),
        (state['CANONICAL_TRAINER_PYTHON'],'native_future_state_frozen_block_attribution_v1',['--preflight'],0),
        ('/usr/bin/python3','native_future_state_frozen_block_attribution_v1',[],75),
        (state['CANONICAL_TRAINER_PYTHON'],'native_future_state_frozen_final864_mode_gap_v1',[],75),
        (state['CANONICAL_TRAINER_PYTHON'],'native_future_state_frozen_block_attribution_v1',['--mode','TRAIN'],75),
        (state['CANONICAL_TRAINER_PYTHON'],'native_future_state_frozen_block_attribution_v1',['--v3-frozen-block-attribution-plan-sha256',f['digest']],75)):
        argv=[python,'-m','gx1.audit.'+module,'--v3-frozen-block-attribution-plan-sha256',f['digest']]+extra
        tail='validate_target_command '+' '.join(map(shlex.quote,argv))
        tail+='\nprintf "%s,%s,%s,%s,%s,%s\\n" "$NATIVE_V3_FROZEN_BLOCK_ATTRIBUTION" "$TRAINER_MAX_WALL_SECONDS" "$TRAINER_MODEL_MAX_WALL_SECONDS" "$TRAINER_GPU_MAX_POWER_DRAW_W" "$TRAINER_GPU_MAX_MEMORY_USED_MIB" "$TRAINER_OUT_BUNDLE_DIR"\n'
        result=subprocess.run(['bash','-c',script+tail],cwd=f['repo'],capture_output=True,text=True)
        assert result.returncode==expected,result.stderr
        if expected==0:assert result.stdout.strip()=='true,1800,1770,250,12288,'+str(f['output'])


def _guard_hashes():
    repo=Path(g.__file__).resolve().parents[2]
    return{n:hashlib.sha256(subprocess.check_output(['git','show',g._V3_FROZEN_BLOCK_ATTRIBUTION_PARENT+':'+n],cwd=repo)).hexdigest()for n in OWNERS}


def test_frozen_block_source_bridge_preserves_old95_owners_and_guards():
    proof=g.native_v3_frozen_block_attribution_guard_source_bridge(_guard_hashes())
    assert proof['old_guard_routes_preserved']and proof['old_numeric_resource_environment_function_unchanged']
    assert len(proof['old_95_source_owners_byte_unchanged'])==95


@pytest.mark.parametrize('bad',('producer_hash','old_cap','new_route_cap','old_adapter'))
def test_frozen_block_bridge_rejects_unreviewed_changes(monkeypatch,bad):
    hashes=_guard_hashes();repo=Path(g.__file__).resolve().parents[2]
    if bad=='producer_hash':hashes[OWNERS[0]]='0'*64
    elif bad=='old_adapter':
        read=Path.read_bytes;target=repo/'gx1/audit/native_future_state_fixed_continuation_v1.py'
        monkeypatch.setattr(Path,'read_bytes',lambda p,*a,**k:read(p,*a,**k)+(b'\n# mutation\n'if p==target else b''))
    else:
        read=Path.read_text
        def changed(path,*a,**kw):
            text=read(path,*a,**kw)
            if path==repo/OWNERS[0]and bad=='old_cap':text=text.replace('_MAX_MEMORY_BYTES = 20 * 1024**3','_MAX_MEMORY_BYTES = 21 * 1024**3',1)
            if path==repo/OWNERS[1]and bad=='new_route_cap':
                fragment=g._V3_FROZEN_BLOCK_ATTRIBUTION_SHELL_ADDITIONS[1][2]
                text=text.replace(fragment,fragment.replace('TRAINER_GPU_MAX_POWER_DRAW_W=250','TRAINER_GPU_MAX_POWER_DRAW_W=251'),1)
            return text
        monkeypatch.setattr(Path,'read_text',changed)
    with pytest.raises(RuntimeError):g.native_v3_frozen_block_attribution_guard_source_bridge(hashes)
