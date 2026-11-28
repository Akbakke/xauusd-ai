"""Focused synthetic execution-generation and boot-evidence recovery checks."""
import copy,json
from pathlib import Path
import pytest
from gx1.audit import native_future_state_step_observer_cold_recovery_v1 as r


def old_plan(tmp_path):
    cp=tmp_path/'REFERENCE_C433.pt';cp.write_bytes(b'synthetic-stat-only')
    old={'schema_version':'original','stage':'original','source_commit':'old-head','output_root':'/original/cold','lane':'cold','observer_enabled':True,'starting_cursor':433,'ending_cursor':434,'checkpoint_cursors':[434],'technical_weights_never_initializer':True,'metadata':{'original':{'recipe':'unchanged'},'producer':{'schedule':'original432'},'receipt':{'proof':'all10-final432'},'trajectory':{'schedule':{'logical_batches':434,'last_takes':[0,1]},'runtime_compatibility':{'version':'old'},'contract_sha256':'technical-original'},'runtime':{'source_commit':'old-head'}},'source_owners':{r.q.SELF:'1'*64,r.q.CORE:'2'*64,r.q.OBSERVER:'3'*64,r.o.GUARDS[0]:'4'*64,r.o.GUARDS[1]:'5'*64,r.o.DEV_GUARD_TEST:'6'*64},'inputs':{'reference_checkpoint':{'path':str(cp),'sha256':'7'*64}},'parents':{'reference':{'plan':{'path':'/original/reference.json','sha256':'8'*64}}},'protocol':{'path':'/original/protocol.json','sha256':'9'*64},'limits':{'model_forwards':1,'optimizer_steps':1,'observer_records':1},'execution':{'internal_seconds':1770},'execution_profile':'old','physical_input_identity':{'actual_metadata_contract_recomputed':True},'recipe':{'unchanged':'AdamW'},'model_contract':{'unchanged':True},'guard_source_bridge':{'old':'bridge'}}
    return old


def test_new_execution_keeps_old_C433_C434_trajectory_packet_parents_and_inputs(tmp_path):
    old=old_plan(tmp_path);before=copy.deepcopy(old);runtime={'source_commit':'new-head'};inputs={**old['inputs'],'original_cold_plan':{'path':'/original/cold.json','sha256':'a'*64},'recovery_design':r.DESIGN};plan=r.execution_plan(old,runtime,'new-head',{'new':'owner'},inputs,{'new':'guard-generation'})
    assert old==before and plan['metadata']['original_cold']==old
    for field in('lane','observer_enabled','starting_cursor','ending_cursor','checkpoint_cursors','technical_weights_never_initializer','parents','protocol','limits','execution','recipe','model_contract','physical_input_identity'):assert plan[field]==old[field]
    for field in('original','producer','receipt','trajectory'):assert plan['metadata'][field]==old['metadata'][field]
    assert plan['inputs']['reference_checkpoint']==old['inputs']['reference_checkpoint']and plan['reference_plan']==old['parents']['reference']['plan']
    assert plan['output_root']==str(r.ROOT)and plan['output_root']!=old['output_root']and plan['source_commit']=='new-head'
    runtime['source_commit']='later';assert plan['metadata']['runtime']['source_commit']=='new-head'


@pytest.mark.parametrize('changed',[None,'guard',r.q.SELF,r.q.CORE,r.q.OBSERVER])
def test_build_plan_allows_only_guard_generation_and_new_adapter_owners(monkeypatch,tmp_path,changed):
    old=old_plan(tmp_path);runtime={'source_commit':'new-head'};extra={'original_cold_plan':{'path':'/old/cold.json','sha256':'a'*64}};seen=[]
    monkeypatch.setattr(r,'original_metadata',lambda:(old,{},extra,old['physical_input_identity']))
    def ref(path):
        name=str(path.relative_to(r.REPO))if Path(path).is_relative_to(r.REPO)else str(path)
        value=old['source_owners'].get(name,'e'*64)
        if name==(r.o.GUARDS[0]if changed=='guard'else changed):value='f'*64
        return {'path':str(path),'sha256':value}
    monkeypatch.setattr(r,'ref',ref);monkeypatch.setattr(r.subprocess,'check_output',lambda args,**kwargs:'new-head\n'if args[1]=='rev-parse'else b'old-guard-source')
    monkeypatch.setattr(r.core,'future_cuda_require_runtime',lambda value:seen.append('fresh-runtime'));monkeypatch.setattr(r.train,'runtime_compatible',lambda *a,**k:seen.append('compatible'))
    monkeypatch.setattr(r.guard,'native_v3_actual_step_observer_cold_recovery_guard_source_bridge',lambda *a,**k:{'guard':'bridge'});monkeypatch.setattr(r.guard,'_native_v3_development_execution',lambda lane:old['execution'])
    if changed in(r.q.SELF,r.q.CORE,r.q.OBSERVER):
        with pytest.raises(ValueError,match='ONLY_GUARD_GENERATION'):r.build_plan(runtime)
    else:
        actual=r.build_plan(runtime);assert actual['metadata']['trajectory']==old['metadata']['trajectory']and actual['lane']=='cold'and {r.SELF,r.TEST}<=set(actual['source_owners'])
    assert seen==['fresh-runtime','compatible']


@pytest.mark.parametrize('case',['match','mismatch','query_error'])
def test_boot_bookend_keeps_actual_values_or_query_cause_and_never_passes_unknown(monkeypatch,case):
    def windows():
        if case=='query_error':raise RuntimeError('physical boot query failed exactly')
        return 'changed-windows'if case=='mismatch'else'windows'
    monkeypatch.setattr(r.core,'physical_windows_boot_id',windows);monkeypatch.setattr(Path,'read_text',lambda *a,**k:'wsl\n')
    observed=r.observe_boots();match=r.boot_matches({'physical_windows_boot_id':'windows','wsl_boot_id':'wsl'},observed)
    assert match is(case=='match')and observed['observed']['wsl_boot_id']=='wsl'
    if case=='query_error':assert observed['query_errors']['physical_windows_boot_id']=={'type':'RuntimeError','message':'physical boot query failed exactly'}
    if case=='mismatch':assert observed['observed']['physical_windows_boot_id']=='changed-windows'and not observed['query_errors']
    json.dumps(observed,allow_nan=False)


@pytest.mark.parametrize('admitted',[False,True])
def test_new_flat_gate_has_typed_one_use_and_only_technical_training_authority(admitted):
    binding={'path':'/new/recovery.json','sha256':'a'*64};value=r.gate(binding,admitted);assert r.require_gate(value,binding,admitted)==value
    value['remaining_invocations']=bool(value['remaining_invocations'])
    with pytest.raises(ValueError,match='TYPED_GATE'):r.require_gate(value,binding,admitted)


@pytest.mark.parametrize('boot_pass',[False,True])
def test_run_delegates_unchanged_physical_and_preserves_math_proof_on_failed_boot(monkeypatch,tmp_path,boot_pass):
    plan=old_plan(tmp_path);plan['original_cold_plan']={'path':'/old/failed-cold.json','sha256':'a'*64};binding={'path':'/new/plan.json','sha256':'b'*64};pref={'path':str(r.o.POLICY),'sha256':'c'*64};live={'physical_windows_boot_id':'windows','wsl_boot_id':'wsl'};published={};calls=[];report={'observer_off_on_all10_equal':True,'full_cursor434_proof':{'original':'all10'}}
    monkeypatch.setattr(r,'ROOT',tmp_path/'new-run');monkeypatch.setattr(r.guard,'require_native_v3_actual_step_observer_cold_recovery_execution',lambda **k:{'qualified':True});monkeypatch.setattr(r,'inspect',lambda *a:(plan,binding,pref,live));monkeypatch.setattr(r,'snapshot',lambda *a:{'path':'/new/snapshot.json','sha256':'c'*64});monkeypatch.setattr(r.core,'future_cpu_process_identity',lambda:{});monkeypatch.setattr(r.sys,'addaudithook',lambda *a:None);monkeypatch.setattr(r.o,'recovery_check_inputs',lambda *a,**k:None);monkeypatch.setattr(r.core.common,'firewall_state',lambda:{'denied_attempts':0})
    def physical(*args):calls.append(args);return report
    monkeypatch.setattr(r.q,'physical',physical);checks=dict.fromkeys(('plan','sources','inputs','policy','boots','HEAD'),True);checks['boots']=boot_pass;evidence={'boot_observation':{'observed':{'physical_windows_boot_id':'changed'if not boot_pass else'windows','wsl_boot_id':'wsl'},'query_errors':{}}};monkeypatch.setattr(r,'bookends',lambda *a:(checks,evidence))
    def publish(root,name,value):published[name]=copy.deepcopy(value);return {'path':str(root/name),'sha256':'d'*64}
    monkeypatch.setattr(r.o,'development_publish',publish)
    if boot_pass:assert set(r.run('b'*64))=={'result','terminal'}
    else:
        with pytest.raises(RuntimeError,match='BOOKEND_FAILED'):r.run('b'*64)
    assert len(calls)==1 and calls[0][0]is plan and published['RESULT.json']['report']==report and published['RESULT.json']['bookend_evidence']==evidence
    assert published['TERMINAL.json']['exit_code']==int(not boot_pass)and published['TERMINAL.json']['remaining_invocations']==0 and published['TERMINAL.json']['relaunch_allowed']is False


@pytest.mark.parametrize('collision',[None,'recovery_design','reference_checkpoint'])
def test_new_design_role_preserves_original_bindings_and_rejects_any_collision(collision):
    legacy={'path':'/old/FIT004-recovery-design.json','sha256':'1'*64};checkpoint={'path':'/old/REFERENCE_C433.pt','sha256':'2'*64};original={'recovery_design':legacy,'reference_checkpoint':checkpoint};before=copy.deepcopy(original);extra={'observer_cold_recovery_design':r.DESIGN}
    if collision is not None:extra[collision]={'path':'/wrong/replacement.json','sha256':'3'*64}
    if collision is not None:
        with pytest.raises(ValueError,match='INPUT_ROLE_COLLISION'):r.merge_inputs(original,extra)
    else:
        merged=r.merge_inputs(original,extra);assert merged['recovery_design']==legacy and merged['reference_checkpoint']==checkpoint and merged['observer_cold_recovery_design']==r.DESIGN
        merged['recovery_design']['path']='/later/mutation';assert original==before
    assert original==before
