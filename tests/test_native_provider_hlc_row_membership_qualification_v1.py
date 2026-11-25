"""Synthetic admission and callback-order checks; no native model or payload."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from gx1.audit import native_provider_hlc_row_membership_qualification_v1 as m


def binding(name):return {'path':'/synthetic/'+name,'sha256':'a'*64}


@pytest.fixture
def packet_chain(monkeypatch):
    a={'schema_version':'gx1_source_row_membership_packet_parity_acceptance_v1','packet_parity_qualified':True,'native_model_admitted':False,'training_authorized':False,'test_authorized':False,'result':binding('result.json'),'terminal':binding('terminal.json'),'design':binding('design.json'),'clock_schedule':binding('clock.json')}
    report={'in_memory_41_arrays_bookends_unchanged':True,'all9_feature_keys_exact_dtype_shape_stride_bytes':True,'all14_truth_fields_and_certificate_metadata_exact':True,'clock_schedule':a['clock_schedule']}
    result={'status':'COMPLETE_SOURCE_ROW_MEMBERSHIP_PACKET_PARITY','design':a['design'],'all_bookends_unchanged':True,'report':report}
    terminal={'status':'COMPLETE','exit_code':0,'design':a['design'],'result':a['result'],'remaining_invocations':0,'relaunch_allowed':False}
    acceptance=binding('acceptance.json');schedule={'contract_sha256':'b'*64}
    values={acceptance['path']:a,a['result']['path']:result,a['terminal']['path']:terminal,a['clock_schedule']['path']:schedule}
    monkeypatch.setattr(m,'reference',lambda ref:Path(ref['path']))
    monkeypatch.setattr(m,'read',lambda ref:copy.deepcopy(values[ref['path']]))
    seen=[];monkeypatch.setattr(m.order,'validate_schedule',lambda value:seen.append(value)or value)
    return acceptance,a,result,terminal,schedule,seen


def test_authenticated_complete_packet_chain_validates_the_full_schedule(packet_chain):
    ref,a,result,terminal,schedule,seen=packet_chain
    actual=m.packet_metadata(ref)
    assert actual==(a,result,a['clock_schedule'],schedule)
    assert seen==[schedule]


@pytest.mark.parametrize('where,key,value',[
    ('a','native_model_admitted',True),('a','packet_parity_qualified',False),('a','training_authorized',True),
    ('a','schema_version','old_wholebatch_acceptance'),('a','clock_schedule',binding('foreign-clock.json')),
    ('result','status','PARTIAL'),('result','all_bookends_unchanged',False),
    ('report','in_memory_41_arrays_bookends_unchanged',False),('report','all9_feature_keys_exact_dtype_shape_stride_bytes',False),
    ('report','all14_truth_fields_and_certificate_metadata_exact',False),
    ('terminal','status','FAILED_CONSUMED'),('terminal','exit_code',1),('terminal','design',binding('foreign-design.json')),
    ('terminal','result',binding('foreign-result.json')),('terminal','remaining_invocations',1),('terminal','relaunch_allowed',True)])
def test_partial_or_foreign_packet_evidence_never_becomes_native_eligibility(packet_chain,where,key,value):
    ref,a,result,terminal,_,_=packet_chain
    target={'a':a,'result':result,'report':result['report'],'terminal':terminal}[where]
    target[key]=value
    with pytest.raises(ValueError):m.packet_metadata(ref)


@pytest.mark.parametrize('reject',[False,True])
def test_actual_packet_binding_callback_runs_before_any_training_update(reject):
    tree=ast.parse(Path(m.__file__).read_text())
    physical=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='physical')
    step=next(n for n in physical.body if isinstance(n,ast.FunctionDef)and n.name=='step')
    events=[];steps=[];record={'schema_version':'synthetic_membership_binding'};packet={'truth':'synthetic'}
    def make_packet(context,take):events.append('packet');return packet
    def guard(*args):
        events.append('binding')
        if reject:raise ValueError('bad source/issue/truth/support before update')
        return record
    def update(*args):events.append('training');return {'skip_update':False,'logical_step':1}
    def prefix(schedule,entries,cursor):events.append('prefix');assert entries==steps and cursor==1
    namespace={'adapter':SimpleNamespace(packet=make_packet),'order':SimpleNamespace(step_binding=guard,validate_prefix=prefix),'train':SimpleNamespace(training_step=update),'context':{},'batches':[[1,9,20]],'plan':{'schedule':{}},'activity':None,'steps':steps,'model':None,'optimizer':None,'counts':{},'need':m.need}
    exec(compile(ast.Module(body=[step],type_ignores=[]),str(m.__file__),'exec'),namespace)
    if reject:
        with pytest.raises(ValueError,match='before update'):namespace['step'](0)
        assert events==['packet','binding']and steps==[]
    else:
        assert namespace['step'](0)==packet
        assert events==['packet','binding','training','prefix']
        assert steps[0]['order_binding']==record


def test_minimal_counts_do_not_mutate_or_requalify_old_replay_probes():
    assert m.LIMITS['warm']['model_forwards']==4 and m.LIMITS['cold']['model_forwards']==2
    assert m.LIMITS['warm']['optimizer_steps']==3 and m.LIMITS['cold']['optimizer_steps']==1
    assert m.LIMITS['warm']['backward_calls']==3 and m.LIMITS['cold']['backward_calls']==1
    assert m.LIMITS['warm']['autograd_calls']==m.LIMITS['cold']['autograd_calls']==0
    assert m.replay.LIMITS['warm']['model_forwards']==9 and m.replay.LIMITS['warm']['autograd_calls']==3
    assert m.LIMITS['warm']['order_array_loads']==m.LIMITS['cold']['order_array_loads']==1


def test_each_gate_is_technical_prefix_only_and_oneuse():
    for admitted in(False,True):
        gate=m.gate(binding('plan.json'),admitted)
        assert gate['main_fit_authorized']is gate['full_epoch_training_allowed']is gate['test_authorized']is False
        assert gate['relaunch_allowed']is False
        assert set(gate['lanes'])=={'warm','cold'}
        assert all(v=={'execution_authorized':admitted,'remaining_invocations':int(admitted)}for v in gate['lanes'].values())


def test_trajectory_contains_whole_membership_schedule_descriptor_and_prefix():
    plan={'inputs':{'clock_schedule':binding('clock.json')},'model_contract':{'contract_sha256':'c'*64},'schedule':{'descriptor':{'order':binding('order.npy')},'logical_batches':3694},'technical_prefix':m.PREFIX,'recipe':{'seed':20260911},'metadata':{'target_contract':{},'truth_sha256':'d'*64,'runtime':{'gpu':'g','library_versions':{}}},'source_owners':{'owner.py':'e'*64}}
    trajectory=m.trajectory(plan)
    assert trajectory['schedule']==plan['schedule']and trajectory['technical_prefix']==m.PREFIX
    mutated=copy.deepcopy(plan);mutated['schedule']['descriptor']['order']=binding('foreign-order.npy')
    assert m.trajectory(mutated)['contract_sha256']!=trajectory['contract_sha256']


def test_current_shell_bridge_preserves_old_numeric_and_common_guard_bytes():
    names={'scripts/gx1_capped_run.sh','gx1/contracts/gx1_capped_execution_v1.py','gx1/audit/native_provider_hlc_replay_v1.py','gx1/audit/native_future_state_training_v1.py'}
    proof=m.source_bridge(names)
    assert proof['old_numeric_and_shared_guard_bytes_unchanged']is True
    assert set(proof['old_owner_sha256'])==names


def test_shell_bridge_rejects_an_undeclared_or_modified_fragment(monkeypatch):
    additions=list(m.SHELL_ADDITIONS);anchor,position,fragment=additions[0]
    additions[0]=(anchor,position,fragment+'# unregistered\n')
    monkeypatch.setattr(m,'SHELL_ADDITIONS',tuple(additions))
    with pytest.raises(ValueError,match='EXACT_SHELL_INSERTION'):m.source_bridge({'scripts/gx1_capped_run.sh'})


@pytest.mark.parametrize('lane',['warm','cold'])
def test_preflight_creates_no_claim_and_invokes_no_physical_path(monkeypatch,capsys,lane):
    seen=[];monkeypatch.setattr(m.sys,'argv',['qualification','--membership-qualification-plan-sha256','a'*64,'--lane',lane,'--preflight'])
    monkeypatch.setattr(m,'require_scope',lambda sha,lane,**kw:seen.append(('scope',sha,lane,kw)))
    monkeypatch.setattr(m,'inspect',lambda sha,lane,admitted:seen.append(('inspect',sha,lane,admitted))or({},binding('plan.json'),binding('policy.json'),{}))
    monkeypatch.setattr(m,'run',lambda *_:pytest.fail('preflight must not run'))
    m.main();result=json.loads(capsys.readouterr().out)
    assert result['array_payloads_decoded']==result['models']==result['optimizer_updates']==0
    assert result['claim_created']is False
    assert seen==[('scope','a'*64,lane,{'preflight':True}),('inspect','a'*64,lane,False)]
