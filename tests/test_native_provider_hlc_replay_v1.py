"""Tiny replay admission/report fixtures; no native model, payload or CUDA work."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from gx1.audit import native_provider_hlc_replay_v1 as m


def initial_fixture():
 binding={'path':'/fixture/INITIAL_PLAN.json','sha256':'b'*64};model='c'*64;target='d'*64;state='e'*64
 base={'model_contract':{'contract_sha256':model},'metadata':{'target_contract':{'contract_sha256':target}}}
 result={'status':'COMPLETE_NATIVE_PROVIDER_HLC_WHOLE_TRAIN_INITIAL','source_policy_inputs_boot_unchanged':True,'plan':binding,'report':{'whole_TRAIN_initialbaseline_qualified':True,'initial':{'rows':354550,'initial_prior_exact':True},'counts':{'optimizer_steps':0},'initial_state_sha256':state}}
 acceptance={'full_initialbaseline_qualified':True,'result':m.BASE_RESULT}
 receipt={'plan':binding,'model_state_sha256':state,'model_contract_sha256':model,'target_contract_sha256':target,'initial_only':True,'optimizer_updates':0}
 return acceptance,result,base,receipt


def test_exact_full_initial_metadata_accepts():assert m.require_initial_metadata(*initial_fixture())is True


@pytest.mark.parametrize('change',['partial','updated','prior','contract','foreign_plan','not_initial'])
def test_initial_metadata_rejects_unqualified_or_foreign_reference(change):
 a,r,b,i=initial_fixture()
 if change=='partial':r['report']['initial']['rows']=90112
 elif change=='updated':r['report']['counts']['optimizer_steps']=1
 elif change=='prior':r['report']['initial']['initial_prior_exact']=False
 elif change=='contract':i['target_contract_sha256']='f'*64
 elif change=='foreign_plan':i['plan']={'path':'/foreign','sha256':'a'*64}
 elif change=='not_initial':i['initial_only']=False
 with pytest.raises(ValueError):m.require_initial_metadata(a,r,b,i)


@pytest.mark.parametrize('admitted',[False,True])
def test_flags_bind_only_technical_update_scope(admitted):
 p={k:False for k in m.FLAGS}
 if admitted:p.update(training_enabled=True,native_training_enabled=True)
 m.validate_flags(p,admitted)
 for key in m.FLAGS:
  wrong=dict(p);wrong[key]=not p[key]
  with pytest.raises(ValueError,match='EXACT_TECHNICAL_TRAINING_FLAGS'):m.validate_flags(wrong,admitted)


def test_clock_prefix_is_fixed_and_target_independent(monkeypatch):
 monkeypatch.setattr(m,'ROWS',288);source=np.arange(1000,1288,dtype=np.int64);issues=np.arange(288,dtype=np.int64)*300000000000
 c={'source_rows':source,'truth':{'issue_ns':issues}};schedule={'whole_issue_ns_sha256':m.core.digest(issues),'whole_source_rows_sha256':m.core.digest(source),'batch_source_rows_sha256':[m.core.digest(source[j*96:(j+1)*96])for j in range(3)],'batch_issue_ns_sha256':[m.core.digest(issues[j*96:(j+1)*96])for j in range(3)]}
 batches=m.prefix_batches(c,{'schedule':schedule});assert np.array_equal(np.concatenate(batches),np.arange(288))
 schedule['batch_source_rows_sha256'][1]='0'*64
 with pytest.raises(ValueError,match='CLOCK_PREFIX_BINDING'):m.prefix_batches(c,{'schedule':schedule})


def warm_fixture(monkeypatch,tmp_path):
 roots={k:tmp_path/k for k in ('warm','cold')};monkeypatch.setattr(m,'ROOTS',roots);binding={'path':'/fixture/replayplan','sha256':'a'*64};policy={'path':'/fixture/policy','sha256':'b'*64}
 plan={'inputs':{},'model_contract':{'fixture':True},'schedule':{'logical_batches':3,'batches_per_pass':3,'update_flags':[True,True,True]},'recipe':{'fixed':True},'metadata':{'target_contract':{'fixture':'v2'},'truth_sha256':'c'*64,'runtime':{'gpu':{'uuid':'fixture'},'library_versions':{'fixture':'1'}}},'source_owners':{}}
 def ref(path):return {'path':str(path),'sha256':'d'*64}
 def proof(cursor):
  p={k:'e'*64 for k in ('trajectory','position','model_state','optimizer_state','scheduler_state','rng_state','training','optimizer_parameter_names','steps','scores')};p['trajectory']=m.core.state_sha(m.trajectory(plan));p['position']=m.core.state_sha(m.train.position(plan['schedule'],cursor));return p
 root=roots['warm'];identity={'pid':99,'boot_id':'fixture','start_ticks':10};claim={'plan':binding,'policy_at_claim':policy,'lane':'warm','process_identity':identity}
 receipt={'original_plan':binding,'execution_plan':binding,'producer_lane':'warm','logical_cursor':2,'optimizer_updates':2,'trajectory_sha256':m.trajectory(plan)['contract_sha256'],'checkpoint':ref(root/'CURSOR002.pt'),'proof':proof(2)}
 result={'status':'COMPLETE_TECHNICAL_WARM_PREFIX','plan':binding,'claim':ref(root/'CLAIM.json'),'process_identity':identity,'source_inputs_policy_boot_unchanged':True,'report':{'counts':copy.deepcopy(m.LIMITS['warm']),'trajectory':m.trajectory(plan),'cursor2_receipt':ref(root/'CURSOR002.json'),'cursor3_proof':ref(root/'CURSOR003_PROOF.json')}}
 terminal={'status':'COMPLETE','exit_code':0,'plan':binding,'result':ref(root/'RESULT.json'),'claim':ref(root/'CLAIM.json')};reference={'proof':proof(3),'nonzero_moments':{'backbone':3,'readout':2}}
 values={str(root/'TERMINAL.json'):terminal,str(root/'CLAIM.json'):claim,str(root/'RESULT.json'):result,str(root/'CURSOR002.json'):receipt,str(root/'CURSOR003_PROOF.json'):reference}
 monkeypatch.setattr(m,'ref',ref);monkeypatch.setattr(m,'read',lambda b:values[b['path']]);monkeypatch.setattr(m.core,'future_cpu_parent_stopped',lambda ident:None)
 return binding,policy,plan,terminal,claim,result,receipt,reference


def test_actual_warm_chain_checks_both_proofs_before_decode(monkeypatch,tmp_path):
 b,p,plan,*_=warm_fixture(monkeypatch,tmp_path);r,receipt,reference=m.warm_chain(b,p,plan);assert receipt['logical_cursor']==2 and reference['proof']['position']==m.core.state_sha(m.train.position(plan['schedule'],3))


@pytest.mark.parametrize('change',['policy','incomplete','foreign_receipt','bad_cp2proof','wrong_cp3cursor','unbound_cp3path','missing_body_moments','process_alive'])
def test_cold_preclaim_rejects_bad_parent_or_proof(monkeypatch,tmp_path,change):
 b,p,plan,t,c,r,receipt,reference=warm_fixture(monkeypatch,tmp_path)
 if change=='policy':c['policy_at_claim']={'path':'/different','sha256':'f'*64}
 elif change=='incomplete':t['status']='FAILED_CONSUMED'
 elif change=='foreign_receipt':receipt['original_plan']={'path':'/foreign','sha256':'f'*64}
 elif change=='bad_cp2proof':receipt['proof'].pop('optimizer_state')
 elif change=='wrong_cp3cursor':reference['proof']['position']=m.core.state_sha(m.train.position(plan['schedule'],2))
 elif change=='unbound_cp3path':r['report']['cursor3_proof']['path']='/foreign/proof'
 elif change=='missing_body_moments':reference['nonzero_moments']['backbone']=0
 elif change=='process_alive':monkeypatch.setattr(m.core,'future_cpu_parent_stopped',lambda _:(_ for _ in ()).throw(ValueError('alive')))
 with pytest.raises(ValueError):m.warm_chain(b,p,plan)


@pytest.mark.parametrize('fail',[False,True])
def test_run_success_and_failure_reporting_without_payload_or_model(monkeypatch,tmp_path,fail):
 from gx1.contracts import gx1_capped_execution_v1 as guard
 planfile=tmp_path/'PLAN.json';policyfile=tmp_path/'POLICY.json';planfile.write_text('{}');policyfile.write_text('{}');monkeypatch.setattr(m,'PLAN',planfile);monkeypatch.setattr(m,'POLICY',policyfile);monkeypatch.setattr(m,'ROOTS',{'warm':tmp_path/'warm','cold':tmp_path/'cold'})
 binding=m.ref(planfile);pref=m.ref(policyfile);plan={'inputs':{'feature_archive':{'path':str(tmp_path/'features.npz')},'target_arrays':{'path':str(tmp_path/'targets.npz')}}};live={'physical_windows_boot_id':1,'wsl_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
 monkeypatch.setattr(guard,'require_native_provider_hlc_replay_execution',lambda **kw:{'plan':binding,'output_root':str(m.ROOTS['warm'])});monkeypatch.setattr(m,'inspect',lambda *a:(plan,binding,pref,live));monkeypatch.setattr(m,'check_sources_inputs',lambda p:None);monkeypatch.setattr(m.core,'physical_windows_boot_id',lambda:1);monkeypatch.setattr(m.core,'future_cpu_process_identity',lambda:{'pid':0});monkeypatch.setattr(m.core.common,'firewall_state',lambda:{'denied_attempts':0});monkeypatch.setattr(m.sys,'addaudithook',lambda _:None);monkeypatch.setattr(m.signal,'signal',lambda *a:None);monkeypatch.setattr(m.signal,'setitimer',lambda *a:None)
 def physical(*args):
  if fail:raise ValueError('fixture_failure_before_any_model')
  return {'fixture_model_calls':0}
 monkeypatch.setattr(m,'physical',physical)
 if fail:
  with pytest.raises(ValueError,match='fixture_failure'):m.run(binding['sha256'],'warm')
 else:assert m.run(binding['sha256'],'warm')['result']['path']==str(m.ROOTS['warm']/'RESULT.json')
 terminal=json.loads((m.ROOTS['warm']/'TERMINAL.json').read_text());assert terminal['plan']==binding and terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False
 assert terminal['status']==('FAILED_CONSUMED'if fail else'COMPLETE');assert (m.ROOTS['warm']/'RESULT.json').exists()is(not fail)
