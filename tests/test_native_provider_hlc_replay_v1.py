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


# Explicit None replay additions; all legacy M5 tests above remain byte-identical.
def test_all_tf_replay_preserves_legacy_source_defaults_and_narrow_main_dispatch():
 import ast,subprocess
 anchor='f33da18dd6a0e3d59e4876fa26f8e62ec5ab97b8';before=subprocess.check_output(['git','show',anchor+':'+m.SELF],cwd=m.REPO,text=True);after=Path(m.__file__).read_text()
 old_tree,tree=ast.parse(before),ast.parse(after);old={n.name:n for n in old_tree.body if isinstance(n,ast.FunctionDef)};new={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
 for name,node in old.items():
  if name!='main':assert ast.get_source_segment(before,node)==ast.get_source_segment(after,new[name])
 assignments=lambda value:[ast.dump(n,include_attributes=False)for n in value.body if isinstance(n,(ast.Assign,ast.AnnAssign))]
 assert all(value in assignments(tree)for value in assignments(old_tree))
 dispatch=new['main'].body[0];assert isinstance(dispatch,ast.If)
 assert ast.dump(dispatch.test,include_attributes=False)==ast.dump(ast.parse("sys.argv[1:2] == ['--all-tf-plan-sha256']",mode='eval').body,include_attributes=False)
 assert ast.dump(ast.Module(body=new['main'].body[1:],type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=old['main'].body,type_ignores=[]),include_attributes=False)
 old_tests=subprocess.check_output(['git','show',anchor+':'+m.TEST],cwd=m.REPO);assert Path(__file__).read_bytes().startswith(old_tests)


@pytest.mark.parametrize('admitted',[False,True])
def test_all_tf_replay_gate_authorizes_only_named_technical_lanes(admitted):
 binding={'path':'/synthetic/ALL_REPLAY_PLAN.json','sha256':'a'*64};gate=m.all_tf_gate(binding,admitted)
 assert gate['plan']==binding and gate['relaunch_allowed']is False and gate['main_fit_authorized']is False and gate['test_authorized']is False
 assert set(gate['lanes'])=={'warm','cold'}
 for value in gate['lanes'].values():assert value['execution_authorized']is admitted and value['remaining_invocations']==int(admitted)


def all_tf_initial_fixture():
 from gx1.audit import native_future_state_provider_hlc_initial_v1 as initial_owner
 binding={'path':str(m.ALL_INITIAL_PLAN),'sha256':'a'*64};result_ref={'path':str(m.ALL_INITIAL_ROOT/'RESULT.json'),'sha256':'b'*64};receipt_ref={'path':str(m.ALL_INITIAL_ROOT/'INITIAL.json'),'sha256':'c'*64};state='d'*64;target='e'*64
 base={'stage':m.ALL_INITIAL_NAME.lower(),'output_root':str(m.ALL_INITIAL_ROOT),'recipe':{'backbone_gradient_policy':None,'seed':20260911,'no_optimization':True},'model_contract':{'contract_sha256':m.ALL_CONTRACT_SHA},'metadata':{'target_contract':{'contract_sha256':target}}}
 result={'status':'COMPLETE_NATIVE_PROVIDER_HLC_ALL_TF_WHOLE_TRAIN_INITIAL','source_policy_inputs_boot_unchanged':True,'plan':binding,'report':{'whole_TRAIN_initialbaseline_qualified':True,'initial':{'rows':354550,'batch_count':1385,'initial_prior_exact':True},'counts':copy.deepcopy(initial_owner.ALL_LIMITS),'initial_state_sha256':state,'backbone_gradient_policy':None,'initial_checkpoint':receipt_ref}}
 acceptance={'full_initialbaseline_qualified':True,'result':result_ref,'initial_receipt':receipt_ref}
 receipt={'plan':binding,'checkpoint':{'path':str(m.ALL_INITIAL_ROOT/'INITIAL.pt'),'sha256':'f'*64},'model_state_sha256':state,'model_contract_sha256':m.ALL_CONTRACT_SHA,'target_contract_sha256':target,'initial_only':True,'optimizer_updates':0}
 return acceptance,result,base,receipt


def test_all_tf_replay_requires_complete_own_fresh_None_initial_metadata():
 assert m.all_tf_require_initial_metadata(*all_tf_initial_fixture())is True


@pytest.mark.parametrize('change',['old_M5_contract','old_plan','old_checkpoint','updated','partial_forward_count','truth_layout','partial_population','wrong_batch_count','target','prior'])
def test_all_tf_replay_rejects_old_policy_loaded_or_incomplete_initial_authority(change):
 a,r,b,i=all_tf_initial_fixture()
 if change=='old_M5_contract':b['model_contract']['backbone_gradient_policy']={'policy':'M5_primary'}
 elif change=='old_plan':r['plan']['path']='/synthetic/OLD_M5_INITIAL_PLAN.json'
 elif change=='old_checkpoint':i['checkpoint']['path']='/synthetic/OLD_INITIAL.pt'
 elif change=='updated':r['report']['counts']['optimizer_steps']=1
 elif change=='partial_forward_count':r['report']['counts']['model_forwards']-=1
 elif change=='truth_layout':r['report']['counts']['truth_fields_decoded']=12
 elif change=='partial_population':r['report']['initial']['rows']=96
 elif change=='wrong_batch_count':r['report']['initial']['batch_count']=1384
 elif change=='target':i['target_contract_sha256']='0'*64
 else:r['report']['initial']['initial_prior_exact']=False
 with pytest.raises(ValueError):m.all_tf_require_initial_metadata(a,r,b,i)


def all_tf_warm_fixture(monkeypatch,tmp_path):
 b,p,plan,t,c,r,receipt,reference=warm_fixture(monkeypatch,tmp_path)
 monkeypatch.setattr(m,'ALL_ROOTS',m.ROOTS)
 plan['model_contract']={'contract_sha256':m.ALL_CONTRACT_SHA};plan['recipe']['backbone_gradient_policy']=None
 r['status']='COMPLETE_ALL_TF_TECHNICAL_WARM_PREFIX';r['report']['counts']=copy.deepcopy(m.ALL_LIMITS['warm']);r['report']['trajectory']=m.trajectory(plan)
 receipt['trajectory_sha256']=m.trajectory(plan)['contract_sha256']
 for value,cursor in ((receipt['proof'],2),(reference['proof'],3)):
  value['trajectory']=m.core.state_sha(m.trajectory(plan));value['position']=m.core.state_sha(m.train.position(plan['schedule'],cursor))
 return b,p,plan,t,c,r,receipt,reference


def test_all_tf_warm_chain_requires_own_roots_None_trajectory_all10proofs_and_stopped_parent(monkeypatch,tmp_path):
 import torch
 b,p,plan,*_=all_tf_warm_fixture(monkeypatch,tmp_path)
 monkeypatch.setattr(torch,'load',lambda *a,**k:(_ for _ in ()).throw(AssertionError('NO_PRECLAIM_TENSOR_DECODE')))
 result,receipt,reference=m.all_tf_warm_chain(b,p,plan)
 assert result['status']=='COMPLETE_ALL_TF_TECHNICAL_WARM_PREFIX' and receipt['logical_cursor']==2
 assert len(receipt['proof'])==len(reference['proof'])==10 and reference['nonzero_moments']=={'backbone':3,'readout':2}


@pytest.mark.parametrize('change',['old_M5_trajectory','old_checkpoint_path','policy','missing_proof','wrong_step3','missing_moments','alive'])
def test_all_tf_cold_preclaim_rejects_cross_arm_or_unqualified_warm_parent(monkeypatch,tmp_path,change):
 b,p,plan,t,c,r,receipt,reference=all_tf_warm_fixture(monkeypatch,tmp_path)
 if change=='old_M5_trajectory':receipt['proof']['trajectory']='0'*64
 elif change=='old_checkpoint_path':receipt['checkpoint']['path']='/home/andre2/GX1_RUNS/'+m.NAME+'_warm/CURSOR002.pt'
 elif change=='policy':c['policy_at_claim']={'path':'/foreign/policy','sha256':'f'*64}
 elif change=='missing_proof':receipt['proof'].pop('optimizer_state')
 elif change=='wrong_step3':reference['proof']['position']=m.core.state_sha(m.train.position(plan['schedule'],2))
 elif change=='missing_moments':reference['nonzero_moments']['backbone']=0
 else:monkeypatch.setattr(m.core,'future_cpu_parent_stopped',lambda _:(_ for _ in ()).throw(ValueError('alive')))
 with pytest.raises(ValueError):m.all_tf_warm_chain(b,p,plan)


def all_tf_physical_fixture(monkeypatch,tmp_path):
 import torch
 from gx1.contracts import market_state_ta_integrated_model_v1 as model_owner
 class Leaf(torch.nn.Module):
  def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor([.1],dtype=torch.float32))
  def forward(self,value):return value*self.weight
 class Candidate(torch.nn.Module):
  def __init__(self):
   super().__init__();self.backbone=Leaf();self.readout=Leaf();self.register_buffer('future_state_model_sha256',torch.tensor(list(bytes.fromhex(m.ALL_CONTRACT_SHA)),dtype=torch.uint8))
  @property
  def research_objective(self):return {'contract_sha256':m.ALL_CONTRACT_SHA}
  def forward(self,value):return self.readout(self.backbone(value))
 roots={lane:tmp_path/lane for lane in ('warm','cold')}
 for root in roots.values():root.mkdir()
 monkeypatch.setattr(m,'ALL_ROOTS',roots)
 rng={'cpu':torch.tensor([1,2],dtype=torch.uint8)};state={'backbone.weight':torch.tensor([.1]),'readout.weight':torch.tensor([.1]),'future_state_model_sha256':torch.tensor(list(bytes.fromhex(m.ALL_CONTRACT_SHA)),dtype=torch.uint8)}
 plan={'inputs':{},'model_contract':{'contract_sha256':m.ALL_CONTRACT_SHA},'source_owners':{},'metadata':{'architecture':{},'target_contract':{'fixture':'v2'},'truth_sha256':'a'*64,'runtime':{'gpu':{},'library_versions':{}}},'initial_reference':{'model_state_sha256':m.core.state_sha(state),'rng_sha256':m.core.state_sha(rng)},'schedule':{'logical_batches':3,'batches_per_pass':3,'update_flags':[True]*3},'recipe':{'seed':20260911,'optimizer':{'lr':.0001},'backbone_gradient_policy':None}}
 context={'normalization':{},'input_binding':{},'prior':{},'target_contract':{'fixture':'v2'}};batches=[np.arange(j*96,(j+1)*96,dtype=np.int64)for j in range(3)]
 binding={'path':'/synthetic/ALL_REPLAY_PLAN.json','sha256':'b'*64};policy={'path':'/synthetic/POLICY.json','sha256':'c'*64}
 models,steps,loads,restores,receipts,stored,seeds=[],[],[],[],{},{},[]
 def factory(**kwargs):
  assert kwargs['backbone_gradient_policy']is None and kwargs['device']==torch.device('cuda')
  model=Candidate();models.append(model);return model
 def packet(ctx,take):return {'features':{'value':torch.ones(1)},'truth':{'side_known':np.zeros((len(take),5),bool)}}
 def parity(model,packet,plan,counts,*,equal_tf):
  for _ in range(2):model(packet['features']['value'])
  if not equal_tf:counts['backward_calls']+=2
  return {'synthetic_parity':True}
 def probe(model,packet,plan,counts):model(packet['features']['value']);counts['autograd_calls']+=3;return {'synthetic_probe':True}
 def step(model,optimizer,packet,plan,counts,index):
  assert not packet['truth']['side_known'].any();model(packet['features']['value']);steps.append(index)
  for parameter in model.parameters():parameter.grad=torch.ones_like(parameter)
  optimizer.step();optimizer.zero_grad(set_to_none=True);counts['optimizer_steps']+=1;counts['backward_calls']+=1
  return {'logical_index':index,'skip_update':False,'backbone_data_gradient_l2':1.}
 def score(model,packet,plan):model(packet['features']['value']);return {'model_sha256':m.core.state_sha(model.state_dict())}
 def checkpoint(model,optimizer,**kwargs):
  return {'trajectory':kwargs['trajectory'],'position':m.train.position(plan['schedule'],kwargs['cursor']),'model_state':copy.deepcopy(model.state_dict()),'optimizer_state':copy.deepcopy(optimizer.state_dict()),'scheduler_state':None,'rng_state':copy.deepcopy(rng),'training':model.training,'optimizer_parameter_names':[name for name,_ in model.named_parameters()],'steps':copy.deepcopy(kwargs['steps']),'scores':copy.deepcopy(kwargs['scores'])}
 def proof(cp):return {key:m.core.state_sha(cp[key])for key in ('trajectory','position','model_state','optimizer_state','scheduler_state','rng_state','training','optimizer_parameter_names','steps','scores')}
 def publish_checkpoint(path,cp,counts,*,execution_ref):
  assert execution_ref==binding;stored[str(path)]=copy.deepcopy(cp)
  for key in ('checkpoint_writes','checkpoint_readbacks','checkpoint_receipt_writes'):counts[key]+=1
  receipt={'checkpoint':{'path':str(path),'sha256':m.core.state_sha(cp)},'proof':proof(cp)};receipts[str(path)]=receipt
  return {'path':str(path.with_suffix('.json')),'sha256':'d'*64}
 def load(path,**kwargs):
  assert path==str(roots['warm']/'CURSOR002.pt') and kwargs=={'map_location':'cpu','weights_only':True};loads.append(path);return copy.deepcopy(stored[path])
 def restore(model,optimizer,cp,**kwargs):
  assert kwargs['expected_cursor']==2 and kwargs['original_plan_ref']==binding and kwargs['expected_state_proof']==proof(cp)
  restores.append(kwargs);model.load_state_dict(cp['model_state'],strict=True);optimizer.load_state_dict(cp['optimizer_state'])
 monkeypatch.setattr(model_owner,'build_native_future_state_model',factory);monkeypatch.setattr(m.adapter,'load_context',lambda meta:context);monkeypatch.setattr(m.adapter,'packet',packet);monkeypatch.setattr(m,'prefix_batches',lambda c,p:batches)
 monkeypatch.setattr(m.core,'future_cuda_seed_and_runtime',lambda request:seeds.append(request));monkeypatch.setattr(m.core,'training_rng',lambda:copy.deepcopy(rng));monkeypatch.setattr(m.core,'future_cuda_parity_pair',parity);monkeypatch.setattr(m.core,'future_cuda_gradient_probe',probe)
 monkeypatch.setattr(m.train,'training_step',step);monkeypatch.setattr(m,'score',score);monkeypatch.setattr(m.train,'make_checkpoint',checkpoint);monkeypatch.setattr(m.train,'checkpoint_proof',proof);monkeypatch.setattr(m.train,'publish_checkpoint_receipt',publish_checkpoint);monkeypatch.setattr(m.train,'strict_restore',restore)
 monkeypatch.setattr(torch,'load',load);monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
 return plan,binding,policy,models,steps,loads,restores,receipts,stored,seeds


def test_all_tf_real_tiny_AdamW_moments_same_arm_C2_CPU_load_and_exact_step3_replay(monkeypatch,tmp_path):
 plan,binding,policy,models,steps,loads,restores,receipts,stored,seeds=all_tf_physical_fixture(monkeypatch,tmp_path)
 warm=m.all_tf_physical(plan,binding,policy,{},'warm');assert warm['counts']==m.ALL_LIMITS['warm']
 reference=m.read(warm['cursor3_proof']);receipt=receipts[str(m.ALL_ROOTS['warm']/'CURSOR002.pt')]
 monkeypatch.setattr(m,'all_tf_warm_chain',lambda *a:({'report':warm},receipt,reference))
 cold=m.all_tf_physical(plan,binding,policy,{},'cold')
 assert warm['nonzero_moments_at_cursor2']['backbone']>0 and warm['nonzero_moments_at_cursor2']['readout']>0
 assert cold['counts']==m.ALL_LIMITS['cold'] and cold['cold_replay_bit_exact']is True
 assert cold['full_cursor3_proof']==reference['proof'] and len(cold['full_cursor3_proof'])==10
 assert steps==[0,1,2,2] and len(models)==len(seeds)==2 and len(loads)==len(restores)==1
 assert cold['restored_nonzero_moments']['backbone']>0 and cold['restored_nonzero_moments']['readout']>0
 assert cold['weights_authority']=='TECHNICAL_ONLY_NEVER_MAIN_INITIALIZER_OR_SELECTION'
 assert all(not module._forward_pre_hooks for model in models for module in model.modules())


def all_tf_run_fixture(monkeypatch,tmp_path,lane='warm'):
 from gx1.contracts import gx1_capped_execution_v1 as guard
 roots={key:tmp_path/key for key in ('warm','cold')};monkeypatch.setattr(m,'ALL_ROOTS',roots)
 planfile=tmp_path/'ALL_PLAN.json';policyfile=tmp_path/'POLICY.json';planfile.write_text('{}');policyfile.write_text('{}')
 monkeypatch.setattr(m,'ALL_PLAN',planfile);monkeypatch.setattr(m,'POLICY',policyfile);binding=m.ref(planfile);pref=m.ref(policyfile)
 plan={'inputs':{'feature_archive':{'path':str(tmp_path/'FEATURES.npz')},'target_arrays':{'path':str(tmp_path/'TARGETS.npz')}}};live={'physical_windows_boot_id':1,'wsl_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
 monkeypatch.setattr(guard,'require_native_provider_hlc_all_tf_replay_execution',lambda **k:{'plan':binding,'output_root':str(roots[lane])})
 monkeypatch.setattr(m,'all_tf_inspect',lambda *a:(plan,binding,pref,live));monkeypatch.setattr(m,'all_tf_check_sources_inputs',lambda p:None)
 monkeypatch.setattr(m.core,'physical_windows_boot_id',lambda:1);monkeypatch.setattr(m.core,'future_cpu_process_identity',lambda:{'pid':0});monkeypatch.setattr(m.core.common,'firewall_state',lambda:{'denied_attempts':0})
 hooks=[];monkeypatch.setattr(m.sys,'addaudithook',hooks.append);monkeypatch.setattr(m.signal,'signal',lambda *a:None);monkeypatch.setattr(m.signal,'setitimer',lambda *a:None)
 monkeypatch.setattr(m,'all_tf_physical',lambda *a:{'synthetic':True,'weights_authority':'TECHNICAL_ONLY_NEVER_MAIN_INITIALIZER_OR_SELECTION'})
 return plan,binding,pref,hooks


@pytest.mark.parametrize('failure',[False,True])
def test_all_tf_replay_run_reports_complete_or_consumed_failure_without_fallback(monkeypatch,tmp_path,failure):
 plan,binding,pref,hooks=all_tf_run_fixture(monkeypatch,tmp_path)
 def physical(*args):
  if failure:raise ValueError('SYNTHETIC_ALL_TF_REPLAY_FAILURE')
  return {'synthetic':True,'weights_authority':'TECHNICAL_ONLY_NEVER_MAIN_INITIALIZER_OR_SELECTION'}
 monkeypatch.setattr(m,'all_tf_physical',physical)
 if failure:
  with pytest.raises(ValueError,match='SYNTHETIC_ALL_TF_REPLAY_FAILURE'):m.all_tf_run(binding['sha256'],'warm')
 else:assert m.all_tf_run(binding['sha256'],'warm')['result']['path']==str(m.ALL_ROOTS['warm']/'RESULT.json')
 terminal=json.loads((m.ALL_ROOTS['warm']/'TERMINAL.json').read_text());assert terminal['status']==('FAILED_CONSUMED'if failure else'COMPLETE')
 assert terminal['remaining_invocations']==0 and terminal['relaunch_allowed']is False and terminal['plan']==binding
 assert (m.ALL_ROOTS['warm']/'RESULT.json').exists()is(not failure)
 assert len(hooks)==1
 for value in plan['inputs'].values():hooks[0]('open',(value['path'],))
 hooks[0]('open',(str(m.ALL_ROOTS['warm']/'CURSOR002.pt'),))
 for path in ('/synthetic/OLD_M5_CURSOR002.pt','/synthetic/FINAL.pt','/synthetic/other.npz'):
  with pytest.raises(RuntimeError,match='PAYLOAD_PATH_FORBIDDEN'):hooks[0]('open',(path,))
 with pytest.raises(FileExistsError):m.all_tf_run(binding['sha256'],'warm')


def test_all_tf_replay_preflight_uses_new_lane_command_and_zero_payloads(monkeypatch,tmp_path,capsys):
 import sys,torch
 _,binding,_,hooks=all_tf_run_fixture(monkeypatch,tmp_path)
 def forbidden(*a,**k):raise AssertionError('NO_REPLAY_PAYLOAD_IN_PREFLIGHT')
 monkeypatch.setattr(torch,'load',forbidden);monkeypatch.setattr(np,'load',forbidden);monkeypatch.setattr(m,'all_tf_physical',forbidden)
 for lane in ('warm','cold'):
  monkeypatch.setattr(sys,'argv',['replay','--all-tf-plan-sha256',binding['sha256'],'--lane',lane,'--preflight']);m.main();result=json.loads(capsys.readouterr().out)
  assert result['lane']==lane and result['plan']==binding and result['payloads_decoded']==result['models']==0 and result['CUDA_initialized']is False and result['output_created']is False
 assert hooks==[] and not any(root.exists()for root in m.ALL_ROOTS.values())
 monkeypatch.setattr(sys,'argv',sys.argv+['--preflight'])
 with pytest.raises(ValueError,match='EXACT_CLI'):m.main()

# Explicit aggregate-baseline metadata acceptance; no payload/model is used here.
def segmented_initial_fixture(monkeypatch):
 from gx1.audit import native_future_state_provider_hlc_initial_v1 as owner
 base_ref={'path':str(owner.SEG_BASE_PLAN),'sha256':'a'*64};initial_ref={'path':str(owner.SEG_ROOT/'INITIAL.json'),'sha256':'b'*64};cp_ref={'path':str(owner.SEG_ROOT/'INITIAL.pt'),'sha256':'c'*64};dataset_ref={'path':'/fixture/segmented_dataset.json','sha256':'d'*64};certificate={'path':'/fixture/certificate.json','sha256':'e'*64};state='1'*64;rng='2'*64
 sha='3'*64;_,last_root=owner.segment_paths(sha);execution_ref={'path':str(owner.segment_paths(sha)[0]),'sha256':sha};result_ref={'path':str(last_root/'RESULT.json'),'sha256':'4'*64};terminal_ref={'path':str(last_root/'TERMINAL.json'),'sha256':'5'*64};claim={'path':str(last_root/'CLAIM.json'),'sha256':'6'*64}
 base={'schema_version':owner.SEG_BASE_SCHEMA,'stage':owner.SEG_STAGE,'output_root':str(owner.SEG_ROOT),'recipe':copy.deepcopy(owner.SEG_RECIPE),'model_contract':{'contract_sha256':m.ALL_CONTRACT_SHA},'metadata':{'target_contract':{'contract_sha256':'7'*64},'truth_sha256':'8'*64,'dataset':dataset_ref}}
 receipt={'plan':base_ref,'initial_only':True,'optimizer_updates':0,'checkpoint':cp_ref,'model_contract_sha256':m.ALL_CONTRACT_SHA,'target_contract_sha256':'7'*64,'model_state_sha256':state,'rng_sha256':rng}
 c1=owner.segment_limits(0,True);c1.update(model_forwards=15,backbone_forwards=14,readout_forwards=14,output_archive_write_calls=14,output_archive_readbacks=14,completed_batch_receipt_write_calls=14,committed_batches=14)
 c2=owner.segment_limits(14,False);nodes=[{'counts':c1},{'counts':c2}];counts={k:sum(x['counts'][k]for x in nodes)for k in owner.SEG_COUNT_KEYS}
 score={'rows':m.ROWS,'batch_count':owner.BATCHES,'batch_rows':owner.BATCH,'initial_prior_exact':True,'whole_TRAIN_initial':True,'state_sha256_before_and_after':state,'rng_sha256_before_and_restored':rng,'truth_sha256':'8'*64,'target_observation_certificate_ref':certificate,'output_sha256':{'activity_logits':'9'*64,'side_logits':'a'*64},'output_arrays_collected':2}
 report={'whole_TRAIN_initialbaseline_qualified':True,'unique_complete_batches':owner.BATCHES,'actual_NN_forward_rows':m.ROWS,'initial':score,'source_inputs_unchanged_across_segments':True,'each_segment_policy_boot_unchanged':True,'initial_checkpoint':initial_ref,'initial_state_sha256':state,'initial_rng_sha256':rng,'physical_execution_counts':counts,'counts':c2}
 result={'status':'COMPLETE_NATIVE_PROVIDER_HLC_ALL_TF_SEGMENTED_WHOLE_TRAIN_INITIAL','plan':base_ref,'execution_plan':execution_ref,'claim':claim,'report':report,'source_inputs_unchanged_across_segments':True,'each_segment_policy_boot_unchanged':True,'completed_execution_terminal':terminal_ref,'completed_execution_result':result_ref}
 acceptance={'full_initialbaseline_qualified':True,'result':{'path':str(owner.SEG_ROOT/'RESULT.json'),'sha256':'b'*64},'initial_receipt':initial_ref};terminal={'status':'COMPLETE','exit_code':0,'plan':execution_ref,'base_plan':base_ref,'claim':claim,'result':result_ref};execution_result=copy.deepcopy(result)
 parent={'prefix':{'cursor':owner.BATCHES,'rows_done':m.ROWS},'initial_receipt':initial_ref,'initial':receipt,'segments':nodes,'receipts':[None]*owner.BATCHES}
 values={terminal_ref['path']:terminal,result_ref['path']:execution_result,dataset_ref['path']:{'target_observation_certificate_ref':certificate}}
 monkeypatch.setattr(m,'read',lambda b:values[b['path']]);monkeypatch.setattr(m,'ref',lambda p:cp_ref);monkeypatch.setattr(owner,'segment_closed_parent',lambda *args:parent)
 return acceptance,result,base,receipt,parent,terminal,execution_result


def test_segmented_initial_accepts_complete_own_aggregate_not_single_process_counts(monkeypatch):
 a,r,b,i,*_=segmented_initial_fixture(monkeypatch)
 assert m.all_tf_require_segmented_initial_metadata(a,r,b,i)is True
 assert r['report']['physical_execution_counts']['model_constructions']==2
 assert r['report']['physical_execution_counts']['model_forwards']==1386


@pytest.mark.parametrize('change',['partial','prior','old_root','old_policy','wrong_recipe','updated','restored','missing_commit','missing_parent_receipt','bad_final_terminal','wrong_final_result','rng','truth','certificate','counter_mismatch'])
def test_segmented_initial_rejects_incomplete_foreign_or_updated_proof(monkeypatch,change):
 a,r,b,i,parent,t,execution=segmented_initial_fixture(monkeypatch)
 if change=='partial':r['report']['actual_NN_forward_rows']=3584
 elif change=='prior':r['report']['initial']['initial_prior_exact']=False
 elif change=='old_root':a['result']['path']=str(m.ALL_INITIAL_ROOT/'RESULT.json')
 elif change=='old_policy':b['model_contract']['backbone_gradient_policy']='M5_primary'
 elif change=='wrong_recipe':b['recipe']['batch_rows']=96
 elif change=='updated':r['report']['physical_execution_counts']['optimizer_steps']=1
 elif change=='restored':r['report']['physical_execution_counts']['model_state_restores']=1
 elif change=='missing_commit':parent['prefix']['cursor']-=1
 elif change=='missing_parent_receipt':parent['receipts'].pop()
 elif change=='bad_final_terminal':t['status']='FAILED_CONSUMED'
 elif change=='wrong_final_result':execution['report']['initial']['rows']=3584
 elif change=='rng':i['rng_sha256']='f'*64
 elif change=='truth':r['report']['initial']['truth_sha256']='f'*64
 elif change=='certificate':r['report']['initial']['target_observation_certificate_ref']={'path':'/foreign','sha256':'f'*64}
 elif change=='counter_mismatch':r['report']['physical_execution_counts']['model_forwards']+=1
 with pytest.raises(ValueError):m.all_tf_require_segmented_initial_metadata(a,r,b,i)


def test_segmented_replay_inspect_uses_explicit_schema_builder(monkeypatch,tmp_path):
 planfile=tmp_path/'segmented_replay.json';planfile.write_text('{}');monkeypatch.setattr(m,'ALL_PLAN',planfile);monkeypatch.setattr(m,'ALL_ROOTS',{'warm':tmp_path/'warm','cold':tmp_path/'cold'});policy=tmp_path/'policy.json';monkeypatch.setattr(m,'POLICY',policy)
 binding=m.ref(planfile);p={'schema_version':m.SEGMENTED_REPLAY_SCHEMA,'metadata':{'runtime':{'fixture':True}},'inputs':{'baseline_acceptance':{'fixture':True}},'source_commit':'a'*40};v={k:False for k in m.FLAGS};v[m.ALL_STAGE]=m.all_tf_gate(binding,False);policy.write_text(json.dumps(v));original_read=m.read;monkeypatch.setattr(m,'read',lambda b:p if b['path']==str(planfile)else original_read(b));calls=[]
 monkeypatch.setattr(m,'build_all_tf_segmented_replay_plan',lambda *args:(calls.append(args),p)[1]);monkeypatch.setattr(m,'build_all_tf_plan',lambda *args:pytest.fail('segmented plan reached legacy builder'));monkeypatch.setattr(m,'all_tf_check_sources_inputs',lambda *args:None);monkeypatch.setattr(m.core,'collect_future_cuda_runtime_metadata',lambda:{'fixture':True});monkeypatch.setattr(m.core,'future_cuda_runtime_matches',lambda *args:None)
 assert m.all_tf_inspect(binding['sha256'],'warm',False)[0]is p and len(calls)==1
