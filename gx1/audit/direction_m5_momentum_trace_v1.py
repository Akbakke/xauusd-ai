"""Trace a fixed M5 price intervention through already-qualified model inputs.

This reuses the completed intervention's exact parents and predictions, and
records only aggregate activation magnitudes. No optimizer or trading path.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from gx1.audit import direction_m5_momentum_intervention_v1 as base

REPO=base.REPO
ROOT=base.ROOT
STAGE='independent_m5_momentum_trace_20261029_002'
OUTPUT_NAME='DIRECTION_M5_MOMENTUM_TRACE_20261029_002'
TIMEFRAMES=base.TIMEFRAMES
FAMILY='momentum_flow_encoder'
JOINT_FIELDS=('local','fused','mtf_repr','global_context')


def require(value,why):
 if not bool(value):raise RuntimeError('M5_MOMENTUM_TRACE_'+why)


def _digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as handle:
  for chunk in iter(lambda:handle.read(1048576),b''):h.update(chunk)
 return h.hexdigest()


class TraceRecorder:
 """Pair unchanged/neutralized calls and aggregate only detached activations."""
 def __init__(self,reference):
  require(set(reference)=={'train','oof'},'REFERENCE_ROLES')
  self.reference=reference;self.offsets={'train':0,'oof':0};self.pending={};self.stats={};self.calls=0

 def _update(self,role,name,before,after):
  require(before.shape==after.shape and before.numel()>0,'ACTIVATION_SHAPE:'+name)
  x=before.detach().to(dtype=torch.float64);y=after.detach().to(dtype=torch.float64);delta=y-x
  require(bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()),'ACTIVATION_FINITE:'+name)
  row=self.stats.setdefault(role,{}).setdefault(name,{'cells':0,'baseline_sumsq':0.,'delta_sumsq':0.,'delta_abs_sum':0.,'max_abs_delta':0.})
  row['cells']+=x.numel();row['baseline_sumsq']+=float(torch.sum(x*x).item());row['delta_sumsq']+=float(torch.sum(delta*delta).item());row['delta_abs_sum']+=float(torch.sum(delta.abs()).item());row['max_abs_delta']=max(row['max_abs_delta'],float(delta.abs().max().item()))

 def wrap(self,original_forward):
  def traced(model,features,*,kind,counters,ceiling,torch,grad=False,return_outputs=False):
   require(not grad and not return_outputs,'NO_GRAD_FORWARD')
   role,sep,label=kind.partition('_');require(sep and role in self.reference and label in ('REPLAY','all_four'),'KIND')
   backbone=model.backbone;price=backbone.mtf_family_proj[FAMILY];gc=backbone.gc_mtf_family_proj
   require(all(tf+'__'+FAMILY in gc for tf in TIMEFRAMES),'GC_FAMILIES')
   captured={'price':[],'gc':{},'joint':[],'head':[]};handles=[]
   def price_hook(_module,_args,output):captured['price'].append(output.detach().clone())
   def gc_hook(tf):
    def hook(_module,_args,output):
     require(tf not in captured['gc'],'DUPLICATE_GC:'+tf);captured['gc'][tf]=output.detach().clone()
    return hook
   def joint_hook(_module,args):
    require(len(args)==1 and args[0].ndim==2 and args[0].shape[1]==512,'JOINT_SHAPE');captured['joint'].append(args[0].detach().clone())
   def head_hook(_module,args):
    require(len(args)==1 and args[0].ndim==2 and args[0].shape[1]==128,'HEAD_SHAPE');captured['head'].append(args[0].detach().clone())
   try:
    handles.append(price.register_forward_hook(price_hook))
    for tf in TIMEFRAMES:handles.append(gc[tf+'__'+FAMILY].register_forward_hook(gc_hook(tf)))
    handles.append(backbone.entry_q_joint_norm.register_forward_pre_hook(joint_hook))
    handles.append(model.direction_head.register_forward_pre_hook(head_hook))
    output=original_forward(model,features,kind=kind,counters=counters,ceiling=ceiling,torch=torch,grad=grad,return_outputs=return_outputs)
   finally:
    for handle in handles:handle.remove()
   require(len(captured['price'])==5 and set(captured['gc'])==set(TIMEFRAMES) and len(captured['joint'])==len(captured['head'])==1,'HOOK_ACCOUNTING')
   n=int(output.shape[0]);offset=self.offsets[role];expected=self.reference[role]['original' if label=='REPLAY' else 'all_four'][offset:offset+n]
   require(expected.shape==(n,60) and np.array_equal(output.detach().cpu().numpy(),expected),'EXACT_PRIOR_LOGITS:'+kind)
   captured['logits']=output.detach().clone();self.calls+=1
   if label=='REPLAY':
    require(role not in self.pending,'PAIR_ORDER');self.pending[role]=captured
   else:
    require(role in self.pending,'PAIR_MISSING');first=self.pending.pop(role)
    for idx,tf in enumerate(TIMEFRAMES):
     self._update(role,'price_projection_'+tf,first['price'][idx],captured['price'][idx])
     self._update(role,'gc_projection_'+tf,first['gc'][tf],captured['gc'][tf])
    for idx,name in enumerate(JOINT_FIELDS):self._update(role,'joint_source_'+name,first['joint'][0][:,idx*128:(idx+1)*128],captured['joint'][0][:,idx*128:(idx+1)*128])
    self._update(role,'entry_hidden',first['head'][0],captured['head'][0]);self._update(role,'logits_all_60',first['logits'],captured['logits'])
    self.offsets[role]+=n
   return output
  return traced

 def result(self):
  require(not self.pending and self.offsets=={'train':256,'oof':1736} and self.calls==250,'COMPLETE_PAIR_COUNTS')
  out={}
  for role,rows in self.stats.items():
   out[role]={}
   for name,v in rows.items():
    n=v['cells'];require(n>0,'EMPTY_AGGREGATE');out[role][name]={'cells':n,'baseline_rms':(v['baseline_sumsq']/n)**.5,'delta_rms':(v['delta_sumsq']/n)**.5,'mean_abs_delta':v['delta_abs_sum']/n,'max_abs_delta':v['max_abs_delta']}
  return out


def run(*,plan_path,plan_sha256,device,out_dir):
 from gx1.contracts import local_random_access_campaign_v2 as binding,immutable_event_authority_v1 as publication
 from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as window
 from gx1.scripts import run_entry_native_direction_research_window_v1 as common
 plan=binding.read_bound_json(Path(plan_path),plan_sha256)
 require(plan['stage']==STAGE and str(out_dir)==plan['output_dir'] and out_dir.name==OUTPUT_NAME,'PLAN_SCOPE')
 design=binding.read_bound_json(Path(plan['design']['path']),plan['design']['sha256']);previous=design['previous_paired_outputs']
 policy=json.loads((REPO/'NEXT_RUN_POLICY.json').read_text());prior=policy['independent_m5_momentum_intervention_20261029_003']
 require(prior['status']=='COMPLETE_REVIEWED_CONSUMED' and prior['paired_outputs']==previous and prior['execution_authorized'] is False,'PREVIOUS_SCOPE')
 binding.require_binding(previous,label='previous paired outputs');require(_digest(previous['path'])==previous['sha256'],'PREVIOUS_SHA')
 with np.load(previous['path'],allow_pickle=False) as archive:
  reference={role:{name:np.ascontiguousarray(archive[role+'__'+name]) for name in ('original','all_four')} for role in ('train','oof')}
 require(reference['train']['original'].shape==reference['train']['all_four'].shape==(256,60) and reference['oof']['original'].shape==reference['oof']['all_four'].shape==(1736,60),'REFERENCE_SHAPE')
 recorder=TraceRecorder(reference);old_forward=window._forward;old_metadata=window._launch_metadata
 def metadata_with_previous(*args,**kwargs):
  context=old_metadata(*args,**kwargs);stat=common._stat(previous['path']);require(common._hash_file(previous['path'])==previous,'PREVIOUS_CONTEXT_SHA')
  context['info']['metadata'][previous['path']]={'ref':dict(previous),'stat':stat};return context
 old_values={key:getattr(base,key) for key in ('STAGE','OUTPUT_NAME','ARMS','ALPHA','SOURCE_PATHS')}
 base.STAGE=STAGE;base.OUTPUT_NAME=OUTPUT_NAME;base.ARMS=('all_four',);base.ALPHA=.05/(1*2*2*3)
 base.SOURCE_PATHS=old_values['SOURCE_PATHS']+('gx1/audit/direction_m5_momentum_trace_v1.py','tests/test_direction_m5_momentum_trace_v1.py')
 window._forward=recorder.wrap(old_forward);window._launch_metadata=metadata_with_previous
 try:
  report=base.run(plan_path=plan_path,plan_sha256=plan_sha256,device=device,out_dir=out_dir)
  rows=recorder.result();require(report['forward_calls']==250 and report['optimizer_steps']==0 and report['model_state_unchanged'] is True,'BASE_TERMINAL')
  require(_digest(previous['path'])==previous['sha256'],'PREVIOUS_FINAL_SHA')
  trace={'schema_version':'gx1_m5_momentum_activation_trace_v1','status':'COMPLETE','plan':{'path':str(plan_path),'sha256':plan_sha256},'previous_paired_outputs':previous,'exact_previous_original_and_ablated_logits':True,'forward_calls':recorder.calls,'rows':dict(recorder.offsets),'activation_aggregates':rows,'optimizer_steps':0,'test_data_used':False,'economic_edge_qualified':False,'limitations':'Aggregated internal activations on the same reused TRAIN diagnostic and Q1 development subsets. Time-window cells overlap; price and GC projections have different feature sets and scales. Neutralization may be out of distribution and does not prove retraining benefit or economic edge.'}
  result_path=out_dir/'TRACE_RESULT.json';publication._publish_bytes(result_path,binding.canonical_bytes(trace));terminal={'status':'COMPLETE','trace_result':{'path':str(result_path),'sha256':_digest(result_path)},'base_result_path':str(out_dir/'OUTPUT/RESULT.json'),'test_data_used':False}
  publication._publish_bytes(out_dir/'TRACE_TERMINAL.json',binding.canonical_bytes(terminal));return terminal
 finally:
  window._forward=old_forward;window._launch_metadata=old_metadata
  for key,value in old_values.items():setattr(base,key,value)
