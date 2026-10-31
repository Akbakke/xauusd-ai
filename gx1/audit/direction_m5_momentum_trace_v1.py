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



def _upstream_geometry(activation, gradients):
 """Local activation derivatives; a missing derivative is not an observed zero."""
 x=activation.detach().to(dtype=torch.float64)
 require(bool(torch.isfinite(x).all()),'UPSTREAM_ACTIVATION_FINITE')
 row={'cells':x.numel(),'activation_sumsq':float(x.square().sum()),'activation_rms':float(x.square().mean().sqrt())}
 values=[]
 for label,g in zip(('five','other59'),gradients):
  state=('not_requires_grad' if not activation.requires_grad else 'disconnected') if g is None else ('zero' if not bool(torch.count_nonzero(g)) else 'nonzero')
  row[label+'_gradient_state']=state
  if g is None:
   row[label+'_sumsq']=row[label+'_norm']=row[label+'_dot_activation']=None;values.append(None)
  else:
   require(g.shape==activation.shape and bool(torch.isfinite(g).all()),'UPSTREAM_GRADIENT_DOMAIN')
   value=g.detach().to(dtype=torch.float64);square=float(value.square().sum())
   row[label+'_sumsq']=square;row[label+'_norm']=square**.5
   row[label+'_dot_activation']=float((value*x).sum());values.append(value)
 if all(v is not None for v in values):
  dot=float((values[0]*values[1]).sum());combined=float((values[0]+values[1]).square().sum())
  row.update(dot=dot,combined_sumsq=combined,
   cosine=dot/(row['five_norm']*row['other59_norm']) if row['five_norm'] and row['other59_norm'] else None,
   five_norm_div_other59_norm=row['five_norm']/row['other59_norm'] if row['other59_norm'] else None)
 else:row.update(dot=None,combined_sumsq=None,cosine=None,five_norm_div_other59_norm=None)
 return row


class UpstreamGradientRecorder:
 """TRAIN diagnostic hooks only; the caller owns source/data/guard/state proofs.

 Two reverse calls measure five-minute versus other-59 contributions using the
 actual dense-loss mask and total denominator. Captures never change requires_grad.
 Projection-output pair equality checks their additive merge, not attenuation.
 sum(g*x) is a local derivative for activation scaling, not a parameter update.
 """
 def __init__(self):
  self.batches=[];self._captured=None;self._handles=[];self.reverse_calls=0

 def capture(self,model):
  from contextlib import contextmanager
  @contextmanager
  def installed():
   require(self._captured is None and not self._handles,'UPSTREAM_CAPTURE_ACTIVE')
   require(not model.training and all(not m.training for m in model.modules()),'UPSTREAM_EVAL_MODE')
   self._captured={'price':[],'gc':{},'joint':[],'head':[]}
   backbone=model.backbone
   def price(_module,args,output):
    require(len(args)==1,'UPSTREAM_PRICE_ARGS');self._captured['price'].append((args[0],output))
   def gc(tf):
    def hook(_module,args,output):
     require(len(args)==1 and tf not in self._captured['gc'],'UPSTREAM_GC_ACCOUNTING')
     self._captured['gc'][tf]=(args[0],output)
    return hook
   def pre(key):
    def hook(_module,args):
     require(len(args)==1,'UPSTREAM_PRE_ARGS');self._captured[key].append(args[0])
    return hook
   try:
    self._handles.append(backbone.mtf_family_proj[FAMILY].register_forward_hook(price))
    for tf in TIMEFRAMES:self._handles.append(backbone.gc_mtf_family_proj[tf+'__'+FAMILY].register_forward_hook(gc(tf)))
    self._handles.append(backbone.entry_q_joint_norm.register_forward_pre_hook(pre('joint')))
    self._handles.append(model.direction_head.register_forward_pre_hook(pre('head')))
    yield self
   except BaseException:
    self._captured=None
    raise
   finally:
    for handle in self._handles:handle.remove()
    self._handles=[]
  return installed()

 def measure(self,logits,target,cutoff_ns,*,reverse=None):
  from gx1.contracts.entry_native_direction_dense_v1 import dense_direction_binary_loss
  require(self._captured is not None,'UPSTREAM_CAPTURE_REQUIRED')
  captured=self._captured;self._captured=None
  try:
   require(len(captured['price'])==5 and set(captured['gc'])==set(TIMEFRAMES)
    and len(captured['joint'])==len(captured['head'])==1,'UPSTREAM_HOOK_ACCOUNTING')
   require(logits.requires_grad,'UPSTREAM_LOGITS_GRAPH')
   loss=dense_direction_binary_loss(logits,**target,cutoff_ns=cutoff_ns)
   require(loss['update_allowed'],'UPSTREAM_EMPTY_ELIGIBLE')
   n=logits.shape[0];points={}
   for i,tf in enumerate(TIMEFRAMES):
    price,gc=captured['price'][i],captured['gc'][tf]
    require(price[1].shape==gc[1].shape and price[1].ndim==3
     and price[1].shape[0]==n and price[1].shape[2]==128,'UPSTREAM_MATCHED_PROJECTIONS')
    for branch,pair in (('price',price),('gc',gc)):
     require(pair[0].ndim==3 and pair[0].shape[:2]==pair[1].shape[:2],'UPSTREAM_PROJECTION_INPUT')
     for side,tensor in zip(('input','output'),pair):points[branch+'_projection_'+side+'_'+tf]=tensor
   points['joint_input']=captured['joint'][0];points['head_input']=captured['head'][0]
   require(points['joint_input'].shape==(n,512) and points['head_input'].shape==(n,128),'UPSTREAM_FUSION_SHAPE')
   require(all(isinstance(x,torch.Tensor) and x.device==logits.device and x.dtype==logits.dtype
    and bool(torch.isfinite(x).all()) for x in points.values()),'UPSTREAM_CAPTURE_DOMAIN')
   mask=loss['eligible_mask'];count=loss['counts']['eligible_cells']
   bce=torch.nn.functional.binary_cross_entropy_with_logits(logits,(target['sign']==1).to(logits.dtype),reduction='none')
   other=mask.clone();other[:,4]=False
   pieces=(bce[:,4][mask[:,4]].sum()/count,bce[other].sum()/count)
   rounding=(count+len(pieces))*torch.finfo(logits.dtype).eps
   require(rounding<1 and bool(torch.isclose(sum(pieces),loss['loss'],rtol=rounding/(1-rounding),atol=0)),'UPSTREAM_LOSS_DECOMPOSITION')
   names=[name for name,x in points.items() if x.requires_grad]
   require(bool(names),'UPSTREAM_DIFFERENTIABLE_CAPTURES')
   reverse=torch.autograd.grad if reverse is None else reverse
   gradients=[]
   for index,piece in enumerate(pieces):
    values=reverse(piece,[points[name] for name in names],retain_graph=index==0,create_graph=False,allow_unused=True)
    self.reverse_calls+=1
    require(len(values)==len(names),'UPSTREAM_REVERSE_ARITY')
    gradients.append(dict(zip(names,values)))
   equality={}
   for tf in TIMEFRAMES:
    pair=[]
    for component in gradients:
     a=component.get('price_projection_output_'+tf);b=component.get('gc_projection_output_'+tf)
     pair.append((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)))
    equality[tf]=pair
   require(all(all(v) for v in equality.values()),'UPSTREAM_ADDITIVE_MERGE_GRADIENT')
   rows={name:_upstream_geometry(x,[g.get(name) for g in gradients]) for name,x in points.items()}
   for index,name in enumerate(JOINT_FIELDS):
    sl=slice(index*128,(index+1)*128)
    rows['joint_source_'+name]=_upstream_geometry(points['joint_input'][:,sl],
     [g['joint_input'][:,sl] if g.get('joint_input') is not None else None for g in gradients])
   batch={'parents':n,'eligible_cells':count,'eligible_five_minute_cells':int(mask[:,4].sum()),
    'loss_five':float(pieces[0].detach()),'loss_other59':float(pieces[1].detach()),
    'additive_output_gradient_equal':equality,'locations':rows}
   self.batches.append(batch)
   return batch
  finally:
   captured.clear()

 def result(self):
  require(self._captured is None and not self._handles and bool(self.batches),'UPSTREAM_RESULT_BOUNDARY')
  require(self.reverse_calls==2*len(self.batches),'UPSTREAM_REVERSE_COUNTS')
  aggregate={}
  for name in self.batches[0]['locations']:
   rows=[batch['locations'][name] for batch in self.batches]
   cells=sum(row['cells'] for row in rows);square=sum(row['activation_sumsq'] for row in rows)
   out={'cells':cells,'activation_sumsq':square,'activation_rms':(square/cells)**.5}
   for label in ('five','other59'):
    states=[row[label+'_gradient_state'] for row in rows]
    out[label+'_batch_states']={state:states.count(state) for state in ('not_requires_grad','disconnected','zero','nonzero')}
    for metric in ('sumsq','dot_activation'):
     values=[row[label+'_'+metric] for row in rows]
     out[label+'_'+metric]=sum(values) if all(v is not None for v in values) else None
    v=out[label+'_sumsq'];out[label+'_norm']=v**.5 if v is not None else None
   for metric in ('dot','combined_sumsq'):
    values=[row[metric] for row in rows];out[metric]=sum(values) if all(v is not None for v in values) else None
   a,b=out['five_norm'],out['other59_norm']
   out['cosine']=out['dot']/(a*b) if a and b and out['dot'] is not None else None
   out['five_norm_div_other59_norm']=a/b if a is not None and b else None
   aggregate[name]=out
  return {'batch_count':len(self.batches),'parents':sum(b['parents'] for b in self.batches),
   'reverse_calls':self.reverse_calls,'batches':self.batches,'aggregate':aggregate,
   'measurement':'eval activation gradients; shared eligible-cell denominator; no parameter-update claim',
   'interpretation':'Projection-output gradient equality is an additive-merge identity. Raw gradient norms depend on activation scale; dot_activation is a local activation-scaling derivative. No attenuation or predictive-value claim.'}


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
