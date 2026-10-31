import numpy as np
import pytest
import torch
from torch import nn
from gx1.audit import direction_m5_momentum_intervention_v1 as base
from gx1.audit import direction_m5_momentum_trace_v1 as trace
from tests.test_entry_gc_mtf_model_v1 import bound_history,candidate_case


class _Backbone(nn.Module):
 def __init__(self):
  super().__init__();self.mtf_family_proj=nn.ModuleDict({trace.FAMILY:nn.Linear(4,128)})
  self.gc_mtf_family_proj=nn.ModuleDict({tf+'__'+trace.FAMILY:nn.Linear(2,128,bias=False) for tf in trace.TIMEFRAMES})
  self.entry_q_joint_norm=nn.LayerNorm(512)

class _Toy(nn.Module):
 def __init__(self):
  super().__init__();self.backbone=_Backbone();self.direction_head=nn.Linear(128,60)
 def forward(self,features):
  b=self.backbone;items=[]
  for tf in trace.TIMEFRAMES:
   price=b.mtf_family_proj[trace.FAMILY](features['price'][tf]);gc=b.gc_mtf_family_proj[tf+'__'+trace.FAMILY](features['gc'][tf]);items.append((price+gc).mean(dim=1))
  mtf=torch.stack(items).mean(dim=0);local=torch.ones_like(mtf);context=local*2;fused=local+mtf*.5
  joint=b.entry_q_joint_norm(torch.cat((local,fused,mtf,context),dim=1));hidden=joint[:,256:384]
  return self.direction_head(hidden)

def _forward(model,features,*,kind,counters,ceiling,torch,grad=False,return_outputs=False):
 counters['calls']+=1
 with torch.no_grad():return model(features)


def test_real_hooks_compare_matched_calls_without_changing_other_branches():
 torch.manual_seed(271);model=_Toy().eval();features={'price':{},'gc':{}}
 for i,tf in enumerate(trace.TIMEFRAMES):
  features['price'][tf]=torch.randn(2,i+2,4);features['gc'][tf]=torch.randn(2,i+2,2)
 with torch.no_grad():
  original=model(features).numpy().copy();h,_=base.install_neutralization(model.backbone.mtf_family_proj[trace.FAMILY],(0,1,2,3))
  try:ablated=model(features).numpy().copy()
  finally:h.remove()
 reference={'train':{'original':original,'all_four':ablated},'oof':{'original':original,'all_four':ablated}}
 rec=trace.TraceRecorder(reference);f=rec.wrap(_forward);counters={'calls':0}
 z=f(model,features,kind='train_REPLAY',counters=counters,ceiling=2,torch=torch)
 h,_=base.install_neutralization(model.backbone.mtf_family_proj[trace.FAMILY],(0,1,2,3))
 try:z2=f(model,features,kind='train_all_four',counters=counters,ceiling=2,torch=torch)
 finally:h.remove()
 assert np.array_equal(z.numpy(),original) and np.array_equal(z2.numpy(),ablated)
 assert counters['calls']==2 and rec.offsets['train']==2 and not rec.pending
 for tf in trace.TIMEFRAMES:
  assert rec.stats['train']['price_projection_'+tf]['delta_sumsq']>0
  assert rec.stats['train']['gc_projection_'+tf]['delta_sumsq']==0
 assert rec.stats['train']['joint_source_local']['delta_sumsq']==0
 assert rec.stats['train']['joint_source_global_context']['delta_sumsq']==0
 assert rec.stats['train']['joint_source_mtf_repr']['delta_sumsq']>0
 assert rec.stats['train']['entry_hidden']['delta_sumsq']>0
 assert rec.stats['train']['logits_all_60']['delta_sumsq']>0
 assert not model.backbone.mtf_family_proj[trace.FAMILY]._forward_hooks
 assert not model.backbone.entry_q_joint_norm._forward_pre_hooks
 assert not model.direction_head._forward_pre_hooks


def test_trace_rejects_unmatched_prior_logits():
 torch.manual_seed(91);model=_Toy().eval();features={'price':{},'gc':{}}
 for tf in trace.TIMEFRAMES:features['price'][tf]=torch.randn(2,2,4);features['gc'][tf]=torch.randn(2,2,2)
 wrong=np.full((2,60),999.,dtype=np.float32);ref={role:{'original':wrong,'all_four':wrong} for role in ('train','oof')}
 with pytest.raises(RuntimeError,match='EXACT_PRIOR_LOGITS'):
  trace.TraceRecorder(ref).wrap(_forward)(model,features,kind='train_REPLAY',counters={'calls':0},ceiling=1,torch=torch)
 assert not model.direction_head._forward_pre_hooks


def test_trace_hooks_match_real_native_model_routes(bound_history,candidate_case):
 from gx1.features import entry_specialist_feature_groups_v1 as owners
 from gx1.features.entry_direction_momentum_profile_v1 import compile_price_momentum_profile
 from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EXACT_SPECIALIST_NAMES
 from gx1.contracts.entry_native_direction_dense_v1 import DenseNativeEntryDirectionResearch
 from tests.test_entry_gc_mtf_model_v1 import _reference
 from tests.test_entry_v10_ctx_model_shapes import _make_dynamic_m5_model
 history=bound_history[0];_,_,inputs,normalization=candidate_case
 profile=compile_price_momentum_profile(history);physical=profile['physical_field_names'];signal=list(physical['signal']);mtf=list(physical['mtf'])
 aliases=owners.model_native_context_temporal_alias_policy(signal)['aliases'];grouped=owners.group_features_by_specialist(signal)
 kwargs=dict(seq_input_dim=len(signal),snap_input_dim=len(signal),
   specialist_input_indices={f:[i for i,n in enumerate(signal) if n in set(grouped[f])] for f in EXACT_SPECIALIST_NAMES},
   **{tf.lower()+'_seq_dim':len(mtf) for tf in history['layout']['timeframes']},
   **{tf.lower()+'_seq_len':history['layout']['window_rows'][tf] for tf in history['layout']['timeframes']},
   multi_tf_specialist_input_indices={f:list(v) for f,v in owners.require_multi_tf_specialist_routing_v4(mtf).items()},
   temporal_alias_signal_indices=[int(x['signal_index']) for x in aliases],temporal_alias_ctx_cont_indices=[int(x['ctx_cont_index']) for x in aliases],
   input_normalization=normalization,gc_mtf_history=history,dropout=0.)
 torch.manual_seed(8193);backbone=_make_dynamic_m5_model(**kwargs,entry_execution_profile=_reference(profile)).eval();model=DenseNativeEntryDirectionResearch(backbone).eval()
 with torch.no_grad():
  torch.manual_seed(912);backbone.mtf_family_proj[trace.FAMILY].weight.copy_(torch.randn_like(backbone.mtf_family_proj[trace.FAMILY].weight)*.1)
  original=model(**inputs)['logit_up'].numpy().copy();h,_=base.install_neutralization(backbone.mtf_family_proj[trace.FAMILY],(0,1,2,3))
  try:ablated=model(**inputs)['logit_up'].numpy().copy()
  finally:h.remove()
 assert not np.array_equal(original,ablated)
 def forward(actual,features,*,kind,counters,ceiling,torch,grad=False,return_outputs=False):
  counters['calls']+=1
  with torch.no_grad():return actual(**features)['logit_up']
 ref={role:{'original':original,'all_four':ablated} for role in ('train','oof')};rec=trace.TraceRecorder(ref);f=rec.wrap(forward);counters={'calls':0}
 f(model,inputs,kind='train_REPLAY',counters=counters,ceiling=2,torch=torch)
 h,_=base.install_neutralization(backbone.mtf_family_proj[trace.FAMILY],(0,1,2,3))
 try:f(model,inputs,kind='train_all_four',counters=counters,ceiling=2,torch=torch)
 finally:h.remove()
 assert rec.offsets['train']==len(original) and rec.stats['train']['joint_source_mtf_repr']['delta_sumsq']>0
 assert all(rec.stats['train']['gc_projection_'+tf]['delta_sumsq']==0 for tf in trace.TIMEFRAMES)


def test_validator_dispatches_only_the_fixed_trace_plan(monkeypatch,tmp_path):
 from gx1.scripts import validate_entry_model_native_technical_checkpoint_v1 as validator
 calls=[];monkeypatch.setattr(trace,'run',lambda **kwargs:calls.append(kwargs) or {'status':'SYNTHETIC'})
 plan='/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_M5_MOMENTUM_TRACE_PLAN_20261029_002.json'
 args=['--direction-m5-momentum-intervention-plan',plan,'--direction-m5-momentum-intervention-plan-sha256','a'*64,'--device','cuda','--out-dir',str(tmp_path)]
 assert validator.main(args)==0 and len(calls)==1 and str(calls[0]['plan_path'])==plan
 with pytest.raises(SystemExit):validator.main(args+['--device','cuda'])


def test_design_path_is_bound_to_exact_intervention_stage():
 assert base.design_path_for_stage('independent_m5_momentum_intervention_20261029_003').name=='DIRECTION_M5_MOMENTUM_INTERVENTION_20261029.json'
 assert base.design_path_for_stage(trace.STAGE).name=='DIRECTION_M5_MOMENTUM_TRACE_20261029.json'
 with pytest.raises(RuntimeError,match='DESIGN_STAGE'):base.design_path_for_stage('other_stage')


class _LinearUpstreamToy(_Toy):
 def __init__(self):
  super().__init__();self.backbone.entry_q_joint_norm=nn.Identity()

def _upstream_case(*,input_grad=True):
 torch.manual_seed(317)
 model=_LinearUpstreamToy().eval();features={'price':{},'gc':{}}
 for i,tf in enumerate(trace.TIMEFRAMES):
  for branch,width in (('price',4),('gc',2)):
   features[branch][tf]=torch.randn(2,i+2,width,requires_grad=input_grad)
 decisions=torch.tensor([10**15,10**15+31*60*10**9],dtype=torch.int64)
 terminal=decisions[:,None]+torch.arange(1,61,dtype=torch.int64)[None,:]*60*10**9
 sign=torch.ones(2,60,dtype=torch.int8);sign[:,1::2]=-1
 observed=torch.ones(2,60,dtype=torch.bool)
 observed[0,0]=False;sign[0,0]=0;sign[0,1]=0
 target={'decision_ns':decisions,'terminal_time_ns':terminal,'sign':sign,'observed':observed}
 return model,features,target,int(decisions[0]+60*60*10**9)

def test_upstream_two_reverses_match_linear_chain_analytic_gradients_and_preserve_state():
 model,features,target,cutoff=_upstream_case()
 original={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
 recorder=trace.UpstreamGradientRecorder();calls=[]
 def reverse(*args,**kwargs):
  calls.append(kwargs.copy());return torch.autograd.grad(*args,**kwargs)
 with torch.no_grad():expected_logits=model(features)
 with recorder.capture(model):
  logits=model(features)
  assert torch.equal(logits.detach(),expected_logits)
 row=recorder.measure(logits,target,cutoff,reverse=reverse)
 assert len(calls)==2 and calls[0]['retain_graph'] is True and calls[1]['retain_graph'] is False
 assert all(p.grad is None for p in model.parameters())
 assert all(torch.equal(model.state_dict()[k],v) for k,v in original.items()) and torch.equal(torch.get_rng_state(),rng)
 assert not model.direction_head._forward_pre_hooks and not model.backbone.mtf_family_proj[trace.FAMILY]._forward_hooks
 mask=target['observed']&(target['sign']!=0)&(target['terminal_time_ns']<cutoff)
 n=int(mask.sum());residual=torch.where(mask,logits.detach().sigmoid()-(target['sign']==1).to(logits.dtype),0.)/n
 w=model.direction_head.weight.detach()
 head5=residual[:,4:5]*w[4:5];other=residual@w-head5
 assert row['eligible_cells']==n and row['eligible_five_minute_cells']==int(mask[:,4].sum())
 for tf in trace.TIMEFRAMES:
  length=features['price'][tf].shape[1]
  g5=head5[:,None,:].expand(-1,length,-1)/(5*length)
  go=other[:,None,:].expand(-1,length,-1)/(5*length)
  for branch,layer in [('price',model.backbone.mtf_family_proj[trace.FAMILY]),('gc',model.backbone.gc_mtf_family_proj[tf+'__'+trace.FAMILY])]:
   output=row['locations'][branch+'_projection_output_'+tf]
   assert output['five_sumsq']==pytest.approx(float(g5.double().square().sum()),rel=2e-5)
   assert output['dot']==pytest.approx(float((g5.double()*go.double()).sum()),rel=2e-5,abs=1e-12)
   gin=g5@layer.weight.detach();input_row=row['locations'][branch+'_projection_input_'+tf]
   assert input_row['five_sumsq']==pytest.approx(float(gin.double().square().sum()),rel=2e-5)
   assert input_row['five_dot_activation']==pytest.approx(float((gin.double()*features[branch][tf].detach().double()).sum()),rel=2e-5,abs=1e-12)
  assert row['additive_output_gradient_equal'][tf]==[True,True]
 summary=recorder.result()
 assert summary['reverse_calls']==2 and summary['batch_count']==1 and summary['parents']==2
 assert summary['aggregate']['head_input']['five_norm']==pytest.approx(float(head5.double().norm()),rel=2e-5)
 assert row['locations']['joint_source_local']['five_gradient_state']=='zero'
 assert row['locations']['joint_source_mtf_repr']['five_gradient_state']=='nonzero'


def test_upstream_masks_zero_five_and_nondifferentiable_inputs_are_distinct():
 model,features,target,cutoff=_upstream_case(input_grad=False)
 target['sign'][:,4]=0
 recorder=trace.UpstreamGradientRecorder()
 with recorder.capture(model):
  row=recorder.measure(model(features),target,cutoff)
 output=row['locations']['price_projection_output_m5'];inputs=row['locations']['price_projection_input_m5']
 assert output['five_gradient_state']=='zero' and output['five_sumsq']==0.
 assert output['cosine'] is None and inputs['five_gradient_state']=='not_requires_grad'
 assert inputs['five_sumsq'] is None and inputs['five_dot_activation'] is None
 assert inputs['activation_sumsq']>0
 assert recorder.result()['aggregate']['price_projection_input_m5']['five_norm'] is None


def test_upstream_disconnected_is_not_fabricated_zero():
 x=torch.ones(2,3,4,requires_grad=True)
 missing=trace._upstream_geometry(x,[None,None])
 zero=trace._upstream_geometry(x,[torch.zeros_like(x),torch.zeros_like(x)])
 assert missing['five_gradient_state']=='disconnected' and missing['five_sumsq'] is None
 assert zero['five_gradient_state']=='zero' and zero['five_sumsq']==0.
 assert missing['dot'] is None and zero['dot']==0. and zero['cosine'] is None


def test_upstream_aggregates_add_squares_not_batch_norms():
 model,features,target,cutoff=_upstream_case();recorder=trace.UpstreamGradientRecorder()
 for _ in range(2):
  with recorder.capture(model):row=recorder.measure(model(features),target,cutoff)
 result=recorder.result();location='price_projection_input_m15'
 assert result['parents']==4 and result['reverse_calls']==4
 aggregate=result['aggregate'][location];single=row['locations'][location]
 assert aggregate['five_sumsq']==pytest.approx(2*single['five_sumsq'])
 assert aggregate['five_norm']==pytest.approx(2**.5*single['five_norm'])
 assert aggregate['five_dot_activation']==pytest.approx(2*single['five_dot_activation'])
 assert aggregate['cosine']==pytest.approx(single['cosine'])
 assert aggregate['cells']==2*single['cells']


@pytest.mark.parametrize('fault',['train_mode','no_graph','empty','wrong_clock','exception'])
def test_upstream_fails_closed_and_removes_hooks(fault):
 model,features,target,cutoff=_upstream_case();recorder=trace.UpstreamGradientRecorder()
 if fault=='train_mode':model.train()
 if fault=='empty':target['sign'].zero_()
 if fault=='wrong_clock':target['terminal_time_ns'][0,0]+=1
 with pytest.raises(RuntimeError):
  with recorder.capture(model):
   if fault=='exception':raise RuntimeError('synthetic interrupt')
   if fault=='no_graph':
    with torch.no_grad():logits=model(features)
   else:logits=model(features)
   recorder.measure(logits,target,cutoff)
 assert not model.direction_head._forward_pre_hooks and not model.backbone.mtf_family_proj[trace.FAMILY]._forward_hooks
 assert not recorder._handles and recorder._captured is None and not recorder.batches
 assert all(p.grad is None for p in model.parameters())
