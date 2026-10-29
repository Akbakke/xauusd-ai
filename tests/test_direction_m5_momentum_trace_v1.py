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
 plan='/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_M5_MOMENTUM_TRACE_PLAN_20261029_001.json'
 args=['--direction-m5-momentum-intervention-plan',plan,'--direction-m5-momentum-intervention-plan-sha256','a'*64,'--device','cuda','--out-dir',str(tmp_path)]
 assert validator.main(args)==0 and len(calls)==1 and str(calls[0]['plan_path'])==plan
 with pytest.raises(SystemExit):validator.main(args+['--device','cuda'])
