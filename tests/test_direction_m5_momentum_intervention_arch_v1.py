import torch
from gx1.audit import direction_m5_momentum_intervention_v1 as audit
from gx1.features import entry_specialist_feature_groups_v1 as owners
from gx1.features.entry_direction_momentum_profile_v1 import MOMENTUM_FAMILY,compile_price_momentum_profile
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EXACT_SPECIALIST_NAMES
from tests.test_entry_gc_mtf_model_v1 import bound_history,candidate_case,_reference
from tests.test_entry_v10_ctx_model_shapes import _make_dynamic_m5_model

def test_hook_uses_real_mtf_architecture_and_removes_only_target_path(bound_history,candidate_case):
    history=bound_history[0];_,_,base_inputs,normalization=candidate_case
    profile=compile_price_momentum_profile(history);physical=profile['physical_field_names'];signal=list(physical['signal']);mtf=list(physical['mtf'])
    aliases=owners.model_native_context_temporal_alias_policy(signal)['aliases'];grouped=owners.group_features_by_specialist(signal)
    kwargs=dict(seq_input_dim=len(signal),snap_input_dim=len(signal),
      specialist_input_indices={f:[i for i,n in enumerate(signal) if n in set(grouped[f])] for f in EXACT_SPECIALIST_NAMES},
      **{tf.lower()+'_seq_dim':len(mtf) for tf in history['layout']['timeframes']},
      **{tf.lower()+'_seq_len':history['layout']['window_rows'][tf] for tf in history['layout']['timeframes']},
      multi_tf_specialist_input_indices={f:list(v) for f,v in owners.require_multi_tf_specialist_routing_v4(mtf).items()},
      temporal_alias_signal_indices=[int(x['signal_index']) for x in aliases],temporal_alias_ctx_cont_indices=[int(x['ctx_cont_index']) for x in aliases],
      input_normalization=normalization,gc_mtf_history=history,dropout=0.0)
    torch.manual_seed(8193);model=_make_dynamic_m5_model(**kwargs,entry_execution_profile=_reference(profile)).eval()
    inputs={k:v.clone() for k,v in base_inputs.items()};inputs['seq_m5'][:,:,mtf.index('rsi14_centered')]+=2.0
    layer=model.mtf_family_proj[MOMENTUM_FAMILY];indices=model._multi_tf_specialist_index_tuples[MOMENTUM_FAMILY];fields=model._input_normalization_field_names['mtf_m5'];routed=[fields[i] for i in indices];rsi=routed.index('rsi14_centered')
    with torch.no_grad():
      torch.manual_seed(912);layer.weight.copy_(torch.randn_like(layer.weight)*0.1)
      observer,calls,_=audit.install_input_observer(layer,tuple(routed))
      try:baseline=model(**inputs)['entry_q_joint_hidden'].clone()
      finally:observer.remove()
      assert tuple(calls)==audit.TIMEFRAMES
      handle,calls=audit.install_neutralization(layer,(rsi,))
      try:ablated=model(**inputs)['entry_q_joint_hidden'].clone()
      finally:handle.remove()
      assert len(calls)==5 and not torch.equal(baseline,ablated)
      restored=model(**inputs)['entry_q_joint_hidden']
      assert torch.equal(baseline,restored)
