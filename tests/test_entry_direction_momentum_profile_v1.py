"""Synthetic route and learning-contact checks; no actual data or edge claim."""
import pytest
import torch

from gx1.features import entry_specialist_feature_groups_v1 as original
from gx1.features.entry_direction_momentum_profile_v1 import (
    MOMENTUM_FAMILY, PRICE_MOMENTUM_FIELDS, PROFILE_NAME,
    compile_price_momentum_profile, require_entry_execution_profile_reference,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EXACT_SPECIALIST_NAMES
from tests.test_entry_gc_mtf_model_v1 import bound_history, candidate_case, _reference
from tests.test_entry_native_direction_feature_view_v1 import bound_case
from tests.test_entry_v10_ctx_model_shapes import _make_dynamic_m5_model


def test_compact_candidate_adds_only_four_genuine_owned_mtf_fields(bound_history):
    history=bound_history[0]
    parent=original.compile_gc_mtf_geometry_entry_execution_profile(history)
    candidate=compile_price_momentum_profile(history)
    assert candidate['research_parent_entry_execution_profile']==_reference(parent)
    assert candidate['physical_field_names']==parent['physical_field_names']
    assert tuple(candidate['mtf_specialists'])==tuple(EXACT_SPECIALIST_NAMES)
    assert candidate['macro_context_fields']==parent['macro_context_fields']
    extra=set(candidate['mtf_indices'])-set(parent['mtf_indices'])
    names=candidate['physical_field_names']['mtf']
    assert {names[i] for i in extra}==set(PRICE_MOMENTUM_FIELDS)
    assert set(candidate['mtf_indices_by_specialist'][MOMENTUM_FAMILY])==extra
    assert len(parent['mtf_indices'])==178 and len(candidate['mtf_indices'])==182
    for f in parent['base_mtf_specialists']:
        assert candidate['mtf_indices_by_specialist'][f]==parent['mtf_indices_by_specialist'][f]
    ref=_reference(candidate)
    assert require_entry_execution_profile_reference(ref,gc_mtf_history=history)==ref
    assert require_entry_execution_profile_reference(_reference(parent),gc_mtf_history=history)==_reference(parent)
    with pytest.raises(RuntimeError):
        require_entry_execution_profile_reference({**ref,'contract_sha256':'0'*64},gc_mtf_history=history)
    # Research routing must not become a selectable production contract.
    from gx1.contracts.entry_exit_production_architecture_v1 import entry_exit_production_architecture_contract
    with pytest.raises(RuntimeError):
        entry_exit_production_architecture_contract(entry_execution_profile=ref,gc_mtf_history=history)


def test_new_price_path_preserves_old_seed_and_accepts_training_gradients(bound_history,candidate_case):
    history=bound_history[0]
    _unused,_existing,inputs,normalization=candidate_case
    parent=original.compile_gc_mtf_geometry_entry_execution_profile(history)
    candidate=compile_price_momentum_profile(history)
    physical=parent['physical_field_names']; signal=list(physical['signal']);mtf=list(physical['mtf'])
    aliases=original.model_native_context_temporal_alias_policy(signal)['aliases']
    grouped=original.group_features_by_specialist(signal)
    kwargs=dict(seq_input_dim=len(signal),snap_input_dim=len(signal),
        specialist_input_indices={f:[i for i,n in enumerate(signal) if n in set(grouped[f])] for f in EXACT_SPECIALIST_NAMES},
        **{tf.lower()+'_seq_dim':len(mtf) for tf in history['layout']['timeframes']},
        **{tf.lower()+'_seq_len':history['layout']['window_rows'][tf] for tf in history['layout']['timeframes']},
        multi_tf_specialist_input_indices={f:list(v) for f,v in original.require_multi_tf_specialist_routing_v4(mtf).items()},
        temporal_alias_signal_indices=[int(x['signal_index']) for x in aliases],
        temporal_alias_ctx_cont_indices=[int(x['ctx_cont_index']) for x in aliases],
        input_normalization=normalization,gc_mtf_history=history,dropout=0.0)
    torch.manual_seed(8193);old=_make_dynamic_m5_model(**kwargs,entry_execution_profile=_reference(parent)).eval()
    torch.manual_seed(8193);new=_make_dynamic_m5_model(**kwargs,entry_execution_profile=_reference(candidate)).eval()
    old_parameters=dict(old.named_parameters());new_parameters=dict(new.named_parameters())
    assert set(old_parameters)<set(new_parameters)
    for name,value in old_parameters.items():assert torch.equal(value,new_parameters[name]),name
    with torch.no_grad():
        previous=old(**inputs)['entry_q_joint_hidden'];initial=new(**inputs)['entry_q_joint_hidden']
    assert torch.equal(previous,initial)
    projection=new.mtf_family_proj[MOMENTUM_FAMILY]
    assert torch.count_nonzero(projection.weight)==0 and torch.count_nonzero(projection.bias)==0
    new.zero_grad(set_to_none=True)
    new(**inputs)['entry_q_joint_hidden'].square().mean().backward()
    assert projection.weight.grad is not None and torch.isfinite(projection.weight.grad).all()
    assert torch.count_nonzero(projection.weight.grad)>0
    with torch.no_grad():projection.weight.add_(projection.weight.grad,alpha=-0.1)
    perturbed={k:v.clone() for k,v in inputs.items()}
    perturbed['seq_m5'][:,-1,mtf.index('rsi14_centered')]+=0.25
    with torch.no_grad():
        a=new(**inputs)['entry_q_joint_hidden'];b=new(**perturbed)['entry_q_joint_hidden']
    assert not torch.equal(a,b)


def test_dense_feature_view_binds_separate_price_profile_and_source(bound_case, monkeypatch):
    from copy import deepcopy
    from gx1.contracts import entry_native_direction_feature_view_v1 as view_owner
    from tests.test_entry_native_direction_feature_view_v1 import _constructor_spy, _ref
    args=deepcopy(bound_case['arguments'])
    args['source_bindings'].update({k:_ref(k+'.py') for k in view_owner.DENSE_SOURCE_REF_KEYS})
    original_view=view_owner.bind_native_direction_feature_view(**args,dense_minutes=True)
    args['source_bindings'].update({k:_ref(k+'.py') for k in view_owner.PRICE_MOMENTUM_SOURCE_REF_KEYS})
    with pytest.raises(RuntimeError,match='OBJECTIVE_MODE'):
        view_owner.bind_native_direction_feature_view(**args,price_momentum=True)
    view=view_owner.bind_native_direction_feature_view(**args,dense_minutes=True,price_momentum=True)
    assert view['price_momentum_profile'] is True
    assert view['entry_execution_profile']['name']==PROFILE_NAME
    assert view['prefix_normalization']==original_view['prefix_normalization']
    assert view['physical_field_names_sha256']==original_view['physical_field_names_sha256']
    assert view['research_objective']==original_view['research_objective']
    calls,norm_calls=_constructor_spy(monkeypatch)
    model=view_owner.build_native_direction_research_model(view,
        prefix_normalization=args['prefix_normalization'],constructor_metadata=args['constructor_metadata'],device=torch.device('cpu'))
    assert model.backbone.entry_execution_profile==view['entry_execution_profile']
    assert len(calls)==len(norm_calls)==1
    missing=deepcopy(args);missing['source_bindings'].pop('price_momentum_profile_source')
    with pytest.raises(RuntimeError,match='SOURCE_REF_KEYS'):
        view_owner.bind_native_direction_feature_view(**missing,dense_minutes=True,price_momentum=True)
