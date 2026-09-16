from __future__ import annotations

import copy

import pytest
import torch

from gx1.contracts.unified_exit_reference_policy_v1 import (
    build_reference_policy_hold_targets, reference_policy_contract,
    require_reference_policy_contract,
)
from gx1.contracts.unified_exit_economics_objective_v2 import (
    PROPER_POLICY_CERTIFICATE_SCHEMA_VERSION, require_proper_policy_certificate,
    seal_proper_policy_certificate,
)


def inputs(steps=5, rows=1):
    return dict(
        policy=reference_policy_contract(),
        hold_reward_bps=torch.tensor([[[(-1.)**i*(i+1), (i-2.)/3] for i in range(steps)]]*rows),
        elapsed_wall_clock_gamma=torch.tensor([[.99-(i%7)/100 for i in range(steps)]]*rows),
        transition_available_mask=torch.ones((rows,steps),dtype=torch.bool),
        successor_terminal_mask=torch.zeros((rows,steps,2),dtype=torch.bool),
        boundary_action_q_bps=torch.tensor([[[-7.,0.],[9.,0.]]]*rows),
        boundary_action_valid_mask=torch.ones((rows,2,2),dtype=torch.bool),
        boundary_right_censored_mask=torch.zeros((rows,2),dtype=torch.bool),
    )


def explicit_stopping_expectation(rewards, gammas, boundary):
    # Enumerate every first-EXIT time, plus the surviving boundary continuation.
    # This deliberately differs from the weighted-reward implementation.
    p=119/120;total=0.;prefix=0.;discount=1.
    for k,(r,g) in enumerate(zip(rewards,gammas),start=1):
        prefix+=discount*r;discount*=g
        total+=(p**(k-1))*(1-p)*prefix
    return total+p**len(rewards)*(prefix+discount*boundary)


@pytest.mark.parametrize('steps',[1,2,5,120])
def test_matches_explicit_expectation_over_all_stopping_times(steps):
    data=inputs(steps,rows=2)
    data['hold_reward_bps'][1]*=-3
    actual=build_reference_policy_hold_targets(**data)
    for row in range(2):
        for side in range(2):
            expected=explicit_stopping_expectation(
                data['hold_reward_bps'][row,:,side].tolist(),
                data['elapsed_wall_clock_gamma'][row].tolist(),
                float(data['boundary_action_q_bps'][row,side,0]),
            )
            assert float(actual['hold_target_bps'][row,side])==pytest.approx(expected,rel=3e-6,abs=3e-5)
    assert torch.equal(actual['hold_target_bps'],actual['observed_reward_component_bps']+actual['bootstrap_component_bps'])


def test_censored_prefix_uses_its_boundary_and_preserves_negative_value():
    data=inputs(5,rows=2);data['transition_available_mask'][0,2:]=False
    data['boundary_right_censored_mask'][0]=True
    actual=build_reference_policy_hold_targets(**data)
    short=copy.deepcopy(data)
    for key in ['hold_reward_bps','elapsed_wall_clock_gamma','transition_available_mask','successor_terminal_mask']:
        short[key]=short[key][:1,:2]
    for key in ['boundary_action_q_bps','boundary_action_valid_mask','boundary_right_censored_mask']:
        short[key]=short[key][:1]
    expected=build_reference_policy_hold_targets(**short)
    torch.testing.assert_close(actual['hold_target_bps'][0],expected['hold_target_bps'][0],rtol=0,atol=0)
    assert actual['bootstrap_component_bps'][0,0]<0  # Never max with EXIT=0.
    data['hold_reward_bps'][0,2:]=1e10
    torch.testing.assert_close(build_reference_policy_hold_targets(**data)['hold_target_bps'],actual['hold_target_bps'],rtol=0,atol=0)


def test_real_terminal_stops_only_its_side_not_the_other_side():
    data=inputs();data['successor_terminal_mask'][0,1:,0]=True
    data['boundary_action_valid_mask'][0,0,0]=False
    data['boundary_action_q_bps'][0,0,0]=float('nan')
    actual=build_reference_policy_hold_targets(**data)
    expected=float(data['hold_reward_bps'][0,0,0]+(119/120)*data['elapsed_wall_clock_gamma'][0,0]*data['hold_reward_bps'][0,1,0])
    assert actual['hold_target_bps'][0,0]==pytest.approx(expected)
    assert actual['bootstrap_component_bps'][0,0]==0 and actual['bootstrap_component_bps'][0,1]>0
    data['hold_reward_bps'][0,2:,0]=1e10
    assert build_reference_policy_hold_targets(**data)['hold_target_bps'][0,0]==actual['hold_target_bps'][0,0]


def test_120_is_computation_boundary_not_a_holding_limit():
    data=inputs(120);data['elapsed_wall_clock_gamma'].fill_(1);data['hold_reward_bps'].zero_()
    result=build_reference_policy_hold_targets(**data)
    expected=(119/120)**120
    assert result['boundary_bootstrap_weight'][0,0]==pytest.approx(expected,rel=5e-6)
    assert .36<expected<.37 and result['hold_target_bps'][0,0]<0


def test_side_swap_commutes_and_targets_have_no_gradient_or_rng_draw():
    data=inputs();data['hold_reward_bps'].requires_grad_(True);data['boundary_action_q_bps'].requires_grad_(True)
    before=torch.random.get_rng_state().clone();result=build_reference_policy_hold_targets(**data)
    swapped=copy.deepcopy(data)
    for key in ['hold_reward_bps','successor_terminal_mask']:
        swapped[key]=swapped[key].flip(2)
    for key in ['boundary_action_q_bps','boundary_action_valid_mask','boundary_right_censored_mask']:
        swapped[key]=swapped[key].flip(1)
    flipped=build_reference_policy_hold_targets(**swapped)
    torch.testing.assert_close(result['hold_target_bps'].flip(1),flipped['hold_target_bps'],rtol=0,atol=0)
    assert not result['hold_target_bps'].requires_grad
    assert torch.equal(before,torch.random.get_rng_state())


def test_reference_has_finite_expected_absorption_without_maximum_hold():
    p=119/120;policy=reference_policy_contract()
    certificate=seal_proper_policy_certificate(dict(
        schema_version=PROPER_POLICY_CERTIFICATE_SCHEMA_VERSION,
        proof_kind='absorbing_markov_chain_policy_v1',policy_sha256=policy['policy_sha256'],
        economic_terminal_policy_sha256='e'*64,transition_matrix=[[p,1-p],[0.,1.]],
        initial_distribution=[1.,0.],terminal_state_indices=[1],
        claimed_nonterminal_spectral_radius=p,claimed_maximum_expected_steps_to_absorption=120.,
        numeric_tolerance=1e-9,
    ))
    proof=require_proper_policy_certificate(certificate,expected_policy_sha256=policy['policy_sha256'])
    assert proof['verified_maximum_expected_steps_to_absorption']==pytest.approx(120)
    assert p**1000>0  # Not a hard holding-time cutoff.


@pytest.mark.parametrize('mutation',['probability','numeric_type','semantics','sha','extra'])
def test_policy_identity_rejects_silent_changes(mutation):
    policy=reference_policy_contract()
    if mutation=='probability':policy['hold_probability_numerator']=118
    elif mutation=='numeric_type':policy['hold_probability_numerator']=119.0
    elif mutation=='semantics':policy['value_semantics']='optimal_Q'
    elif mutation=='sha':policy['policy_sha256']='0'*64
    else:policy['extra']=True
    with pytest.raises(RuntimeError,match='CONTRACT_INVALID'):require_reference_policy_contract(policy)


@pytest.mark.parametrize('mutation',['gap','empty','gamma','nan_reward','dtype','exit_q','valid_q_nan','exit_invalid','fake_terminal','censored_terminal','resurrection','too_long'])
def test_malformed_evidence_fails_closed(mutation):
    data=inputs(121 if mutation=='too_long' else 5)
    if mutation=='gap':data['transition_available_mask'][0,1]=False
    elif mutation=='empty':data['transition_available_mask'].zero_()
    elif mutation=='gamma':data['elapsed_wall_clock_gamma'][0,0]=0
    elif mutation=='nan_reward':data['hold_reward_bps'][0,0,0]=float('nan')
    elif mutation=='dtype':data['hold_reward_bps']=data['hold_reward_bps'].double()
    elif mutation=='exit_q':data['boundary_action_q_bps'][0,0,1]=1
    elif mutation=='valid_q_nan':data['boundary_action_q_bps'][0,0,0]=float('nan')
    elif mutation=='exit_invalid':data['boundary_action_valid_mask'][0,0,1]=False
    elif mutation=='fake_terminal':data['boundary_action_valid_mask'][0,0,0]=False
    elif mutation=='censored_terminal':
        data['successor_terminal_mask'][0,-1,0]=True;data['boundary_action_valid_mask'][0,0,0]=False;data['boundary_right_censored_mask'][0,0]=True
    elif mutation=='resurrection':data['successor_terminal_mask'][0,0,0]=True
    with pytest.raises(RuntimeError,match='UNIFIED_EXIT_REFERENCE_'):build_reference_policy_hold_targets(**data)
