"""Identical observed reference targets across TRAIN/control; synthetic only."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from gx1.scripts import run_unified_exit_random_access_val_v1 as val
from gx1.contracts.unified_exit_reference_policy_v1 import reference_policy_contract
from gx1.contracts.unified_exit_bounded_val_cohort_v1 import build_chronological_control_cohort
from tests.test_liquidation_relative_learning_v1 import relative_fixture,_TwoRowHead
from tests.test_coherent_reference_entry_v1 import _inputs,_collate,_run,CUTOFF
from tests.test_unified_exit_random_access_training_v1 import _normalization
from tests import test_unified_exit_random_access_state_view_v1 as states
from tests.test_native_chronological_control_source import _control_plan,_binding


def _factory(batch,count):
    clock=states._clock();authority=states._authority(clock,known=True)
    entry={'entry_row_index':0,'entry_m1_start_row':479,'available_state_count':count,
           'entry_episode_binding_sha256':batch['entry_episode_binding_sha256'][0],
           'entry_fill_binding_sha256':batch['entry_fill_binding_sha256'][0]}
    calls=[]
    def materialize(entry,offset):
        calls.append(offset)
        at_boundary=offset==count-1
        view=states._materialize(offset-1 if at_boundary else offset,counts=(count,count),reference_policy=reference_policy_contract())
        return view['successor' if at_boundary else 'current']
    return SimpleNamespace(entries=[entry],times=clock,closure=authority,source_split='train',
        economics_objective_contract=states._objective(),economic_step_provider=states._EconomicProvider(authority['artifact_sha256']),
        economic_step_manifest={'manifest_sha256':'d'*64,'economic_step_model_sha256':'b'*64,'economic_step_source_manifest_sha256':'c'*64},
        normalization=_normalization(),materialize_state=materialize,calls=calls)


@pytest.mark.parametrize('count',[2,3,121,122,600])
def test_control_entry_and_exit_equal_native_train_reference_exactly(relative_fixture,count):
    item,bindings=_inputs(counts=(count,count));batch=_collate(item,bindings);train=_run(batch)
    factory=_factory(batch,count);model=_TwoRowHead().eval();teacher=copy.deepcopy(model).requires_grad_(False)
    output={val.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY:torch.tensor([[3.]],dtype=torch.float32)}
    with torch.inference_mode():
        rows,target=val._bounded_reference_exit_observations(model=model,boundary_model=teacher,entry_output=output,
            boundary_entry_output=output,state_factory=factory,child_rows=[0],device=torch.device('cpu'),
            reference_policy=reference_policy_contract(),reference_cutoff_time_ns=CUTOFF,return_reference_targets=True)
        before=list(factory.calls)
        targets,valid,_=val._candidate_anchor_targets(target_model=teacher,target_entry_output=output,state_factory=factory,
            child_rows=[0],device=torch.device('cpu'),reference_hold_targets=target['hold_target_bps'],reference_policy=reference_policy_contract())
    torch.testing.assert_close(target['hold_target_bps'],train['entry_reference_target_evidence']['hold_target_bps'],rtol=0,atol=0)
    torch.testing.assert_close(targets[0],train['entry_targets'][1],rtol=0,atol=0)
    assert factory.calls==before==[0,min(120,count-1)]
    assert teacher.calls==model.calls==1
    assert rows[0]['target_hold_bps']==target['hold_target_bps'][0].tolist()
    assert rows[0]['boundary_censored']==[count<=121]*2
    assert min(rows[0]['boundary_bootstrap_weight'])>0
    assert valid.tolist()==[[True,True,True]] and targets[0,2]==0 and not targets.requires_grad
    assert all(p.grad is None for p in teacher.parameters())


def test_control_reference_cutoff_checks_boundary_close_before_prices_or_states(relative_fixture):
    batch=_collate(*_inputs());factory=_factory(batch,600)
    end=int(factory.times.asi8[479+120])+60_000_000_000
    lengths,policy,_,trace=val._bounded_reference_trace(state_factory=factory,child_rows=[0],device=torch.device('cpu'),
        reference_policy=reference_policy_contract(),reference_cutoff_time_ns=end)
    assert lengths==[120] and not bool(trace['boundary_right_censored_mask'].any())
    def forbidden(*a,**kw):raise AssertionError('out-of-period data accessed')
    factory.economic_step_provider=SimpleNamespace(materialize_training_projection=forbidden)
    factory.materialize_state=forbidden
    with pytest.raises(RuntimeError,match='SUPPORT_CROSSES_CUTOFF'):
        val._bounded_reference_exit_observations(model=None,boundary_model=None,entry_output={},boundary_entry_output={},
            state_factory=factory,child_rows=[0],device=torch.device('cpu'),reference_policy=policy,
            reference_cutoff_time_ns=end-1,return_reference_targets=True)
    with pytest.raises(RuntimeError,match='CUTOFF_BINDING_INVALID'):
        val._bounded_reference_trace(state_factory=factory,child_rows=[0],device=torch.device('cpu'),reference_cutoff_time_ns=end)


@pytest.mark.parametrize('fault',['shape','dtype','gradient','nonfinite','economics'])
def test_reference_entry_rejects_invalid_detached_value_before_model_call(relative_fixture,fault):
    factory=_factory(_collate(*_inputs()),600);q=torch.ones(1,2)
    if fault=='shape':q=torch.ones(1,3)
    elif fault=='dtype':q=q.double()
    elif fault=='gradient':q.requires_grad_(True)
    elif fault=='nonfinite':q[0,0]=float('nan')
    else:factory.economics_objective_contract={}
    with pytest.raises(RuntimeError,match='REFERENCE_ANCHOR_INVALID'):
        val._candidate_anchor_targets(target_model=None,target_entry_output={val.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY:torch.ones(1,1)},
            state_factory=factory,child_rows=[0],device=torch.device('cpu'),reference_hold_targets=q,reference_policy=reference_policy_contract())
    assert factory.calls==[]


def _context(tmp_path):
    binding,frame,children,parents,design=_control_plan(tmp_path)
    design['targets']={'reference_policy':reference_policy_contract()}
    p=Path(binding['path']);p.write_text(json.dumps(design));scope=build_chronological_control_cohort(_binding(p))
    model=torch.nn.Linear(1,3).eval().requires_grad_(False);teacher=copy.deepcopy(model)
    class Rows(torch.utils.data.Dataset):
        def __len__(self):return 10000
        def __getitem__(self,row):
            return {'entry_row_index':row,'seq_x':torch.tensor([float(row)]),'snap_x':torch.zeros(1),
                    'ctx_cont':torch.zeros(1),'ctx_cat':torch.zeros(1,dtype=torch.long)}
    entries=[{'entry_row_index':i,'entry_episode_binding_sha256':'1'*64,'entry_fill_binding_sha256':'2'*64} for i in range(6000)]
    factory=SimpleNamespace(source_split='train',entries=entries,economics_objective_contract={'reward_accounting':'liquidation_advantage_v1'},
        economic_step_provider=SimpleNamespace(materialize_training_projection=lambda *a:{'exit_reward_bps':np.array([-4.])}))
    return dict(model=model,dataset=Rows(),parent_rows=parents.tolist(),device=torch.device('cpu'),batch_size=64,
        candidate_target_model=teacher,exit_boundary_model=teacher,candidate_state_factory=factory,
        candidate_child_rows=children.tolist(),evaluation_cohort=scope)


@pytest.mark.parametrize('fault',['missing_teacher','changed_teacher','train_mode','wrong_parent','wrong_source'])
def test_chronological_control_requires_same_frozen_teacher_and_coordinates_before_forward(tmp_path,monkeypatch,fault):
    args=_context(tmp_path)
    if fault=='missing_teacher':args['exit_boundary_model']=None
    elif fault=='changed_teacher':
        args['exit_boundary_model']=copy.deepcopy(args['candidate_target_model'])
        with torch.no_grad():args['exit_boundary_model'].weight.add_(1.)
    elif fault=='train_mode':args['candidate_target_model'].train()
    elif fault=='wrong_parent':args['parent_rows'][0]+=1
    else:args['candidate_state_factory'].source_split='val'
    def forbidden(*a,**kw):raise AssertionError('forward happened before binding checks')
    monkeypatch.setattr(val,'_model_forward_fp32',forbidden)
    with pytest.raises(RuntimeError):val._entry_representations(**args)


def test_entry_control_integration_reuses_same_exit_reference_and_exact_sparse_ids(tmp_path,monkeypatch):
    args=_context(tmp_path);forward_calls=[];reference_rows=[]
    def forward(model,seq,snap,**kw):
        forward_calls.append(model)
        return {val.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY:seq,'entry_action_q_bps':seq.expand(-1,3).contiguous()}
    def reference(**kw):
        assert kw['return_reference_targets'] is True
        assert kw['reference_policy']==reference_policy_contract()
        assert kw['reference_cutoff_time_ns']==int(pd.Timestamp('2026-06-01T00:00Z').value)
        assert kw['boundary_model'] is args['candidate_target_model']
        children=kw['child_rows'];q=torch.tensor([[float(i%13),-2.] for i in children])
        rows=[{'entry_row_index':i,'state_index':0,'target_hold_bps':v} for i,v in zip(children,q.tolist())]
        reference_rows.extend(rows);return rows,{'hold_target_bps':q}
    monkeypatch.setattr(val,'_model_forward_fp32',forward)
    monkeypatch.setattr(val,'_multi_tf_kwargs_from_batch',lambda *a:{})
    monkeypatch.setattr(val,'_bounded_reference_exit_observations',reference)
    monkeypatch.setattr(val,'_new_active_head_epoch_accumulator',lambda:{})
    monkeypatch.setattr(val,'_accumulate_active_head_epoch',lambda *a:None)
    monkeypatch.setattr(val,'_active_head_epoch_diagnostics',lambda *a:({},None))
    monkeypatch.setattr(val,'accumulate_route_diagnostics_v1',lambda *a:None)
    monkeypatch.setattr(val,'finalize_route_diagnostics_v1',lambda *a:{})
    reps,diag,q=val._entry_representations(**args)
    assert len(forward_calls)==8 and forward_calls.count(args['candidate_target_model'])==4
    assert reps[:,0].tolist()==args['parent_rows']
    assert diag['bounded_exit_anchor_observations']==reference_rows
    rows=diag['bounded_entry_observations']
    assert [r['entry_row_index'] for r in rows]==args['candidate_child_rows']
    for row,exit_row in zip(rows,reference_rows):
        expected=torch.tensor(exit_row['target_hold_bps'],dtype=torch.float32)*(119/120)-4
        assert row['target_q_bps']==expected.tolist()+[0.]
    evidence=diag['candidate_active_head_evidence']
    assert evidence['entry_q_target_semantics']=='observed_reference_anchor_V_mu_without_hindsight_action'
    assert evidence['reference_measurement']['frozen_design']==args['evaluation_cohort']['plan']
    assert evidence['target_updated_from_val_or_test'] is False


@pytest.mark.parametrize('fault',['missing','unpaired','changed'])
def test_reference_entry_requires_same_declared_policy(relative_fixture,fault):
    factory=_factory(_collate(*_inputs()),600);q=torch.ones(1,2);policy=reference_policy_contract()
    if fault=='missing':policy=None
    elif fault=='unpaired':q=None
    else:policy['hold_probability_numerator']=118
    with pytest.raises(RuntimeError,match='REFERENCE_'):
        val._candidate_anchor_targets(target_model=None,
            target_entry_output={val.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY:torch.ones(1,1)},
            state_factory=factory,child_rows=[0],device=torch.device('cpu'),
            reference_hold_targets=q,reference_policy=policy)
    assert factory.calls==[]
