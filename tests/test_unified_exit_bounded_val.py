from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gx1.contracts.local_random_access_campaign_v2 import canonical_sha256, file_sha256
from gx1.contracts.unified_exit_bounded_val_cohort_v1 import (
    build_bounded_val_cohort, require_bounded_val_cohort,
)
from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_complete_val_observation
from gx1.contracts.unified_exit_random_access_val_evaluator_v1 import (
    require_random_access_val_evaluation_result_v1,
    run_resumable_random_access_val_evaluation_v1,
)
from tests.test_unified_exit_random_access_val_evaluator_v1 import _checkpoint_binding, _entry_policy, _with_route_outputs
from tests.test_unified_exit_random_access_val_rollout_v1 import _fixture, VAL_ENTRY_COHORT_SIZE, economics


def _cohort(tmp_path, ids=(7, 31, 5000)):
    index = tmp_path / 'val.index'
    index.write_bytes(b'fixed-source-index')
    plan = tmp_path / 'frozen.json'
    value = {
        'decision': 'FROZEN_CANDIDATE_AND_COHORT_NOT_LAUNCH_AUTHORITY',
        'fit_frozen': True, 'further_tuning_on_existing_train_control_forbidden': True,
        'execution': {'training_enabled': False, 'optimizer_steps': 0},
        'selection': {'population_rows': VAL_ENTRY_COHORT_SIZE, 'requested_rows': len(ids),
                      'selection_uses_outcomes': False, 'rows': [{'entry_row_index': i} for i in ids]},
        'val_index': {'path': str(index), 'sha256': file_sha256(index)},
    }
    plan.write_text(json.dumps(value))
    return build_bounded_val_cohort({'path': str(plan), 'sha256': file_sha256(plan)})


@pytest.mark.parametrize("accounting", ["terminal_cash_v2", economics.LIQUIDATION_ADVANTAGE_REWARD_ACCOUNTING])
def test_sparse_cohort_pause_resume_preserves_economics_and_exact_coverage(tmp_path, accounting):
    scope = _cohort(tmp_path)
    thresholds = np.ones((VAL_ENTRY_COHORT_SIZE, 2), dtype=np.float32)
    thresholds[31] = [0, 2]
    thresholds[5000] = [99, 99]
    model, representations, adapter, contract = _fixture(
        thresholds=thresholds, counts=np.full(VAL_ENTRY_COHORT_SIZE, 3), evaluation_cohort=scope,
        reward_accounting=accounting)
    model = _with_route_outputs(model)
    binding = _checkpoint_binding(contract, adapter, tmp_path)
    kwargs = dict(model=model, entry_decision_representations=representations, adapter=adapter,
                  checkpoint_binding=binding, entry_route_diagnostics={},
                  entry_policy_decisions=_entry_policy(adapter, binding), policy_batch_size=4)
    paused = run_resumable_random_access_val_evaluation_v1(
        **kwargs, progress_path=tmp_path/'resume.json', result_path=tmp_path/'resumed-result.json',
        max_forwards_this_invocation=1)
    assert paused['decision'] == 'PAUSED_RESUMABLE'
    progress = json.loads((tmp_path/'resume.json').read_text())
    assert len(progress['active_side_mask']) == len(progress['trade_accumulators']) == 3
    resumed = run_resumable_random_access_val_evaluation_v1(
        **kwargs, progress_path=tmp_path/'resume.json', result_path=tmp_path/'resumed-result.json',
        max_forwards_this_invocation=100)
    direct = run_resumable_random_access_val_evaluation_v1(
        **kwargs, progress_path=tmp_path/'direct.json', result_path=tmp_path/'direct-result.json',
        max_forwards_this_invocation=100)
    assert resumed['trade_outcomes'] == direct['trade_outcomes']
    assert resumed['entry_exit_policy_metrics'] == direct['entry_exit_policy_metrics']
    assert resumed['entry_pair_cohort_size'] == 3 and resumed['side_trade_count'] == 6
    assert [r['entry_row_index'] for r in resumed['trade_outcomes']] == [7, 7, 31, 31, 5000, 5000]
    assert resumed['right_censored_side_trade_count'] == 2
    assert resumed['entry_exit_policy_metrics']['selected_non_exited_count'] == 1
    assert resumed['source_population_fully_evaluated'] is False
    assert resumed['evaluation_scope'] == 'bounded_development_val'
    # Complete coverage of a selected cohort must never satisfy the full-VAL gate.
    with pytest.raises(RuntimeError, match='FULL_VAL_OUTCOMES_REQUIRED'):
        require_complete_val_observation(resumed)
    checked = require_random_access_val_evaluation_result_v1(
        resumed, rollout_contract_sha256=contract['contract_sha256'],
        checkpoint_binding_sha256=binding['binding_sha256'],
        execution_contract_sha256=resumed['execution_contract_sha256'])
    assert checked == resumed
    bad = copy.deepcopy(resumed)
    bad['source_population_fully_evaluated'] = True
    bad['semantic_result_sha256'] = canonical_sha256({k:v for k,v in bad.items() if k!='semantic_result_sha256'})
    with pytest.raises(RuntimeError, match='COHORT_MISMATCH'):
        require_random_access_val_evaluation_result_v1(
            bad, rollout_contract_sha256=contract['contract_sha256'],
            checkpoint_binding_sha256=binding['binding_sha256'],
            execution_contract_sha256=resumed['execution_contract_sha256'])


@pytest.mark.parametrize('ids', [(7,7), (31,7), (-1,7), (7,5508), (True,7)])
def test_bound_cohort_rejects_ambiguous_source_rows(tmp_path, ids):
    with pytest.raises(RuntimeError, match='ENTRY_IDENTITIES_INVALID'):
        _cohort(tmp_path, ids)


def test_bound_cohort_rejects_plan_or_row_changes(tmp_path):
    scope = _cohort(tmp_path)
    bad = copy.deepcopy(scope)
    bad['entry_row_indices'][0] = 8
    bad['cohort_sha256'] = canonical_sha256({k:v for k,v in bad.items() if k!='cohort_sha256'})
    with pytest.raises(RuntimeError, match='BINDING_MISMATCH'):
        require_bounded_val_cohort(bad)
    (tmp_path/'frozen.json').write_text('{}')
    with pytest.raises(RuntimeError):
        require_bounded_val_cohort(scope)


def test_subset_without_a_frozen_binding_still_fails_closed(tmp_path):
    scope = _cohort(tmp_path)
    model, representations, adapter, contract = _fixture(
        thresholds=np.ones((VAL_ENTRY_COHORT_SIZE,2)),
        counts=np.full(VAL_ENTRY_COHORT_SIZE,2), evaluation_cohort=scope)
    from gx1.contracts.unified_exit_random_access_val_rollout_v1 import require_random_access_val_rollout_contract
    unbound = {k:v for k,v in contract.items() if k not in {'evaluation_cohort','contract_sha256'}}
    unbound['contract_sha256'] = canonical_sha256(unbound)
    with pytest.raises(RuntimeError, match='CONTRACT_INVALID'):
        require_random_access_val_rollout_contract(unbound)


def _readout_plan(tmp_path):
    import torch
    from tests.test_entry_v10_ctx_model_shapes import _make_model
    _cohort(tmp_path)
    model = _make_model(dropout=0.1)
    base = {k:v.clone() for k,v in model.state_dict().items()}
    parent = tmp_path/'parent.pt'
    torch.save({'schema_version':'gx1_candidate_training_session_v1', 'model_state':base, 'target_model_state':base}, parent)
    ep = tmp_path/'entry.pt'
    xp = tmp_path/'exit.pt'
    torch.save({'weight':torch.full_like(base['head_entry_action_q.weight'], 0.25),
                'bias':torch.full_like(base['head_entry_action_q.bias'], 0.5)}, ep)
    torch.save({'difference_weight':torch.full_like(base['head_exit_action.weight'][0], 0.75),
                'difference_bias':torch.full_like(base['head_exit_action.bias'][0], 0.125)}, xp)
    path = tmp_path/'frozen.json'
    plan = json.loads(path.read_text())
    for name,p in [('base_online_checkpoint',parent),('entry_readout',ep),('exit_readout',xp)]:
        plan[name] = {'path':str(p),'sha256':file_sha256(p)}
    path.write_text(json.dumps(plan))
    return model,base,{'path':str(path),'sha256':file_sha256(path)},plan


def test_frozen_online_binding_preserves_backbone_buffers_rng_and_parent(tmp_path):
    import torch
    from gx1.contracts.unified_exit_random_access_val_checkpoint_v1 import (
        bind_frozen_readout_checkpoint_v1, require_selected_weight_ema_checkpoint_binding_v1,
    )
    model,base,plan_binding,plan = _readout_plan(tmp_path)
    target = copy.deepcopy(model)
    rng = torch.get_rng_state().clone()
    binding = bind_frozen_readout_checkpoint_v1(plan_binding=plan_binding,arm='candidate',
                                               model=model,target_model=target)
    assert torch.equal(rng,torch.get_rng_state())
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert binding['ema_used'] is False and binding['optimizer_or_scheduler_loaded'] is False
    assert binding['model_variant'] == 'frozen_online_readout'
    assert file_sha256(Path(plan['base_online_checkpoint']['path'])) == plan['base_online_checkpoint']['sha256']
    changed = {k for k,v in model.state_dict().items() if not torch.equal(v,base[k])}
    assert changed == set(binding['modified_parameter_names'])
    for k,v in model.state_dict().items():
        if not k.startswith(('head_entry_action_q.','head_exit_action.')):
            assert torch.equal(v,base[k])
    assert torch.equal(target.head_entry_action_q.weight,base['head_entry_action_q.weight'])
    assert torch.equal(target.head_exit_action.weight,model.head_exit_action.weight)
    assert require_selected_weight_ema_checkpoint_binding_v1(binding) == binding
    mislabeled = dict(binding,model_variant='weight_ema')
    with pytest.raises(RuntimeError,match='BINDING_MISMATCH'):
        require_selected_weight_ema_checkpoint_binding_v1(mislabeled)
    baseline = bind_frozen_readout_checkpoint_v1(plan_binding=plan_binding,arm='baseline',model=model)
    assert baseline['modified_parameter_names'] == []
    assert all(torch.equal(v,base[k]) for k,v in model.state_dict().items())


def test_frozen_online_binding_rejects_wrong_head_geometry_before_loading(tmp_path):
    import torch
    from pathlib import Path
    from gx1.contracts.unified_exit_random_access_val_checkpoint_v1 import bind_frozen_readout_checkpoint_v1
    model,base,binding,plan = _readout_plan(tmp_path)
    path = Path(plan['entry_readout']['path'])
    torch.save({'weight':torch.ones(1),'bias':torch.ones(1)},path)
    plan['entry_readout']['sha256'] = file_sha256(path)
    pp = Path(binding['path']);pp.write_text(json.dumps(plan));binding['sha256'] = file_sha256(pp)
    with pytest.raises(RuntimeError,match='ENTRY_TENSOR_INVALID'):
        bind_frozen_readout_checkpoint_v1(plan_binding=binding,arm='candidate',model=model)
    assert all(torch.equal(v,base[k]) for k,v in model.state_dict().items())


def test_bounded_entry_forward_keeps_sparse_child_and_parent_coordinates(tmp_path, monkeypatch):
    import torch
    from gx1.scripts import run_unified_exit_random_access_val_v1 as cli
    scope = _cohort(tmp_path)
    child_rows = scope['entry_row_indices']
    parent_rows = [10000 + i for i in child_rows]
    observed = []

    class Rows(torch.utils.data.Dataset):
        def __len__(self):
            return 20000
        def __getitem__(self, row):
            return {'entry_row_index':row, 'seq_x':torch.tensor([float(row)]),
                    'snap_x':torch.zeros(1), 'ctx_cont':torch.zeros(1),
                    'ctx_cat':torch.zeros(1,dtype=torch.long)}

    model = torch.nn.Linear(1,3).eval().requires_grad_(False)
    target = copy.deepcopy(model)
    def forward(_model, seq, snap, **kwargs):
        return {cli.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY: seq,
                'entry_action_q_bps':seq.expand(-1,3).contiguous()}
    def anchors(**kwargs):
        observed.extend(kwargs['child_rows'])
        count = len(kwargs['child_rows'])
        return torch.ones(count,3), torch.ones(count,3,dtype=torch.bool), {'binding_sha256':'a'*64}
    monkeypatch.setattr(cli, '_model_forward_fp32', forward)
    monkeypatch.setattr(cli, '_multi_tf_kwargs_from_batch', lambda *args: {})
    monkeypatch.setattr(cli, '_candidate_anchor_targets', anchors)
    monkeypatch.setattr(cli, '_new_active_head_epoch_accumulator', lambda: {})
    monkeypatch.setattr(cli, '_accumulate_active_head_epoch', lambda *args: None)
    monkeypatch.setattr(cli, '_active_head_epoch_diagnostics', lambda *args: ({},None))
    monkeypatch.setattr(cli, 'accumulate_route_diagnostics_v1', lambda *args: None)
    monkeypatch.setattr(cli, 'finalize_route_diagnostics_v1', lambda *args: {})
    kwargs = dict(model=model,dataset=Rows(),parent_rows=parent_rows,device=torch.device('cpu'),
                  batch_size=2,candidate_target_model=target,candidate_state_factory=object(),
                  candidate_child_rows=child_rows,evaluation_cohort=scope)
    reps,diag,q = cli._entry_representations(**kwargs)
    assert observed == child_rows
    assert reps[:,0].tolist() == parent_rows and q[:,0].tolist() == parent_rows
    rows = diag['bounded_entry_observations']
    assert [r['entry_row_index'] for r in rows] == child_rows
    assert [r['parent_entry_row_index'] for r in rows] == parent_rows
    assert all(r['target_q_bps'] == [1.,1.,1.] for r in rows)
    assert diag['surface'] == 'entry_bounded_cohort_forward'
    with pytest.raises(RuntimeError,match='TARGET_NOT_FROZEN_OR_COHORT_INVALID'):
        cli._entry_representations(**{**kwargs,'candidate_child_rows':[0,1,2]})
    with pytest.raises(RuntimeError,match='PARENT_ENTRY_MAPPING_INVALID'):
        cli._entry_representations(**{**kwargs,'evaluation_cohort':None})
