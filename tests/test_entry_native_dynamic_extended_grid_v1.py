"""Dense extended-grid contracts, loss and synthetic reports; no model or backward."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest
import torch

from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as targets
from gx1.contracts import entry_dynamic_mtf_state_report_v1 as report
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as campaign
from gx1.contracts import entry_native_direction_research_campaign_v1 as private
from gx1.contracts import entry_native_direction_research_v1 as owner
from gx1.contracts import entry_model_native_train_recipe_v1 as recipes
from tests.test_entry_native_direction_feature_view_v1 import bound_case


def test_extended_grid_keeps_every_minute_and_previous_identities():
    old = owner.dynamic_mtf_state_research_contract()
    current = owner.dynamic_mtf_state_research_contract(current_state_input=True, first_state_loss_weight=1.0)
    extended = owner.dynamic_mtf_state_research_contract(current_state_input=True, first_state_loss_weight=1.0,
                                                          extended_forecast_minutes=10080)
    assert old['contract_sha256'] == 'fdfc47ed7b1d6ddbacdd2a8388a09049d0017555d8f79afaa88b5f200e8f532b'
    assert current['contract_sha256'] == '7b62e039ac38d616106180a4b11ee614f0f5dbf857381e4cbdb3dffdbacfee33'
    assert extended['contract_sha256'] not in (old['contract_sha256'], current['contract_sha256'])
    assert extended['lead_minutes'] == list(range(1, 10081))
    assert extended['target_contract'] == old['target_contract']
    assert extended['extended_runtime_and_storage_qualified'] is False
    assert extended['query_coordinates']['lead_center_minutes'] == 5040.5
    assert extended['query_sampling'] is False


@pytest.mark.parametrize('bad', [1440, 0, -1, True, 1441.0, '1441', 3000000, 100000000])
def test_extended_grid_rejects_invalid_or_metadata_budget_excess(bad):
    with pytest.raises(RuntimeError):
        targets.extended_dense_lead_minutes(bad)


def test_extended_grid_requires_current_state_and_first_loss_weight():
    with pytest.raises(RuntimeError, match='STATE_EXTENDED_REQUIRES_CURRENT_INPUT'):
        owner.dynamic_mtf_state_research_contract(extended_forecast_minutes=1441)
    with pytest.raises(RuntimeError, match='EXTENDED_REQUIRES_CURRENT_STATE'):
        recipes.native_direction_research_recipe(batch_size=16, max_updates=2,
                                                dynamic_state=True, extended_forecast_minutes=1441)


def test_extended_recipe_view_and_legacy_grid_are_separately_bound(bound_case):
    args = deepcopy(bound_case['arguments'])
    args['source_bindings']['dynamic_state_target_source'] = {
        'path': '/synthetic/entry_dynamic_mtf_state_targets_v1.py', 'sha256': 'b' * 64}
    args.update(full_feature_profile=True, dynamic_state_supervision=True, current_state_first_loss_weight=1.0)
    v2 = views.bind_native_direction_feature_view(**args)
    args['extended_forecast_minutes'] = 1441
    v3 = views.bind_native_direction_feature_view(**args)
    recipe = recipes.native_direction_research_recipe(batch_size=16, max_updates=2, dynamic_state=True,
                                                       current_state_first_loss_weight=1.0, extended_forecast_minutes=1441,
                                                       extended_storage_reserve_bytes=5 * 1024**3)
    assert recipes.require_native_direction_research_recipe(recipe) == recipe
    assert v3['research_objective']['contract_sha256'] == recipe['research_objective_sha256']
    assert v3['research_objective']['lead_minutes'] == list(range(1, 1442))
    assert v3['physical_field_names_sha256'] == v2['physical_field_names_sha256']
    assert v3['entry_execution_profile'] == v2['entry_execution_profile']
    assert v3['contract_sha256'] != v2['contract_sha256']


def _label_fixture():
    leads = np.arange(1, 1442, dtype=np.int64)
    state = np.broadcast_to((np.arange(len(leads)) % 3).astype(np.int8), (1, 5, len(leads))).copy()
    observed = np.ones_like(state, dtype=bool)
    state[:, :, -2] = -1
    observed[:, :, -2] = False
    prior = report.StatePriorAccumulator(lead_minutes=leads)
    prior.update(state, observed)
    return leads, state, observed, prior.finalize()


def test_extended_synthetic_report_keeps_last_minute_and_does_not_bridge_unknown():
    leads, state, observed, prior = _label_fixture()
    scorer = report.StateScoreAccumulator(prior, lead_minutes=leads)
    probabilities = np.full((*state.shape, 3), 1.0 / 3.0, dtype=np.float32)
    transition = np.full((*state.shape[:2], len(leads) - 1, 3, 3), 1.0 / 3.0, dtype=np.float32)
    scorer.update(probabilities, state=state, eligible=observed,
                  current_state=np.zeros((1, 5), np.int8), current_observed=np.ones((1, 5), bool),
                  sequence_nll=np.zeros((1, 5), np.float32), transition_probabilities=transition)
    result = scorer.finalize()
    assert result['lead_minutes'][-1] == 1441
    assert result['per_lead']['observed'][0][-2:] == [0, 1]
    assert result['adjacent_pair_forecast']['observed'] == [1438] * 5
    json.dumps(result, allow_nan=False)


def test_extended_loss_observes_last_minute_and_excludes_hole():
    leads, state, observed, _ = _label_fixture()
    state[observed] = 1
    minute = 60 * 10**9
    issue = torch.tensor([100 * minute], dtype=torch.int64)
    validity = issue[:, None] + torch.arange(1, 1442, dtype=torch.int64)[None, :] * minute
    result = owner.native_dynamic_mtf_state_loss(torch.zeros((1, 5, 3)), torch.zeros((1, 5, 1440, 3, 3)),
        state=torch.from_numpy(state), observed=torch.from_numpy(observed), issue_ns=issue, validity_ns=validity,
        maturity_ns=validity[:, None].expand(1, 5, 1441).clone(), cutoff_ns=1542 * minute,
        first_state_loss_weight=2.0, extended_forecast_minutes=1441)
    assert result['counts']['eligible_cells'] == 1440 * 5
    assert result['eligible_mask'][0, 0, -2:].tolist() == [False, True]
    assert result['first_state_eligible_cells'] == 5
    torch.testing.assert_close(result['loss'], torch.tensor(3 * math.log(3)), atol=3e-4, rtol=3e-4)


def test_extended_train_audit_must_match_entire_grid(monkeypatch):
    leads, _, _, prior = _label_fixture()
    fold = deepcopy(private.BPS_FOLD)
    prior['parents'] = fold['prefix_rows']
    dummy = {'path': '/synthetic/artifact.json', 'sha256': 'a' * 64}
    sources = {'m1': dummy, 'm5_atr': dummy}
    objective = 'b' * 64
    manifest = {'metadata': {'parents': fold['oof_end_row'], 'definition': {'contract_sha256': 'c' * 64}}}
    audit = {'schema_version': 'gx1_dynamic_mtf_state_label_audit_v1', 'support_status': 'PASS',
        'blockers': [], 'target_manifest': dummy, 'target_sources': sources, 'fold': fold,
        'target_contract_sha256': 'c' * 64, 'objective_sha256': objective, 'lead_minutes': leads.tolist(),
        'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False, 'later_metrics_inspected': False,
        'prior': prior}
    recipe = {'fold': fold, 'target_sources': sources, 'target_manifest': dummy, 'label_audit': dummy,
              'numerical_recipe': {'research_objective_sha256': objective, 'extended_forecast_minutes': 1441}}
    monkeypatch.setattr(campaign.campaign, 'require_binding', lambda ref, **kwargs: ref)
    monkeypatch.setattr(campaign.campaign, 'read_bound_json', lambda *args: audit)
    monkeypatch.setattr(targets, 'read_dynamic_mtf_state_series_metadata', lambda *args, **kwargs: manifest)
    assert campaign.require_dynamic_state_target_binding(recipe) == (manifest, audit)
    audit['lead_minutes'] = list(range(1, 1441))
    with pytest.raises(RuntimeError, match='STATE_TRAIN_LABEL_AUDIT'):
        campaign.require_dynamic_state_target_binding(recipe)


def test_seven_day_loss_uses_every_minute_without_model():
    horizon = 10080
    minute = 60 * 10**9
    issue = torch.tensor([100 * minute], dtype=torch.int64)
    validity = issue[:, None] + torch.arange(1, horizon + 1, dtype=torch.int64)[None, :] * minute
    state = torch.full((1, 5, horizon), 1, dtype=torch.int8)
    result = owner.native_dynamic_mtf_state_loss(torch.zeros((1, 5, 3)),
        torch.zeros((1, 5, horizon - 1, 3, 3)), state=state, observed=torch.ones_like(state, dtype=torch.bool),
        issue_ns=issue, validity_ns=validity, maturity_ns=validity[:, None].expand_as(state).clone(),
        cutoff_ns=(100 + horizon + 1) * minute, first_state_loss_weight=2.0, extended_forecast_minutes=horizon)
    assert result['counts']['eligible_cells'] == 5 * horizon
    assert result['eligible_mask'][0, 0, -1].item()
    torch.testing.assert_close(result['loss'], torch.tensor(3 * math.log(3)), atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize('query_count', [1440, 1441, 10080])
def test_private_initial_shape_follows_declared_grid_without_model(query_count):
    from gx1.scripts.run_entry_native_direction_dense_qualification_v1 import require_dynamic_state_initial_measurement
    measured = {'initial_parity_passed': True,
                'state_artifact': {'arrays': {'probabilities': {'shape': [256, 5, query_count, 3]}}}}
    require_dynamic_state_initial_measurement(measured, 256, query_count=query_count)
    with pytest.raises(RuntimeError, match='STATE_FRESH_INITIAL'):
        require_dynamic_state_initial_measurement(measured, 256, query_count=query_count + 1)
    with pytest.raises(RuntimeError, match='STATE_FRESH_INITIAL'):
        require_dynamic_state_initial_measurement(measured, 255, query_count=query_count)
    measured['initial_parity_passed'] = False
    with pytest.raises(RuntimeError, match='STATE_FRESH_INITIAL'):
        require_dynamic_state_initial_measurement(measured, 256, query_count=query_count)
