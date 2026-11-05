"""Pure synthetic tensors for the registered five native-lead marginal losses."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as targets
from gx1.contracts import entry_native_direction_research_v1 as research
from gx1.scripts.run_entry_native_direction_prefix_fit_window_v1 import _state_loss
from tests.test_entry_native_direction_feature_view_v1 import bound_case


LEADS = (5, 15, 60, 240, 1440)
HISTORICAL = dict(first_state_loss_weight=.03, first_flat_activity_weight=.03)
OLD_OBJECTIVE = '035c842592dc98fab8604e705e4209e7a62510e78e1c66d4dc31916949533dad'


def _case(batch):
    # Second issue, when present, has its +240 cell exactly at the exclusive cutoff.
    minute = targets.MINUTE_NS
    issues = torch.tensor([10, 1211][:batch], dtype=torch.int64) * minute
    validity = issues[:, None] + torch.arange(1, 1441, dtype=torch.int64) * minute
    shape = (batch, len(LEADS), 1440)
    random = np.random.default_rng(1971)
    logits = {name: torch.tensor(random.normal(size=dims), dtype=torch.float32, requires_grad=True)
              for name, dims in [('initial_state_logits', (batch, 5, 3)),
                                 ('state_transition_logits', (batch, 5, 1439, 3, 3))]}
    labels = {'state': torch.full(shape, targets.UNKNOWN, dtype=torch.int8),
              'observed': torch.zeros(shape, dtype=torch.bool), 'issue_ns': issues, 'validity_ns': validity,
              'maturity_ns': validity[:, None, :].expand(shape),
              'first_flat_activity_observed': torch.zeros((batch, 5), dtype=torch.bool),
              'first_flat_activity_is_flat': torch.zeros((batch, 5), dtype=torch.bool)}
    return logits, labels, 1451 * minute


def _observe(labels, row, tf, lead, state):
    labels['state'][row, tf, lead - 1] = state
    labels['observed'][row, tf, lead - 1] = True


def _softmax(values):
    mass = np.exp(values - values.max(axis=-1, keepdims=True))
    return mass / mass.sum(axis=-1, keepdims=True)


def _oracle(logits, cells):
    # Sequential float64 probability propagation, deliberately separate from the parallel log-space owner.
    first = _softmax(logits['initial_state_logits'].detach().numpy().astype(np.float64))
    transitions = _softmax(logits['state_transition_logits'].detach().numpy().astype(np.float64))
    costs = []
    for row, tf, label in cells:
        p = first[row, tf]
        for transition in transitions[row, tf, :LEADS[tf] - 1]:
            p = p @ transition
        costs.append(-math.log(p[label]))
    return sum(costs) / len(costs)


def test_native_endpoints_use_unconditional_marginals_and_exact_global_eligible_denominator():
    logits, labels, cutoff = _case(2)
    obs = [(0, tf, state) for tf, state in enumerate((targets.DOWN, targets.FLAT, targets.UP,
                                                        targets.DOWN, targets.FLAT))]
    obs.extend([(1, 0, targets.UP), (1, 2, targets.DOWN)])
    for row, tf, state in obs:
        _observe(labels, row, tf, LEADS[tf], state)
    # A masked M15, an observed but exactly-on-cutoff H4 and an observed immature D1 all stay outside.
    _observe(labels, 1, 3, LEADS[3], targets.UP)
    _observe(labels, 1, 4, LEADS[4], targets.FLAT)
    assert labels['maturity_ns'][1, 3, LEADS[3] - 1].item() == cutoff
    # Non-endpoint observed neighbours still influence the sequence likelihood, never the endpoint forecasts.
    _observe(labels, 0, 0, LEADS[0] - 1, targets.UP)
    _observe(labels, 1, 1, 1, targets.FLAT)
    labels['first_flat_activity_observed'][1, 1] = labels['first_flat_activity_is_flat'][1, 1] = True

    historical = research.dynamic_mtf_state_research_contract(current_state_input=True, **HISTORICAL)
    old_model = SimpleNamespace(**HISTORICAL, extended_forecast_minutes=None, research_objective=historical)
    old = _state_loss(old_model, logits, labels, cutoff=cutoff)
    objective = research.dynamic_mtf_state_research_contract(current_state_input=True, **HISTORICAL,
                                                              native_endpoint_loss=True)
    assert objective['native_endpoint_lead_minutes'] == list(LEADS)
    assert 'native_endpoint_lead_minutes' not in historical
    model = SimpleNamespace(**HISTORICAL, extended_forecast_minutes=None, native_endpoint_loss=True,
                            research_objective=objective)
    new = _state_loss(model, logits, labels, cutoff=cutoff)
    assert old['research_objective_sha256'] == OLD_OBJECTIVE
    assert objective['contract_sha256'] == new['research_objective_sha256'] != OLD_OBJECTIVE
    assert new['native_endpoint_eligible_cells'] == len(obs)
    assert new['native_endpoint_eligible_per_tf'] == [2, 1, 2, 1, 1]
    component = new['native_endpoint_loss']
    assert component.item() == pytest.approx(_oracle(logits, obs), abs=2e-5)
    torch.testing.assert_close(new['loss'], old['loss'] + component, atol=2e-6, rtol=2e-6)
    grad_first, grad_trans = torch.autograd.grad(component, tuple(logits.values()))
    assert torch.isfinite(grad_first).all() and torch.isfinite(grad_trans).all()
    assert grad_first[0, 0].abs().sum() > 0
    for tf, lead in enumerate(LEADS):
        assert grad_trans[0, tf, lead - 2].abs().sum() > 0
        assert not grad_trans[:, tf, lead - 1:].count_nonzero()
    assert not grad_trans[1, [1, 3, 4]].count_nonzero()


def test_native_log_marginal_survives_genuine_float32_probability_underflow():
    logits, labels, cutoff = _case(1)
    with torch.no_grad():
        for tensor in logits.values():
            tensor.zero_()
            tensor[..., targets.DOWN] = -200.
    _observe(labels, 0, 0, LEADS[0], targets.DOWN)
    result = research.native_dynamic_mtf_state_loss(**logits, **labels, cutoff_ns=cutoff,
                                                   **HISTORICAL, native_endpoint_loss=True)
    # Every row of every transition gives the same class law; exp(-200) underflows in FP32.
    aux = result['native_endpoint_loss']
    assert aux.item() == pytest.approx(200. + math.log(2.), abs=3e-5)
    grad = torch.autograd.grad(aux, logits['state_transition_logits'])[0]
    assert torch.isfinite(grad).all()
    # With identical rows, a shift of every final transition row has the softmax-CE derivative.
    final_by_class = grad[0, 0, LEADS[0] - 2].sum(dim=0)
    # logsumexp backward subtracts float32 log masses near -200; allow one ULP at that scale.
    log_mass_ulp = float(np.spacing(np.float32(200. + math.log(2.))))
    torch.testing.assert_close(final_by_class, torch.tensor([-1., .5, .5]), atol=log_mass_ulp, rtol=0)


def test_no_eligible_native_endpoint_preserves_the_unmodified_loss_or_empty_batch():
    logits, labels, cutoff = _case(1)
    for state in (None, targets.FLAT):
        if state is not None:
            _observe(labels, 0, 0, 1, state)
        old = research.native_dynamic_mtf_state_loss(**logits, **labels, cutoff_ns=cutoff, **HISTORICAL)
        new = research.native_dynamic_mtf_state_loss(**logits, **labels, cutoff_ns=cutoff,
                                                   **HISTORICAL, native_endpoint_loss=True)
        assert new['native_endpoint_eligible_cells'] == 0 and new['native_endpoint_loss'] is None
        assert new['native_endpoint_eligible_per_tf'] == [0] * 5
        assert old['update_allowed'] is new['update_allowed'] is (state is not None)
        if state is None:
            assert old['loss'] is new['loss'] is None
        else:
            torch.testing.assert_close(new['loss'], old['loss'], atol=0, rtol=0)


@pytest.mark.parametrize('kwargs', [
    {}, {'current_state_input': True, 'first_state_loss_weight': .03},
    {'current_state_input': True, **HISTORICAL, 'extended_forecast_minutes': 10080},
])
def test_endpoint_objective_cannot_be_attached_to_a_different_baseline(kwargs):
    with pytest.raises(RuntimeError):
        research.dynamic_mtf_state_research_contract(**kwargs, native_endpoint_loss=True)


def test_numerical_recipe_binds_only_the_opted_in_loss_and_rejects_a_forged_mode():
    from gx1.contracts.entry_model_native_train_recipe_v1 import (
        native_direction_research_recipe, require_native_direction_research_recipe,
    )

    controls = dict(batch_size=16, max_updates=2976, dynamic_state=True,
                    current_state_first_loss_weight=.03, first_flat_activity_weight=.03)
    original = native_direction_research_recipe(**controls)
    endpoint = native_direction_research_recipe(**controls, native_endpoint_loss=True)
    assert original["research_objective_sha256"] == OLD_OBJECTIVE
    assert original["contract_sha256"] == "f8c4f4e4585db44c031452139f9c218b6dd12da360f112882f8a6810d9406757"
    assert endpoint["research_objective_sha256"] == research.dynamic_mtf_state_research_contract(
        current_state_input=True, **HISTORICAL, native_endpoint_loss=True)["contract_sha256"]
    assert require_native_direction_research_recipe(endpoint) == endpoint
    assert endpoint["native_launch_authorized"] is False
    with pytest.raises(RuntimeError):
        require_native_direction_research_recipe({**original, "native_endpoint_loss": True})
    with pytest.raises(RuntimeError):
        native_direction_research_recipe(**controls, native_endpoint_loss=1)


def test_confirmed_age_objective_is_opt_in_and_retains_v5_identity():
    historical = research.dynamic_mtf_state_research_contract(
        current_state_input=True, **HISTORICAL, native_endpoint_loss=True)
    candidate = research.dynamic_mtf_state_research_contract(
        current_state_input=True, **HISTORICAL, native_endpoint_loss=True,
        confirmed_run_age_input=True)
    assert historical["schema_version"] == "gx1_native_dynamic_mtf_endpoint_research_v5"
    assert candidate["schema_version"] == "gx1_native_dynamic_mtf_confirmed_age_research_v6"
    assert candidate["contract_sha256"] != historical["contract_sha256"]
    assert candidate["confirmed_run_age_future_labels_as_inputs"] is False
    assert research.dynamic_mtf_state_research_contract(
        current_state_input=True, **HISTORICAL, native_endpoint_loss=True) == historical
    with pytest.raises(RuntimeError, match="STATE_CONFIRMED_AGE_MODE"):
        research.dynamic_mtf_state_research_contract(
            current_state_input=True, **HISTORICAL, confirmed_run_age_input=True)


def test_feature_view_seals_the_new_loss_without_changing_the_input_profile(bound_case):
    from gx1.contracts import entry_native_direction_feature_view_v1 as views
    from tests.test_entry_native_dynamic_current_routing_v1 import _arguments

    args = dict(_arguments(bound_case), current_state_first_loss_weight=.03, first_flat_activity_weight=.03)
    original = views.bind_native_direction_feature_view(**args)
    endpoint = views.bind_native_direction_feature_view(**args, native_endpoint_loss=True)
    assert original["research_objective"]["contract_sha256"] == OLD_OBJECTIVE
    assert "native_endpoint_loss" not in original and endpoint["native_endpoint_loss"] is True
    assert endpoint["research_objective"] == research.dynamic_mtf_state_research_contract(
        current_state_input=True, **HISTORICAL, native_endpoint_loss=True)
    for key in ("physical_field_names_sha256", "prefix_normalization", "entry_execution_profile"):
        assert original[key] == endpoint[key]
    with pytest.raises(RuntimeError):
        views.bind_native_direction_feature_view(**args, expected=endpoint)
