"""Synthetic signed-MFE proofs for184; no genuine data or native model construction."""
from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import nn

from gx1.contracts import entry_dynamic_m5_loss_scale_v1 as scales
from gx1.contracts import entry_model_native_training_objective_v1 as objective
from gx1.contracts import entry_observed_market_v1 as observed
from gx1.contracts.local_random_access_campaign_v2 import canonical_sha256
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

MFE, CLOSED_MFE = "dynamic_mfe_bps", "dynamic_closed_mfe_bps"
TERMINAL, MAE, CLOSED_MAE = "dynamic_terminal_net_markout_bps", "dynamic_pre_mfe_mae_upper_bps", "dynamic_closed_pre_mfe_mae_upper_bps"
PHASE = objective.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
EARLY, LATER = "early_1_15", "later_16_480"
FROZEN_111_CONTRACT_SHA = "50a1c635cc47d4619cdee035027634abe93689b724a97b24832d12a8895388e4"


def _mfe_case(enabled=(EARLY, LATER), *, endpoints=(MFE, CLOSED_MFE)):
    values = {}
    # Early inclusive SSE8/N3 plus closed9/N1; later inclusive4/N2 plus closed9/N4.
    # Three genuinely observed zero residuals remain in the counts, without FLAT.
    table = {
        MFE: [(0, 0, 0, -2., 2., 2.), (0, 4, 1, 1., -5., 3.), (1, 14, 0, 0., 0., 4.),
              (0, 15, 0, -3., 3., 3.), (1, 479, 1, 0., 0., 1.)],
        CLOSED_MFE: [(1, 0, 1, -1., 2., 1.), (0, 15, 0, 2., 4., 2.),
                     (1, 119, 1, -2., -8., 3.), (1, 479, 0, 0., 0., 5.), (0, 239, 1, 3., 7., 2.)],
    }
    for name, cells in table.items():
        p = torch.zeros((2, 480, 2), dtype=torch.float32)
        y, valid, rms = torch.zeros_like(p), torch.zeros_like(p, dtype=torch.bool), torch.ones((480, 2))
        for row, query, side, target, prediction, unit in cells:
            p[row, query, side], rms[query, side] = prediction, unit
            if name in endpoints and (EARLY if query < 15 else LATER) in enabled:
                y[row, query, side], valid[row, query, side] = target, True
        p[0, 20, 1] = 123.  # Finite masked prediction is not supervision.
        values[name] = (p.requires_grad_(), y, valid, rms)
    return values


def _loss(values):
    p, y, valid, rms = values[MFE]
    cp, cy, cv, crms = values[CLOSED_MFE]
    return scales.dynamic_m5_mfe_horizon_objective(p, y, valid, cp, cy, cv, rms=rms, closed_rms=crms)


def _expected_gradients(values):
    p, cp = values[MFE][0], values[CLOSED_MFE][0]
    g, cg = torch.zeros_like(p), torch.zeros_like(cp)
    g[0, 0, 0], g[0, 4, 1] = 1/4, -1/6
    cg[1, 0, 1] = 3/4
    g[0, 15, 0] = 1/9
    cg[0, 15, 0], cg[1, 119, 1], cg[0, 239, 1] = 1/12, -1/9, 1/6
    return {MFE: g, CLOSED_MFE: cg}


def test_signed_asymmetric_endpoints_pool_true_group_counts_and_backward_without_mutation():
    values = _mfe_case()
    before = {name: tuple(value.detach().clone() for value in spec) for name, spec in values.items()}
    rng = torch.get_rng_state().clone()
    loss, counts = _loss(values)
    assert counts == {EARLY: 4, LATER: 6}
    assert loss.dtype == torch.float32 and float(loss.detach()) == pytest.approx(77/24)
    pooled = [scales.dynamic_m5_scaled_square_sum(*spec[:3], surface=name, rms=spec[3])
              for name, spec in values.items()]
    assert sum(n for _, n in pooled) == 10
    assert float(sum(sse for sse, _ in pooled).detach()) == 30.
    assert float(loss.detach()) != pytest.approx(30/10)
    # Equal endpoint means would give early(8/3+9)/2, which is not17/4.
    assert 17/4 != pytest.approx((8/3+9)/2)
    loss.backward()
    for name, wanted in _expected_gradients(values).items():
        p, y, valid, rms = values[name]
        torch.testing.assert_close(p.grad, wanted)
        assert torch.count_nonzero(p.grad[~valid]) == 0
        assert y.grad is None and rms.grad is None
        assert all(torch.equal(value.detach(), old) for value, old in zip(values[name], before[name]))
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("enabled,expected,counts", [
    ((EARLY,), 17/4, {EARLY: 4, LATER: 0}),
    ((LATER,), 13/6, {EARLY: 0, LATER: 6}),
    ((), None, {EARLY: 0, LATER: 0}),
])
def test_one_group_has_weight_one_and_no_group_omits_MFE(enabled, expected, counts):
    values = _mfe_case(enabled)
    loss, actual = _loss(values)
    assert actual == counts
    if expected is None:
        assert loss is None and all(spec[0].grad is None for spec in values.values())
    else:
        assert float(loss.detach()) == pytest.approx(expected)
        loss.backward()
        for name, wanted in _expected_gradients(values).items():
            selected = wanted.clone() * 2
            if EARLY in enabled: selected[:, 15:] = 0
            else: selected[:, :15] = 0
            torch.testing.assert_close(values[name][0].grad, selected)


@pytest.mark.parametrize("endpoint,expected,counts", [
    (MFE, 7/3, {EARLY: 3, LATER: 2}),
    (CLOSED_MFE, 45/8, {EARLY: 1, LATER: 4}),
])
def test_an_unobserved_endpoint_does_not_receive_equal_weight(endpoint, expected, counts):
    values = _mfe_case(endpoints=(endpoint,))
    loss, actual = _loss(values)
    assert actual == counts and float(loss.detach()) == pytest.approx(expected)
    loss.backward()
    absent = CLOSED_MFE if endpoint == MFE else MFE
    assert torch.count_nonzero(values[absent][0].grad) == 0
    assert torch.count_nonzero(values[endpoint][0].grad) > 0


@pytest.mark.parametrize("fault", ["masked_later_target", "nonfinite_later_target", "zero_later_RMS", "wrong_endpoint_shape"])
def test_full_original_storage_is_validated_before_group_masks(fault):
    values = _mfe_case((EARLY,))
    p, y, valid, rms = values[CLOSED_MFE]
    if fault == "masked_later_target": y[0, 400, 0] = 1.
    elif fault == "nonfinite_later_target": y[0, 400, 0] = float("nan")
    elif fault == "zero_later_RMS": rms[400, 0] = 0.
    else: values[CLOSED_MFE] = (p.detach()[:, :, :1].clone().requires_grad_(), y[:, :, :1], valid[:, :, :1], rms)
    with pytest.raises(RuntimeError, match="DYNAMIC_M5_LOSS_SCALE_RUNTIME_"):
        _loss(values)
    assert all(spec[0].grad is None for spec in values.values())


def _all_surface_case(*, active_mfe):
    from tests.test_entry_dynamic_m5_terminal_horizon_objective_v1 import _all_surface_case as terminal_fixture
    out, batch = terminal_fixture(active_terminal=True)
    values = _mfe_case((EARLY, LATER) if active_mfe else ())
    for name, target_key, valid_key, mae, mae_target in (
        (MFE, observed.DYNAMIC_M5_MFE_TARGET_KEY, observed.DYNAMIC_M5_PATH_VALID_KEY,
         MAE, observed.DYNAMIC_M5_MAE_TARGET_KEY),
        (CLOSED_MFE, observed.DYNAMIC_M5_CLOSED_MFE_TARGET_KEY, observed.DYNAMIC_M5_CLOSED_PATH_VALID_KEY,
         CLOSED_MAE, observed.DYNAMIC_M5_CLOSED_MAE_TARGET_KEY)):
        p, y, valid, rms = values[name]
        out[name], batch[target_key], batch[valid_key] = p, y, valid
        batch[scales.DYNAMIC_M5_LOSS_SCALE_BATCH_KEY]["rms"][name] = rms
        # Path masks are shared with MAE; give MAE genuine zero targets at those cells.
        out[mae] = torch.full_like(p, 2.).requires_grad_()
        batch[mae_target] = torch.zeros_like(y)
    return out, batch, values


@pytest.mark.parametrize("active", [True, False])
def test_native_adapter_changes_only_MFE_and_preserves_terminal170_MAE_forecast_and_diagnostics(active):
    out, batch, _ = _all_surface_case(active_mfe=active)
    device = torch.device("cpu")
    raw, n = trainer._dynamic_m5_task_loss_surfaces(out, batch, device)
    pooled, unit_n = trainer._dynamic_m5_task_loss_surfaces(out, batch, device, loss_units="train_rms")
    snapshots = {name: loss.detach().clone() for name, loss in pooled.items()}
    optimized, terminal_counts = trainer._dynamic_m5_optimizer_task_losses(out, batch, device, pooled_task_losses=pooled)
    assert n == unit_n and terminal_counts == {EARLY: 4, LATER: 6}
    assert float(optimized[TERMINAL].detach()) == pytest.approx(77/24)
    assert float(pooled[TERMINAL].detach()) == 3.
    for name in (MAE, "forecast_return_bps"):
        if name in pooled: assert optimized[name] is pooled[name]
    assert (MFE in optimized) is active
    if active:
        assert float(optimized[MFE].detach()) == pytest.approx(77/24)
        assert float(pooled[MFE].detach()) == 3.
        assert float(raw[MFE].detach()) == pytest.approx(15.3)
    assert set(optimized) == set(pooled)
    assert all(torch.equal(pooled[name].detach(), before) for name, before in snapshots.items())
    after_raw, after_n = trainer._dynamic_m5_task_loss_surfaces(out, batch, device)
    assert n == after_n and all(torch.equal(raw[name], after_raw[name]) for name in raw)


@pytest.mark.parametrize("active", [True, False])
def test_MFE_task_precision_is_applied_once_and_absent_task_has_no_logvariance_gradient(active):
    values = _mfe_case((EARLY, LATER) if active else ())
    mfe, _ = _loss(values)
    model = nn.Module()
    model.entry_training_phase = PHASE
    model.task_log_variances = nn.ParameterDict({name: nn.Parameter(torch.zeros(()))
        for name in trainer.joint_task_names_for_phase(PHASE)})
    with torch.no_grad(): model.task_log_variances[MFE].fill_(math.log(2))
    losses = {"forecast_return_bps": torch.tensor(10., requires_grad=True)}
    if mfe is not None: losses[MFE] = mfe
    joint, _ = trainer._joint_task_loss(model, losses, loss_units="train_rms")
    assert float(joint.detach()) == pytest.approx(10 + (77/48+math.log(2) if active else 0))
    joint.backward()
    if active:
        assert float(model.task_log_variances[MFE].grad) == pytest.approx(1-77/48)
        for name, wanted in _expected_gradients(values).items():
            torch.testing.assert_close(values[name][0].grad, wanted/2)
    else:
        assert model.task_log_variances[MFE].grad is None
        assert all(spec[0].grad is None for spec in values.values())


def test_MFE_optimizer_evidence_is_separate_active_row_weighted_and_scope_limited():
    evidence = trainer._new_dynamic_m5_optimizer_loss_evidence()
    assert trainer._finalize_dynamic_m5_optimizer_loss_evidence(evidence)["dynamic_m5_mfe_optimizer_objective"]["batch_row_weighted_mfe_optimizer_objective_mean"] is None
    for rows, mfe in ((2, 2.), (1, 10.), (3, None)):
        losses = {TERMINAL: torch.tensor(0.)}
        if mfe is not None: losses[MFE] = torch.tensor(mfe)
        trainer._accumulate_dynamic_m5_optimizer_loss_evidence(evidence, losses, {EARLY: 1, LATER: 0}, batch_rows=rows)
    result = trainer._finalize_dynamic_m5_optimizer_loss_evidence(evidence)
    value = result["dynamic_m5_mfe_optimizer_objective"]
    assert value["objective_contract"] == scales.dynamic_m5_mfe_horizon_objective_contract()
    assert value["processed_batch_rows"] == 6 and value["active_mfe_batch_rows"] == 3
    assert value["active_mfe_batches"] == 2 and value["omitted_mfe_batches"] == 1
    assert value["batch_row_weighted_mfe_optimizer_objective_mean"] == pytest.approx(14/3)
    assert value["aggregation_scope"] == "current_train_epoch_call_or_single_curve_validation_pass_no_cross_invocation_history"
    terminal = result["dynamic_m5_terminal_optimizer_objective"]
    assert "mfe_optimizer_evidence" not in terminal
    assert terminal["batch_row_weighted_terminal_optimizer_objective_mean"] == 0.


@pytest.mark.parametrize("route", ["train", "curve"])
def test_native_train_and_curve_use_same_MFE_optimizer_value_with_pooled_diagnostics(monkeypatch, tmp_path, route):
    from tests.test_entry_model_native_training_objective import _dynamic_gate_execution_case
    case = _dynamic_gate_execution_case(monkeypatch, tmp_path, compact=False)
    def optimizer_plumbing(out, batch, device, *, pooled_task_losses):
        losses = dict(pooled_task_losses)
        losses[MFE] = 3 * losses[MFE]
        return losses, {EARLY: 1, LATER: 1}
    monkeypatch.setattr(case.trainer, "_dynamic_m5_optimizer_task_losses", optimizer_plumbing)
    if route == "curve":
        result = case.trainer._validate_dynamic_m5_curves(case.model.eval(), case.loader, torch.device("cpu"))
        loss, stats = result[0], result[-1]
    else:
        optimizer = torch.optim.SGD(case.model.parameters(), lr=0.)
        loss, stats, complete = case.trainer.train_epoch(case.model, case.target, case.loader, optimizer,
            torch.device("cpu"), 1, {}, {})
        assert complete
    assert loss == pytest.approx(3.75)
    assert stats[f"joint_task_raw_loss_mean_{MFE}"] == pytest.approx(.625)
    assert stats[f"joint_task_unit_loss_mean_{MFE}"] == pytest.approx(.625)
    value = stats["dynamic_m5_mfe_optimizer_objective"]
    assert value["batch_row_weighted_mfe_optimizer_objective_mean"] == pytest.approx(1.875)
    assert value["processed_batch_rows"] == value["active_mfe_batch_rows"] == 2
    assert value["active_mfe_batches"] == 1 and value["omitted_mfe_batches"] == 0
    assert stats["dynamic_m5_terminal_optimizer_objective"]["batch_row_weighted_terminal_optimizer_objective_mean"] == pytest.approx(.625)
    assert all(parameter.grad is None for parameter in case.model.exit_path_proj.parameters())


@pytest.mark.parametrize("fault", ["old_schema", "old_MFE_label", "pooled_reduction", "endpoint", "group_boundary"])
def test_MFE_objective_contract_preserves111_terminal170_and_rejects_resealed_drift(fault):
    original_units = scales.dynamic_m5_loss_scale_contract()
    original_terminal = scales.dynamic_m5_terminal_horizon_objective_contract()
    assert original_units["contract_sha256"] == FROZEN_111_CONTRACT_SHA
    value = objective.training_objective_contract_metadata(entry_training_phase=PHASE)
    contract = value["dynamic_mfe_optimizer_objective"]
    assert value["schema_version"] == "entry_model_native_training_objective_v16"
    assert contract == scales.dynamic_m5_mfe_horizon_objective_contract()
    assert contract["loss_unit_contract_sha256"] == FROZEN_111_CONTRACT_SHA
    assert contract["reduction"] == "arithmetic_mean_of_nonempty_group_means"
    assert contract["endpoint_surfaces"] == [MFE, CLOSED_MFE]
    assert value["dynamic_terminal_optimizer_objective"] == original_terminal
    assert value["dynamic_path_loss_reduction"][MAE] == "all_observed_cells_of_both_endpoint_kinds"
    forged = copy.deepcopy(value)
    if fault == "old_schema": forged["schema_version"] = "entry_model_native_training_objective_v15"
    elif fault == "old_MFE_label": forged["dynamic_component_losses"][MFE] = "independently_masked_TRAIN_target_RMS_scaled_squared_error"
    else:
        bad = forged["dynamic_mfe_optimizer_objective"]
        if fault == "pooled_reduction": bad["reduction"] = "pooled_SSE_over_N"
        elif fault == "endpoint": bad["endpoint_surfaces"] = [MFE]
        else: bad["groups"][0]["query_minutes"][1] = 16
        bad.pop("contract_sha256")
        bad["contract_sha256"] = canonical_sha256(bad)
    with pytest.raises(RuntimeError, match="TRAINING_OBJECTIVE_CONTRACT_INVALID"):
        objective.require_training_objective_contract(forged, context="MFE184_TEST")
    assert scales.dynamic_m5_loss_scale_contract() == original_units
    assert scales.dynamic_m5_terminal_horizon_objective_contract() == original_terminal
