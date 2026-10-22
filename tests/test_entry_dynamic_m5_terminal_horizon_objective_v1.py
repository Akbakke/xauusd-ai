"""Synthetic F32 proofs for the selected terminal optimizer reduction.

No genuine data, Transformer construction, checkpoint payload or serving authority.
The older gate fixture below is used only to test native call plumbing; the real
terminal reduction and adapter have independent closed-form tests in this file.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest
import torch
from torch import nn

from gx1.contracts import entry_dynamic_m5_loss_scale_v1 as scales
from gx1.contracts import entry_model_native_training_objective_v1 as objective
from gx1.contracts import entry_observed_market_v1 as observed
from gx1.contracts.local_random_access_campaign_v2 import canonical_sha256
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

TERMINAL, MFE, MAE, CLOSED_MFE, CLOSED_MAE, FORECAST = scales.SURFACE_NAMES
PHASE = objective.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
EARLY, LATER = "early_1_15", "later_16_480"
OTHER_TASKS = (MFE, MAE, "forecast_return_bps")
FROZEN_111_CONTRACT_SHA = "50a1c635cc47d4619cdee035027634abe93689b724a97b24832d12a8895388e4"


def _terminal_case(enabled=(EARLY, LATER)):
    prediction = torch.zeros((2, 480, 3), dtype=torch.float32)
    target = torch.zeros_like(prediction)
    valid = torch.zeros_like(prediction, dtype=torch.bool)
    rms = torch.ones((480, 2), dtype=torch.float32)
    # (row, zero-based query, side, target, prediction, frozen unit).
    # Early SSE/N=17/4; later=13/6; FLAT adds1 and2 true cells respectively.
    cells = [
        (0, 0, 0, 2., 6., 2.), (0, 0, 1, -4., -10., 3.),
        (1, 14, 0, 0., 3., 1.),
        (0, 15, 0, 1., 5., 2.), (0, 15, 1, -1., -7., 3.),
        (1, 479, 0, 2., 6., 4.), (1, 479, 1, -2., -8., 3.),
    ]
    for row, query, side, y, p, unit in cells:
        rms[query, side] = unit
        prediction[row, query, side] = p
        group = EARLY if query < 15 else LATER
        if group in enabled:
            target[row, query, side] = y
            valid[row, query, side] = True
    valid[..., 2] = valid[..., 0] & valid[..., 1]
    prediction[0, 1, 0] = 123.  # Finite masked prediction is not supervision.
    return prediction.requires_grad_(), target, valid, rms


def _expected_balanced_gradient(prediction):
    expected = torch.zeros_like(prediction)
    expected[0, 0, 0], expected[0, 0, 1] = 1/4, -1/6
    expected[1, 14, 0] = 3/4
    expected[0, 15, 0], expected[0, 15, 1] = 1/6, -1/9
    expected[1, 479, 0], expected[1, 479, 1] = 1/24, -1/9
    return expected


def _task_weight_model():
    model = nn.Module()
    model.entry_training_phase = PHASE
    model.task_log_variances = nn.ParameterDict({
        name: nn.Parameter(torch.zeros(()))
        for name in trainer.joint_task_names_for_phase(PHASE)
    })
    return model


def test_unequal_groups_closed_form_autograd_flat_count_and_rng():
    prediction, target, valid, rms = _terminal_case()
    snapshots = [x.detach().clone() for x in (prediction, target, valid, rms)]
    rng = torch.get_rng_state().clone()
    loss, counts = scales.dynamic_m5_terminal_horizon_objective(
        prediction, target, valid, rms=rms)
    assert counts == {EARLY: 4, LATER: 6}
    assert loss.dtype == torch.float32
    assert float(loss.detach()) == pytest.approx(77/24)
    pooled_sse, pooled_n = scales.dynamic_m5_scaled_square_sum(
        prediction, target, valid, surface=TERMINAL, rms=rms)
    assert pooled_n == 10 and float(pooled_sse.detach()) == 30.
    assert float(pooled_sse.detach()/pooled_n) == 3.
    assert int(valid[..., 2].sum()) == 3
    loss.backward()
    torch.testing.assert_close(prediction.grad, _expected_balanced_gradient(prediction))
    assert torch.count_nonzero(prediction.grad[..., 2]) == 0
    assert torch.count_nonzero(prediction.grad[~valid]) == 0
    assert target.grad is None and rms.grad is None
    for value, before in zip((prediction, target, valid, rms), snapshots):
        assert torch.equal(value.detach(), before)
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("enabled,expected,counts", [
    ((EARLY,), 17/4, {EARLY: 4, LATER: 0}),
    ((LATER,), 13/6, {EARLY: 0, LATER: 6}),
    ((), None, {EARLY: 0, LATER: 0}),
])
def test_one_or_no_observed_group_has_no_fabricated_denominator(enabled, expected, counts):
    prediction, target, valid, rms = _terminal_case(enabled)
    loss, observed_counts = scales.dynamic_m5_terminal_horizon_objective(
        prediction, target, valid, rms=rms)
    assert observed_counts == counts
    if expected is None:
        assert loss is None and prediction.grad is None
    else:
        assert float(loss.detach()) == pytest.approx(expected)
        loss.backward()
        wanted = _expected_balanced_gradient(prediction) * 2
        if EARLY not in enabled:
            wanted[:, :15] = 0
        if LATER not in enabled:
            wanted[:, 15:] = 0
        torch.testing.assert_close(prediction.grad, wanted)


@pytest.mark.parametrize("fault", ["nonfinite_target", "masked_target", "nonzero_flat"])
def test_early_slice_validates_invalid_high_horizon_storage_before_slicing(fault):
    prediction, target, valid, rms = _terminal_case()
    prediction = prediction.detach().clone()
    if fault == "nonfinite_target":
        target[0, 479, 0] = float("nan")
    elif fault == "masked_target":
        target[0, 479, 0] = 1.  # This high-horizon cell is unobserved.
    else:
        prediction[0, 479, 2] = 1.
    prediction.requires_grad_()
    with pytest.raises(RuntimeError, match="DYNAMIC_M5_LOSS_SCALE_RUNTIME"):
        scales.dynamic_m5_scaled_square_sum(
            prediction, target, valid, surface=TERMINAL, rms=rms,
            terminal_query_range=(1, 15))
    assert prediction.grad is None


@pytest.mark.parametrize("query_range", [(1, 14), (16, 479), (1, 480), [1, 15]])
def test_only_two_selected_terminal_ranges_are_runtime_options(query_range):
    prediction, target, valid, rms = _terminal_case()
    with pytest.raises(RuntimeError, match="DYNAMIC_M5_TERMINAL_QUERY_RANGE_INVALID"):
        scales.dynamic_m5_scaled_square_sum(
            prediction, target, valid, surface=TERMINAL, rms=rms,
            terminal_query_range=query_range)


def test_nonterminal_surface_cannot_take_terminal_group_range():
    prediction = torch.zeros((1, 480, 2), dtype=torch.float32)
    with pytest.raises(RuntimeError, match="DYNAMIC_M5_TERMINAL_QUERY_RANGE_INVALID"):
        scales.dynamic_m5_scaled_square_sum(
            prediction, prediction.clone(), torch.zeros_like(prediction, dtype=torch.bool),
            surface=MFE, rms=torch.ones((480, 2)), terminal_query_range=(1, 15))


def test_finite_huge_target_overflow_is_rejected_by_existing_joint_owner_before_backward():
    prediction = torch.zeros((1, 480, 3), dtype=torch.float32, requires_grad=True)
    target = torch.zeros_like(prediction)
    valid = torch.zeros_like(prediction, dtype=torch.bool)
    target[0, 0, 0], valid[0, 0, 0] = 1e30, True
    loss, counts = scales.dynamic_m5_terminal_horizon_objective(
        prediction, target, valid, rms=torch.ones((480, 2)))
    assert counts == {EARLY: 1, LATER: 0} and not torch.isfinite(loss)
    model = _task_weight_model()
    with pytest.raises(RuntimeError, match="ENTRY_JOINT_TASK_LOSS_NONFINITE"):
        trainer._joint_task_loss(model, {TERMINAL: loss}, loss_units="train_rms")
    assert prediction.grad is None
    assert all(x.grad is None for x in model.task_log_variances.values())


def _all_surface_case(*, active_terminal):
    from gx1.contracts.entry_observed_market_horizons_v1 import (
        M5_FORECAST_VALID_BATCH_KEY, SUPPORT_FORECAST_TARGET_BATCH_KEY,
        SUPPORT_FORECAST_VALID_BATCH_KEY,
    )
    p, y, mask, rms = _terminal_case((EARLY, LATER) if active_terminal else ())
    out = {TERMINAL: p}
    batch = {"seq_x": torch.zeros(2, 1),
        observed.DYNAMIC_M5_MARKOUT_TARGET_KEY: y,
        observed.DYNAMIC_M5_MARKOUT_VALID_KEY: mask}
    units = {TERMINAL: rms}
    targets, masks = {}, {}
    for name in (MFE, MAE, CLOSED_MFE, CLOSED_MAE):
        prediction = torch.zeros((2, 480, 2), dtype=torch.float32)
        target = torch.zeros_like(prediction)
        valid = torch.zeros_like(prediction, dtype=torch.bool)
        unit = torch.ones((480, 2))
        row, query, side = (1, 479, 1) if name in (CLOSED_MFE, CLOSED_MAE) else (0, 0, 0)
        prediction[row, query, side], target[row, query, side] = (5., 1.) if row else (3., 1.)
        valid[row, query, side] = True
        unit[query, side] = 1. if name == MAE else 2.
        out[name], targets[name], masks[name], units[name] = prediction.requires_grad_(), target, valid, unit
    batch.update({
        observed.DYNAMIC_M5_MFE_TARGET_KEY: targets[MFE],
        observed.DYNAMIC_M5_MAE_TARGET_KEY: targets[MAE],
        observed.DYNAMIC_M5_PATH_VALID_KEY: masks[MFE],
        observed.DYNAMIC_M5_CLOSED_MFE_TARGET_KEY: targets[CLOSED_MFE],
        observed.DYNAMIC_M5_CLOSED_MAE_TARGET_KEY: targets[CLOSED_MAE],
        observed.DYNAMIC_M5_CLOSED_PATH_VALID_KEY: masks[CLOSED_MFE],
    })
    assert torch.equal(masks[MFE], masks[MAE]) and torch.equal(masks[CLOSED_MFE], masks[CLOSED_MAE])
    fp = torch.zeros((2, 10), dtype=torch.float32)
    fy, fm, fu = torch.zeros_like(fp), torch.zeros_like(fp, dtype=torch.bool), torch.ones(10)
    fp[0, 0], fy[0, 0], fm[0, 0], fu[0] = 6., 2., True, 2.
    fp[1, 9], fy[1, 9], fm[1, 9], fu[9] = -9., -3., True, 3.
    out[FORECAST], units[FORECAST] = fp.requires_grad_(), fu
    assert tuple(trainer.FORECAST_HORIZONS) == (1, 5, 12, 24)
    for index, horizon in enumerate(trainer.FORECAST_HORIZONS):
        batch[f"y_forecast_ret_K{horizon}"] = fy[:, index]
    batch.update({
        M5_FORECAST_VALID_BATCH_KEY: fm[:, :4],
        SUPPORT_FORECAST_TARGET_BATCH_KEY: fy[:, 4:],
        SUPPORT_FORECAST_VALID_BATCH_KEY: fm[:, 4:],
        scales.DYNAMIC_M5_LOSS_SCALE_BATCH_KEY: {
            # Unit-primitive fixture only; no published111 artifact identity is claimed.
            "identity": {"synthetic_runtime_unit_fixture_only": True}, "rms": units},
    })
    return out, batch


@pytest.mark.parametrize("active_terminal", [True, False])
def test_optimizer_adapter_preserves_other_three_tasks_and_old_raw_unit_metrics(active_terminal):
    out, batch = _all_surface_case(active_terminal=active_terminal)
    device = torch.device("cpu")
    raw, raw_counts = trainer._dynamic_m5_task_loss_surfaces(out, batch, device)
    pooled, counts = trainer._dynamic_m5_task_loss_surfaces(out, batch, device, loss_units="train_rms")
    before = {name: value.detach().clone() for name, value in pooled.items()}
    optimized, groups = trainer._dynamic_m5_optimizer_task_losses(
        out, batch, device, pooled_task_losses=pooled)
    assert groups == ({EARLY: 4, LATER: 6} if active_terminal else {EARLY: 0, LATER: 0})
    assert raw_counts == counts
    for name, expected in zip(OTHER_TASKS, (2.5, 4., 4.)):
        assert optimized[name] is pooled[name]
        assert float(optimized[name].detach()) == pytest.approx(expected)
    assert set(optimized) == set(pooled)
    if active_terminal:
        assert float(raw[TERMINAL].detach()) == 16.5
        assert float(pooled[TERMINAL].detach()) == 3.
        assert float(optimized[TERMINAL].detach()) == pytest.approx(77/24)
    else:
        assert TERMINAL not in optimized and TERMINAL not in raw
    after_raw, after_counts = trainer._dynamic_m5_task_loss_surfaces(out, batch, device)
    after_pooled, _ = trainer._dynamic_m5_task_loss_surfaces(out, batch, device, loss_units="train_rms")
    assert after_counts == counts
    assert all(torch.equal(pooled[name].detach(), before[name]) for name in pooled)
    assert all(torch.equal(raw[name], after_raw[name]) and torch.equal(pooled[name], after_pooled[name])
               for name in raw)


@pytest.mark.parametrize("active", [True, False])
def test_terminal_precision_and_log_variance_apply_once_or_are_completely_omitted(active):
    prediction, target, valid, rms = _terminal_case((EARLY, LATER) if active else ())
    terminal, _ = scales.dynamic_m5_terminal_horizon_objective(prediction, target, valid, rms=rms)
    losses = {name: torch.tensor(value, dtype=torch.float32, requires_grad=True)
              for name, value in zip(OTHER_TASKS, (2., 3., 5.))}
    if terminal is not None:
        losses[TERMINAL] = terminal
    model = _task_weight_model()
    with torch.no_grad():
        model.task_log_variances[TERMINAL].fill_(math.log(2))
    joint, _ = trainer._joint_task_loss(model, losses, loss_units="train_rms")
    assert float(joint.detach()) == pytest.approx(10. + (77/48 + math.log(2) if active else 0.))
    joint.backward()
    if active:
        assert float(model.task_log_variances[TERMINAL].grad) == pytest.approx(1 - 77/48)
        torch.testing.assert_close(prediction.grad, _expected_balanced_gradient(prediction)/2)
    else:
        assert prediction.grad is None and model.task_log_variances[TERMINAL].grad is None


def test_separate_optimizer_evidence_uses_active_batch_rows_and_reports_all_batch_kinds():
    accumulator = trainer._new_dynamic_m5_optimizer_loss_evidence()
    empty = trainer._finalize_dynamic_m5_optimizer_loss_evidence(accumulator)["dynamic_m5_terminal_optimizer_objective"]
    assert empty["batch_row_weighted_terminal_optimizer_objective_mean"] is None
    trainer._accumulate_dynamic_m5_optimizer_loss_evidence(
        accumulator, {TERMINAL: torch.tensor(2.)}, {EARLY: 3, LATER: 6}, batch_rows=2)
    trainer._accumulate_dynamic_m5_optimizer_loss_evidence(
        accumulator, {TERMINAL: torch.tensor(10.)}, {EARLY: 2, LATER: 0}, batch_rows=1)
    trainer._accumulate_dynamic_m5_optimizer_loss_evidence(
        accumulator, {MFE: torch.tensor(1.)}, {EARLY: 0, LATER: 0}, batch_rows=3)
    value = trainer._finalize_dynamic_m5_optimizer_loss_evidence(accumulator)["dynamic_m5_terminal_optimizer_objective"]
    assert value["objective_contract"] == scales.dynamic_m5_terminal_horizon_objective_contract()
    assert value["group_observed_cells"] == {EARLY: 5, LATER: 6}
    assert value["nonempty_group_batches"] == {"none": 1, "one": 1, "two": 1}
    assert value["processed_batch_rows"] == 6 and value["active_terminal_batch_rows"] == 3
    assert value["batch_row_weighted_terminal_optimizer_objective_mean"] == pytest.approx(14/3)
    assert value["batch_row_weighted_terminal_optimizer_objective_mean"] != pytest.approx(38/11)
    assert value["aggregation_scope"] == "current_train_epoch_call_or_single_curve_validation_pass_no_cross_invocation_history"


@pytest.mark.parametrize("route", ["train", "curve"])
def test_native_train_and_curve_routes_consume_same_optimizer_adapter_with_separate_evidence(monkeypatch, tmp_path, route):
    from tests.test_entry_model_native_training_objective import _dynamic_gate_execution_case
    case = _dynamic_gate_execution_case(monkeypatch, tmp_path, compact=False)
    calls = []
    def optimizer_plumbing(out, batch, device, *, pooled_task_losses):
        # This preexisting fake-model fixture isolates native routing, not group math.
        calls.append(float(pooled_task_losses[TERMINAL].detach()))
        losses = dict(pooled_task_losses)
        losses[TERMINAL] = 2 * losses[TERMINAL]
        return losses, {EARLY: 1, LATER: 1}
    monkeypatch.setattr(case.trainer, "_dynamic_m5_optimizer_task_losses", optimizer_plumbing)
    if route == "curve":
        result = case.trainer._validate_dynamic_m5_curves(case.model.eval(), case.loader, torch.device("cpu"))
        loss, stats = result[0], result[-1]
    else:
        optimizer = torch.optim.SGD(case.model.parameters(), lr=0.)
        loss, stats, complete = case.trainer.train_epoch(
            case.model, case.target, case.loader, optimizer, torch.device("cpu"), 1, {}, {})
        assert complete
    assert calls == [0.625] and loss == pytest.approx(3.125)
    assert stats[f"joint_task_raw_loss_mean_{TERMINAL}"] == pytest.approx(0.625)
    assert stats[f"joint_task_unit_loss_mean_{TERMINAL}"] == pytest.approx(0.625)
    value = stats["dynamic_m5_terminal_optimizer_objective"]
    assert value["batch_row_weighted_terminal_optimizer_objective_mean"] == pytest.approx(1.25)
    assert value["group_observed_cells"] == {EARLY: 1, LATER: 1}
    assert value["nonempty_group_batches"] == {"none": 0, "one": 0, "two": 1}
    assert value["processed_batch_rows"] == value["active_terminal_batch_rows"] == 2
    assert all(parameter.grad is None for parameter in case.model.exit_path_proj.parameters())


def test_new_objective_contract_preserves_111_units_and_rejects_resealed_boundary_drift():
    unit_contract = scales.dynamic_m5_loss_scale_contract()
    assert unit_contract["contract_sha256"] == FROZEN_111_CONTRACT_SHA
    assert scales.DYNAMIC_M5_TERMINAL_HORIZON_GROUPS == ((EARLY, 1, 15), (LATER, 16, 480))
    contract = scales.dynamic_m5_terminal_horizon_objective_contract()
    assert contract["loss_unit_contract_sha256"] == FROZEN_111_CONTRACT_SHA
    assert contract["optimizer_objective_is_pooled_SSE_over_N"] is False
    assert contract["epsilon_clipping_or_unit_fallback"] is False
    metadata = objective.training_objective_contract_metadata(entry_training_phase=PHASE)
    assert metadata["dynamic_terminal_optimizer_objective"] == contract
    assert metadata["pooled_TRAIN_RMS_component_diagnostics_preserved"] is True
    assert metadata["dynamic_component_losses"][TERMINAL] == "mean_nonempty_terminal_horizon_group_TRAIN_RMS_squared_error"
    broken = copy.deepcopy(metadata)
    changed = broken["dynamic_terminal_optimizer_objective"]
    changed["groups"][0]["query_minutes"][1] = 14
    changed.pop("contract_sha256")
    changed["contract_sha256"] = canonical_sha256(changed)
    with pytest.raises(RuntimeError, match="TRAINING_OBJECTIVE_CONTRACT_INVALID"):
        objective.require_training_objective_contract(broken, context="SYNTHETIC_OBJECTIVE")
    assert scales.dynamic_m5_loss_scale_contract() == unit_contract


def test_changed_objective_source_binding_cannot_reopen_saved_session_metadata(monkeypatch, tmp_path):
    from tests.test_candidate_training_session import _contract, _recipe_source_provenance
    contract = _contract()
    provenance = _recipe_source_provenance(source_commit="a" * 40)
    key = "python:gx1/contracts/entry_model_native_training_objective_v1.py"
    provenance["source_bindings"][key] = {
        **provenance["source_bindings"]["trainer"],
        "path": "/repo/gx1/contracts/entry_model_native_training_objective_v1.py"}
    def seal(value):
        value["source_bindings_sha256"] = hashlib.sha256(json.dumps(
            value["source_bindings"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    seal(provenance)
    contract["recipe_source_provenance"] = provenance
    output = tmp_path / "synthetic-session"
    session = trainer._CandidateTrainingSession(out_bundle_dir=output, contract=contract)
    assert not session._active_path.exists()
    def forbidden_checkpoint(*args, **kwargs):
        raise AssertionError("metadata mismatch must fail before checkpoint reading")
    monkeypatch.setattr(torch, "load", forbidden_checkpoint)
    changed = copy.deepcopy(contract)
    changed["recipe_source_provenance"]["source_bindings"][key]["sha256"] = "b" * 64
    seal(changed["recipe_source_provenance"])
    with pytest.raises(RuntimeError, match="CANDIDATE_TRAINING_SESSION_CONTRACT_MISMATCH"):
        trainer._CandidateTrainingSession(out_bundle_dir=output, contract=changed, read_only=True)
    assert not list(tmp_path.rglob("*.pt"))
