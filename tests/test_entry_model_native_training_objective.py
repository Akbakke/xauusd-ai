from __future__ import annotations

import pytest

from gx1.contracts.entry_fitted_q_v1 import entry_fitted_q_contract
from gx1.contracts.entry_model_native_joint_task_weighting_v1 import (
    JOINT_TASK_NAMES,
    joint_task_weighting_objective_contract,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    SCHEMA_VERSION,
    ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
    require_native_serving_training_phase,
    require_training_objective_contract,
    training_objective_contract_metadata,
)


def test_training_objective_contract_is_exact_and_learned() -> None:
    payload = training_objective_contract_metadata()

    assert payload["all_advertised_heads_supervised"] is True
    assert payload["fixed_relative_task_weights"] is False
    assert payload["handwritten_rank_losses"] is False
    assert payload["handwritten_composite_weights"] is False
    assert payload["handwritten_gate_regularization"] is False
    assert payload["fixed_target_normalization_scales"] is False
    assert payload["handwritten_distribution_forcing"] is False
    assert payload["target_units"] == "raw_native_units"
    # Fitted-Q replaced every classification/probability objective: the only
    # Entry authority is the raw-bps Q argmax, and its target is the frozen
    # TRAIN Exit teacher rather than any fixed forward horizon.
    assert payload["classification_or_probability_loss_authority"] is False
    assert payload["fixed_horizon_target_authority"] is False
    assert payload["entry_action_q_loss"] == "masked_raw_bps_mean_squared_error"
    assert payload["unified_exit_action_q_loss"] == (
        "masked_raw_bps_mean_squared_error"
    )
    assert payload["entry_training_phase"] == "joint_fitted_q"
    assert payload["exit_gradients_active"] is True
    assert payload["entry_action_q_target"] == (
        "stop_gradient_frozen_train_exit_target_model_first_state_value"
    )
    # The objective may not invent its own decision rule: it must restate the
    # fitted-Q owner's decision verbatim.
    assert payload["entry_decision_authority"] == (
        entry_fitted_q_contract()["decision"]
    )
    assert payload["joint_task_names"] == list(JOINT_TASK_NAMES)
    assert payload["joint_task_weighting"] == joint_task_weighting_objective_contract()
    assert require_training_objective_contract(payload, context="TEST") == payload


def test_entry_only_phase_declares_observed_market_target_without_exit() -> None:
    payload = training_objective_contract_metadata(entry_training_phase="entry_only")

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["entry_training_phase"] == "entry_only"
    assert payload["exit_gradients_active"] is False
    assert payload["entry_action_q_target"] == (
        "observed_executable_terminal_cash_return_net_of_declared_costs_"
        "at_frozen_train_horizon_without_exit_model"
    )
    assert payload["entry_action_q_loss"] == "masked_raw_bps_mean_squared_error"
    assert require_training_objective_contract(payload, context="TEST") == payload
    # A phase may not borrow the other phase's target or gradient flag.
    for field, bad in (
        ("entry_action_q_target", "stop_gradient_frozen_train_exit_target_model_first_state_value"),
        ("exit_gradients_active", True),
        ("entry_training_phase", "joint_fitted_q"),
    ):
        drifted = dict(payload)
        drifted[field] = bad
        with pytest.raises(RuntimeError, match="TRAINING_OBJECTIVE_CONTRACT_INVALID"):
            require_training_objective_contract(drifted, context="TEST")
    with pytest.raises(RuntimeError, match="ENTRY_TRAINING_PHASE_INVALID"):
        training_objective_contract_metadata(entry_training_phase="exit_only")


@pytest.mark.parametrize(
    "field,bad",
    [
        ("entry_training_phase", "exit_only"),
        ("exit_gradients_active", False),
        ("fixed_relative_task_weights", True),
        ("handwritten_rank_losses", True),
        ("handwritten_composite_weights", True),
        ("handwritten_gate_regularization", True),
        ("fixed_target_normalization_scales", True),
        ("handwritten_distribution_forcing", True),
        ("classification_or_probability_loss_authority", True),
        ("fixed_horizon_target_authority", True),
        ("entry_action_q_loss", "weighted_cross_entropy"),
        ("unified_exit_action_q_loss", "weighted_cross_entropy"),
        ("entry_action_q_target", "pathwise_forward_outcome"),
        ("entry_decision_authority", "expected_utility_threshold"),
        ("target_units", "scaled_bps"),
    ],
)
def test_training_objective_rejects_contract_drift(field: str, bad: object) -> None:
    payload = training_objective_contract_metadata()
    payload[field] = bad

    with pytest.raises(RuntimeError, match="TRAINING_OBJECTIVE_CONTRACT_INVALID"):
        require_training_objective_contract(payload, context="TEST")


def test_training_objective_rejects_extra_surface() -> None:
    payload = training_objective_contract_metadata()
    payload["fixed_positive_loss_weights"] = {"direction": 1.0}

    with pytest.raises(
        RuntimeError,
        match=r"TRAINING_OBJECTIVE_CONTRACT_INVALID.*"
        r"unexpected=\['fixed_positive_loss_weights'\]",
    ):
        require_training_objective_contract(payload, context="TEST")


def test_entry_only_horizon_phase_declares_the_20_day_observed_target_without_exit() -> None:
    payload = training_objective_contract_metadata(entry_training_phase="entry_only_observed_horizon_20d")
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["entry_training_phase"] == "entry_only_observed_horizon_20d"
    assert payload["exit_gradients_active"] is False
    assert "20_trading_days" in payload["entry_action_q_target"] and "without_exit_model" in payload["entry_action_q_target"]
    assert payload["fixed_horizon_target_authority"] is False



def test_m5_path_phase_is_utility_without_exit_or_legacy_auxiliary_learning():
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_M5_PATH_TRAINING_PHASE
    from gx1.contracts.entry_observed_market_v1 import entry_observed_m5_path_contract
    value = training_objective_contract_metadata(entry_training_phase=ENTRY_ONLY_M5_PATH_TRAINING_PHASE)
    assert value['entry_primary_target_contract'] == entry_observed_m5_path_contract()
    assert value['entry_primary_target_contract']['horizon_m5_bars'] == 19
    assert value['entry_primary_target_contract']['hindsight_best_exit_cash_claim'] is False
    assert value['exit_gradients_active'] is False
    assert value['active_joint_task_names'] == ['entry_action_q', 'forecast_return_bps']
    assert value['all_advertised_heads_supervised'] is False
    assert value['fixed_relative_task_weights'] is False
    assert 'dip' in value['inactive_legacy_auxiliary_heads']
    assert require_training_objective_contract(value, context='TEST') == value


def test_dynamic_objective_matches_existing_quality_bps_candidate_without_native_authority():
    from gx1.contracts.entry_observed_market_v1 import entry_observed_dynamic_m5_path_contract
    value = training_objective_contract_metadata(entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    quality = entry_observed_dynamic_m5_path_contract()["candidate_quality_contract"]
    assert value["dynamic_candidate_quality_contract"] == quality
    assert quality["units"] == "quality_bps" and quality["native_authority_admitted"] is False
    assert value["entry_decision_authority"] == "none_dynamic_quality_bps_candidate_not_admitted"
    assert value["dynamic_candidate_is_cash"] is False
    assert value["dynamic_candidate_is_unconditional_expected_quality"] is False
    assert value["dynamic_candidate_has_supervised_target"] is False
    assert value["train_fitted_loss_scales"] is True
    assert value["raw_bps_component_diagnostics_preserved"] is True
    assert value["dynamic_component_losses"] == {
        "dynamic_terminal_net_markout_bps": "mean_nonempty_terminal_horizon_group_TRAIN_RMS_squared_error",
        "dynamic_mfe_bps": "mean_nonempty_mfe_horizon_group_TRAIN_RMS_squared_error",
        **{name: "independently_masked_TRAIN_target_RMS_scaled_squared_error"
           for name in ("dynamic_pre_mfe_mae_upper_bps", "forecast_return_bps")},
    }
    assert value["dynamic_loss_unit_contract"]["fit_partition"] == "train"
    assert value["dynamic_loss_unit_contract"]["raw_target_and_output_units"] == "bps"
    assert value["dynamic_loss_unit_contract"]["centering"] is False
    assert value["dynamic_loss_unit_contract"]["epsilon_or_unit_fallback"] is False
    assert value["dynamic_candidate_native_authority_admitted"] is False
    assert value["exit_gradients_active"] is False
    assert "dynamic_candidate_is_cash_or_bps" not in value
    assert "dynamic_candidate_is_expected_future_ratio" not in value
    assert require_training_objective_contract(value, context="TEST") == value


@pytest.mark.parametrize("fault", ["ratio", "units", "cash", "supervised_best_horizon", "authority"])
def test_dynamic_objective_rejects_stale_ratio_or_false_quality_claims(fault):
    import copy
    value = training_objective_contract_metadata(entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    value = copy.deepcopy(value)
    if fault == "ratio": value["entry_decision_authority"] = "none_dynamic_unitless_ratio_candidate_not_admitted"
    elif fault == "units": value["dynamic_candidate_quality_contract"]["units"] = "unitless"
    elif fault == "cash": value["dynamic_candidate_is_cash"] = True
    elif fault == "supervised_best_horizon": value["dynamic_candidate_has_supervised_target"] = True
    else: value["dynamic_candidate_native_authority_admitted"] = True
    with pytest.raises(RuntimeError, match="TRAINING_OBJECTIVE_CONTRACT_INVALID"):
        require_training_objective_contract(value, context="TEST")


def test_unadmitted_dynamic_phase_cannot_export_or_pass_legacy_smoke():
    from gx1.contracts.entry_model_native_smoke_bundle_audit_v1 import (
        require_smoke_bundle_training_pipeline_contract, require_smoke_bundle_audit_contract)
    objective = training_objective_contract_metadata(entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    with pytest.raises(RuntimeError, match="DYNAMIC_ENTRY_CANDIDATE_NATIVE_AUTHORITY_NOT_ADMITTED"):
        require_native_serving_training_phase(ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE, context="ENTRY_TRAIN_EXPORT")
    assert require_native_serving_training_phase("joint_fitted_q", context="TEST") == "joint_fitted_q"
    for owner in (require_smoke_bundle_training_pipeline_contract, require_smoke_bundle_audit_contract):
        with pytest.raises(RuntimeError, match="DYNAMIC_ENTRY_CANDIDATE_NATIVE_AUTHORITY_NOT_ADMITTED"):
            owner({"model_native_training_objective_contract": {"objective": objective}}, context="TEST")


def test_dynamic_curve_observer_must_be_callable_before_any_forward():
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    with pytest.raises(RuntimeError, match="BATCH_OBSERVER_INVALID"):
        trainer._validate_dynamic_m5_curves(None, None, None, batch_observer=1)



def _dynamic_gate_execution_case(monkeypatch, tmp_path, *, compact):
    """Exercise real gate evidence owners with synthetic loss/target plumbing."""
    from types import SimpleNamespace
    import torch
    from torch import nn
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.features.entry_specialist_feature_groups_v1 import compile_compact_entry_execution_profile

    compiled = compile_compact_entry_execution_profile() if compact else None
    reference = ({key: compiled[key] for key in ("name", "contract_sha256")}
                 if compact else None)
    widths = ({"specialist_gate": len(compiled["local_specialists"]),
               "tf_gate": trainer.ENTRY_MTF_CONTEXT_COUNT,
               "family_tf_cooperation_gate": trainer.ENTRY_MTF_CONTEXT_COUNT * len(compiled["mtf_specialists"])}
              if compact else dict(trainer._MODEL_NATIVE_COOPERATION_GATE_WIDTHS))
    shape = ((trainer.ENTRY_MTF_CONTEXT_COUNT, len(compiled["mtf_indices"]))
             if compact else trainer._MODEL_NATIVE_FEATURE_TF_GATE_SHAPE)
    tasks = trainer.joint_task_names_for_phase(ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    calls = {"forward": 0, "materialize": 0}

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.entry_training_phase = ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
            self.entry_execution_profile = reference
            self.shared_encoder = nn.Linear(1, 1, bias=False)
            self.exit_path_proj = nn.Linear(1, 1).requires_grad_(False)
            with torch.no_grad():
                self.shared_encoder.weight.fill_(0.5)
            self.task_log_variances = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(())) for name in tasks
            })

        def forward(self, seq_x, snap_x, **kwargs):
            calls["forward"] += 1
            rows = int(seq_x.shape[0])
            return {
                "synthetic_loss": self.shared_encoder(seq_x).square().mean(),
                **{name: torch.full((rows, width), 1. / width) for name, width in widths.items()},
                "family_tf_feature_gate": torch.ones(rows, *shape),
                "dynamic_entry_action_quality_bps_candidate": torch.tensor([[2., 1., 0.]]).expand(rows, -1),
            }

    class ForbiddenTarget(nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("dynamic Entry gate diagnostics must not query TARGET/Exit")

    dataset = object.__new__(trainer.EntryV10CtxDataset)
    dataset._unified_exit_lifecycle_v2 = None
    dataset._entry_observed_market_binding = {"identity": {"training_phase": ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE}}
    from tests.test_entry_dynamic_m5_loss_scale import _synthetic_loaded_loss_scales
    trainer._bind_dynamic_m5_loss_scales(dataset, _synthetic_loaded_loss_scales(tmp_path),
        device=torch.device("cpu"))
    batch = {
        "seq_x": torch.tensor([[1.], [2.]]), "snap_x": torch.zeros(2, 1),
        "ctx_cont": torch.zeros(2, 1), "ctx_cat": torch.zeros(2, 1, dtype=torch.long),
        "entry_row_index": torch.tensor([0, 1]),
    }

    class Loader:
        def __init__(self):
            self.dataset = dataset
        def __len__(self):
            return 1
        def __iter__(self):
            yield batch

    def materialize(value, source):
        assert source is dataset
        calls["materialize"] += 1
        return value

    monkeypatch.setattr(trainer, "_materialize_dynamic_m5_batch_targets", materialize)
    monkeypatch.setattr(trainer, "_multi_tf_kwargs_from_batch", lambda *args: {})
    monkeypatch.setattr(trainer, "_model_forward_fp32", lambda model, *args, **kwargs: model(*args, **kwargs))
    monkeypatch.setattr(trainer, "_dynamic_m5_task_loss_surfaces", lambda out, batch, device, *, loss_units="raw_bps": (
        {name: out["synthetic_loss"] for name in tasks},
        {name: int(batch["seq_x"].shape[0]) for name in tasks},
    ))
    def optimizer_gate_only_loss(out, batch, device, *, pooled_task_losses):
        # Gate-execution fixture only: keep the injected fake task losses and
        # their synthetic count. Real terminal group math is tested separately.
        return dict(pooled_task_losses), {
            "early_1_15": int(batch["seq_x"].shape[0]), "later_16_480": 0,
        }
    monkeypatch.setattr(trainer, "_dynamic_m5_optimizer_task_losses", optimizer_gate_only_loss)
    model = Model()
    return SimpleNamespace(trainer=trainer, model=model, target=ForbiddenTarget(),
                           loader=Loader(), widths=widths, shape=shape, calls=calls,
                           exit_initial={name: value.detach().clone() for name, value in model.exit_path_proj.state_dict().items()})


def _assert_dynamic_gate_stats(stats, case):
    for name, width in case.widths.items():
        assert len(stats[name + "_mean_weight"]) == width
        assert stats[name + "_rows"] == 2
    assert len(stats["family_tf_feature_gate_mean_weight"]) == case.shape[0] * case.shape[1]
    assert stats["family_tf_feature_gate_rows"] == 2
    assert case.calls == {"forward": 1, "materialize": 1}


@pytest.mark.parametrize("compact", [False, True])
def test_dynamic_curve_validator_uses_bound_execution_gate_shapes(monkeypatch, tmp_path, compact):
    import math
    import torch
    case = _dynamic_gate_execution_case(monkeypatch, tmp_path, compact=compact)
    result = case.trainer._validate_dynamic_m5_curves(case.model.eval(), case.loader, torch.device("cpu"))
    assert math.isfinite(result[0])
    _assert_dynamic_gate_stats(result[-1], case)
    assert result[-1]["exit_model_forward_count"] == 0


@pytest.mark.parametrize("compact", [False, True])
def test_dynamic_train_epoch_uses_bound_execution_gate_shapes(monkeypatch, tmp_path, compact):
    import math
    import torch
    case = _dynamic_gate_execution_case(monkeypatch, tmp_path, compact=compact)
    optimizer = torch.optim.SGD(case.model.parameters(), lr=0.001)
    loss, stats, complete = case.trainer.train_epoch(
        case.model, case.target, case.loader, optimizer, torch.device("cpu"), 1, {}, {},
    )
    assert complete and math.isfinite(loss)
    _assert_dynamic_gate_stats(stats, case)
    assert stats["exit_training_active"] is False
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in case.model.exit_path_proj.parameters())
    assert all(torch.equal(value, case.exit_initial[name])
               for name, value in case.model.exit_path_proj.state_dict().items())


def test_dynamic_gate_execution_none_keeps_original_defaults():
    from types import SimpleNamespace
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    assert trainer._dynamic_m5_gate_execution_kwargs(SimpleNamespace()) == ({}, {})
    assert trainer._dynamic_m5_gate_execution_kwargs(SimpleNamespace(entry_execution_profile=None)) == ({}, {})


def test_dynamic_gate_execution_rejects_forged_profile_before_diagnostics():
    from types import SimpleNamespace
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    with pytest.raises(RuntimeError, match="COMPACT_ENTRY_EXECUTION_PROFILE_REFERENCE"):
        trainer._dynamic_m5_gate_execution_kwargs(SimpleNamespace(
            entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
            entry_execution_profile={"name": "unregistered", "contract_sha256": "0" * 64},
        ))


@pytest.mark.parametrize("fault", ["old_raw_loss", "control_fit", "centering", "epsilon", "hidden_target_unit"])
def test_dynamic_objective_rejects_loss_unit_semantic_drift(fault):
    import copy
    value = copy.deepcopy(training_objective_contract_metadata(
        entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE))
    if fault == "old_raw_loss":
        value["dynamic_component_losses"]["forecast_return_bps"] = "independently_masked_raw_bps_mean_squared_error"
    elif fault == "control_fit":
        value["dynamic_loss_unit_contract"]["fit_partition"] = "val"
    elif fault == "centering":
        value["dynamic_loss_unit_contract"]["centering"] = True
    elif fault == "epsilon":
        value["dynamic_loss_unit_contract"]["epsilon_or_unit_fallback"] = True
    else:
        value["dynamic_loss_unit_contract"]["raw_target_and_output_units"] = "normalized_target"
    with pytest.raises(RuntimeError, match="TRAINING_OBJECTIVE_CONTRACT_INVALID"):
        require_training_objective_contract(value, context="TEST")
