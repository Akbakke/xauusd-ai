"""Observed Entry native wiring, with small synthetic files and optimizer steps."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    M5_FORECAST_VALID_BATCH_KEY, MODEL_NATIVE_FORECAST_TARGET_COLUMNS,
    MODEL_NATIVE_FORECAST_VALID_COLUMNS, M5_PRIMARY_OBSERVED_EMISSION_SCOPE,
    model_native_aux_target_contract_metadata,
)
from torch import nn

from gx1.contracts import entry_observed_market_v1 as observed
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from tests.test_entry_observed_market import economics
from tests.test_native_prefix_coordinator import (
    physical_prepared, physical_component_templates, physical_component_case, _binding_args,
    PrefixHarness, digest, equal_tree,
)
from tests.test_native_prefix_initial_measurement import initial_scope, prefix_scope, scope
from tests.test_native_prefix_learning_measurement import learning_scope
from tests.test_native_prefix_recipe import native, _write, _bind


def dataset_scope(tmp_path, economics):
    frame = pd.DataFrame({
        "time": pd.date_range("2025-05-01T00:00:00Z", periods=4, freq="5min"),
        observed.GROSS_TARGET_COLUMNS[0]: [10., -8., 3., 2.],
        observed.GROSS_TARGET_COLUMNS[1]: [-12., 6., -5., -4.],
    })
    path = tmp_path / "market.parquet"
    frame.to_parquet(path, index=False)
    source = {
        "parquet": _bind(path), "manifest": _write(tmp_path / "manifest.json", {}),
        "physical_rows": len(frame),
        "clock_sha256": hashlib.sha256(frame.time.array.asi8.astype("<i8").tobytes()).hexdigest(),
    }
    dataset = SimpleNamespace(
        parquet_path=path, df=frame[["time"]].copy(),
        _policy_dependent_auxiliary_bound_rows=np.array([True, True, False, True]),
        _policy_dependent_auxiliary_binding={
            "mode": "original_physical_split_targets", "role": "TRAIN",
            "parent_entry_parquet": source["parquet"], "parent_entry_manifest": source["manifest"],
            "policies": {"direction_policy": {"policy_sha256": "a" * 64}},
        },
    )
    scope_value = {
        "binding": {"path": "/synthetic/design.json", "sha256": "b" * 64},
        "target_contract": observed.entry_observed_market_contract(),
        "training_phase": "entry_only", "economics": economics,
        "reference_horizon_m5_bars": 19, "elapsed_wall_clock_seconds": 5700,
        "direction_policy_sha256": "a" * 64, "target_m1_source_sha256": "c" * 64,
        "physical_sources": {"train": source},
        "train_control_cutoff": "2025-06-01T00:00:00Z",
        "development_control_entry_end_exclusive": "2026-07-01T00:00:00Z",
    }
    return dataset, scope_value


def test_dataset_binding_keeps_price_inputs_and_unadmitted_rows_separate(tmp_path, economics):
    dataset, scope_value = dataset_scope(tmp_path, economics)
    before = dataset.df.copy(deep=True)
    observed.bind_entry_observed_market_dataset(dataset, scope_value)
    pd.testing.assert_frame_equal(dataset.df, before)
    targets = dataset._entry_observed_market_targets
    assert np.isnan(targets[2]).all()
    assert targets[0, 0] > 0 and targets[1, 0] < 0
    assert targets[1, 1] == 2.
    assert np.array_equal(targets[[0, 1, 3], 2], np.zeros(3))
    assert not targets.flags.writeable
    assert dataset._entry_observed_market_binding["admitted_rows"] == 3
    with pytest.raises(RuntimeError, match="ALREADY_BOUND"):
        observed.bind_entry_observed_market_dataset(dataset, scope_value)


@pytest.mark.parametrize("fault", ["parquet_bytes", "input_clock", "clock_digest", "split_cutoff", "wrong_policy", "missing_mask"])
def test_dataset_binding_rejects_changed_source_clock_support_or_policy(tmp_path, economics, fault):
    dataset, scope_value = dataset_scope(tmp_path, economics)
    if fault == "parquet_bytes": dataset.parquet_path.write_bytes(b"changed source")
    elif fault == "input_clock": dataset.df.loc[0, "time"] += pd.Timedelta(minutes=5)
    elif fault == "clock_digest": scope_value["physical_sources"]["train"]["clock_sha256"] = "d" * 64
    elif fault == "split_cutoff": scope_value["train_control_cutoff"] = "2025-05-01T01:00:00Z"
    elif fault == "wrong_policy": scope_value["direction_policy_sha256"] = "d" * 64
    else: dataset._policy_dependent_auxiliary_bound_rows = None
    with pytest.raises(RuntimeError):
        observed.bind_entry_observed_market_dataset(dataset, scope_value)
    assert not hasattr(dataset, "_entry_observed_market_targets")


class SmallEntry(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_encoder = nn.Linear(2, 3)
        self.head_entry_action_q = nn.Linear(3, 3)
        self.entry_decision_token = nn.Linear(3, 2)
        self.exit_path_proj = nn.Linear(2, 2)
        self.head_exit_action = nn.Linear(2, 2)
        self.task_log_variances = nn.ParameterDict({key: nn.Parameter(torch.zeros(())) for key in trainer.JOINT_TASK_NAMES})

    def forward(self, seq_x, snap_x, **kwargs):
        hidden = self.shared_encoder(seq_x)
        return {
            "entry_action_q_bps": self.head_entry_action_q(hidden), "shared": hidden,
            "position_size_logit": hidden[:, 0],
            trainer.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY: self.entry_decision_token(hidden),
        }


class ForbiddenTeacher(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("Entry-only training must never query TARGET")


def training_fixture(monkeypatch, *, inject_exit_loss=False):
    dataset = object.__new__(trainer.EntryV10CtxDataset)
    dataset._unified_exit_lifecycle_v2 = None
    dataset._entry_observed_market_binding = {"identity": {"training_phase": "entry_only"}}
    batch = {
        "seq_x": torch.tensor([[1., 2.], [3., 1.]]),
        "snap_x": torch.zeros(2, 2), "ctx_cont": torch.zeros(2, 1),
        "ctx_cat": torch.zeros(2, 1, dtype=torch.int64), "entry_row_index": torch.tensor([0, 1]),
        "entry_observed_market_target_bps": torch.tensor([[4., -6., 0.], [-8., 6., 0.]]),
        "y_position_size_target": torch.tensor([.5, .5]), "y_position_size_mask": torch.ones(2),
    }
    class Loader:
        def __init__(self):
            self.dataset = dataset
        def __len__(self):
            return 2
        def __iter__(self):
            yield batch
            yield batch
    monkeypatch.setattr(trainer, "_multi_tf_kwargs_from_batch", lambda *args: {})
    monkeypatch.setattr(trainer, "_model_forward_fp32", lambda model, *args, **kwargs: model(*args, **kwargs))
    def forbidden(**kwargs):
        raise AssertionError("Entry-only training must never run Exit forward/backward")
    monkeypatch.setattr(trainer, "_train_unified_exit_full_population", forbidden)
    monkeypatch.setattr(trainer, "_accumulate_cooperation_gate_epoch", lambda *a: None)
    monkeypatch.setattr(trainer, "_accumulate_feature_tf_gate_epoch", lambda *a: None)
    monkeypatch.setattr(trainer, "_finalize_cooperation_gate_epoch", lambda *a: {})
    monkeypatch.setattr(trainer, "_finalize_feature_tf_gate_epoch", lambda *a: {})
    monkeypatch.setattr(trainer, "_finalize_unified_exit_gate_epoch", forbidden)
    monkeypatch.setattr(trainer, "_side_mae_auxiliary_loss", lambda out, *a: (
        out["shared"].square().mean(), {"side_mae_loss": float(out["shared"].detach().square().mean())}))
    monkeypatch.setattr(trainer, "_trendline_event_aux_loss", lambda out, *a: (
        out["shared"].square().mean(), {"trendline_event_loss": 1., "trendline_event_rows": 2,
                                     "trendline_support_rows": 1, "trendline_resistance_rows": 1}))
    monkeypatch.setattr(trainer, "_require_active_aux_head_prediction", lambda out, *a, **kw: out[kw["output_name"]])
    monkeypatch.setattr(trainer, "dip_forecast_task_losses", lambda out, *a: {
        "forecast_return_bps": (out[trainer.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY].square().mean()
                                if inject_exit_loss else out["shared"].square().mean())})
    return Loader()


def test_real_train_epoch_adamw_updates_entry_and_preserves_all_exit_weights(monkeypatch):
    torch.manual_seed(910)
    model = SmallEntry()
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.5)
    supervised, gradients = {}, {}
    loss, stats, complete = trainer.train_epoch(
        model, ForbiddenTeacher().eval(), training_fixture(monkeypatch), optimizer,
        torch.device("cpu"), 1, supervised, gradients)
    assert complete and np.isfinite(loss)
    assert stats["exit_training_active"] is False
    assert stats["unified_exit_raw_bps_q_mse_mean"] is None
    assert "unified_exit_action" not in supervised
    assert supervised["entry_action_q"] and supervised["forecast_return_bps"]
    assert gradients["entry_action_q"] is True
    assert not torch.equal(model.shared_encoder.weight, before["shared_encoder.weight"])
    assert not torch.equal(model.head_entry_action_q.weight, before["head_entry_action_q.weight"])
    for name, parameter in model.named_parameters():
        if observed.is_exit_owned_parameter(name):
            assert torch.equal(parameter, before[name])
            assert parameter not in optimizer.state  # None gradients must not initialize AdamW slots.


def test_exit_gradient_leak_stops_before_optimizer_mutation(monkeypatch):
    torch.manual_seed(910)
    model = SmallEntry()
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.5)
    with pytest.raises(RuntimeError, match="EXIT_GRADIENT_FORBIDDEN"):
        trainer.train_epoch(
            model, ForbiddenTeacher().eval(), training_fixture(monkeypatch, inject_exit_loss=True),
            optimizer, torch.device("cpu"), 1, {}, {})
    assert not optimizer.state
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])


@pytest.mark.parametrize("fault", [None, "missing_control", "different_target", "wrong_source"])
def test_durable_native_contract_binds_both_target_roles(physical_prepared, fault):
    data = physical_prepared
    for dataset, split, role in ((data.train, "train", "TRAIN"), (data.control, "val", "CONTROL256")):
        dataset._entry_observed_market_binding = {
            "identity": {"training_phase": "entry_only", "target_sha256": "a" * 64},
            "role": role, "physical_source": data.case["sources"][split],
        }
    if fault == "missing_control": del data.control._entry_observed_market_binding
    elif fault == "different_target": data.control._entry_observed_market_binding["identity"]["target_sha256"] = "b" * 64
    elif fault == "wrong_source": data.control._entry_observed_market_binding["physical_source"] = data.case["sources"]["train"]
    if fault:
        with pytest.raises(RuntimeError, match="OBSERVED_ENTRY_BINDING_MISMATCH"):
            trainer._prefix_candidate_training_binding(**_binding_args(data))
    else:
        binding, _, _ = trainer._prefix_candidate_training_binding(**_binding_args(data))
        assert binding["entry_observed_market"]["train"] == data.train._entry_observed_market_binding
        assert binding["entry_observed_market"]["control"] == data.control._entry_observed_market_binding


@pytest.mark.parametrize("bound", [True, False])
def test_initial_scope_requires_explicit_new_target_policy(initial_scope, monkeypatch, bound):
    policy, recipe, _, seal = initial_scope
    value = {"path": "/synthetic/observed-entry.json", "sha256": "a" * 64}
    recipe["entry_observed_market"] = value
    monkeypatch.setattr(observed, "require_entry_observed_market_scope", lambda _: {"binding": value})
    if bound: policy["chronological_initial_measurement"]["entry_observed_market"] = value
    seal()
    if bound:
        assert native.require_native_run_scope(recipe, invocation_number=1) == 0
    else:
        with pytest.raises(RuntimeError, match="NOT_AUTHORIZED"):
            native.require_native_run_scope(recipe, invocation_number=1)


def test_old_exit_derived_baseline_cannot_admit_new_entry_training(learning_scope, monkeypatch):
    _, recipe, _, _ = learning_scope
    recipe["entry_observed_market"] = {"path": "/synthetic/design.json", "sha256": "a" * 64}
    monkeypatch.setattr(observed, "require_entry_observed_market_scope", lambda _: {})
    monkeypatch.setattr(observed, "entry_observed_market_identity", lambda _: {"teacher": False})
    with pytest.raises(RuntimeError, match="ENTRY_TARGET_IDENTITY_MISMATCH"):
        native.require_chronological_learning_measurement(recipe)


def test_resume_preserves_bound_target_and_rejects_a_silent_change(physical_prepared, tmp_path):
    data = physical_prepared
    for dataset, split, role in ((data.train, "train", "TRAIN"), (data.control, "val", "CONTROL256")):
        dataset._entry_observed_market_binding = {
            "identity": {"training_phase": "entry_only", "target_sha256": "a" * 64},
            "role": role, "physical_source": data.case["sources"][split],
        }
    harness = PrefixHarness(tmp_path / "observed-resume", data)
    direct, split = harness.root / "DIRECT", harness.root / "SPLIT"
    harness.run_prefix(direct, 4)
    expected = harness.state(direct)
    harness.run_prefix(split, 2)
    harness.run_prefix(split, 4, expected_pointer=digest(harness.pointer(split)))
    actual = harness.state(split)
    for key in ("model_state", "target_model_state", "optimizer_state", "weight_ema_state",
                "lr_scheduler_state", "rng_state", "epoch_order", "training_progress"):
        equal_tree(expected[key], actual[key])
    pointer_before = harness.pointer(split).read_bytes()
    batches_before = list(harness.batches)
    for dataset in (data.train, data.control):
        dataset._entry_observed_market_binding["identity"]["target_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="CONTRACT"):
        harness.run_prefix(split, 5, expected_pointer=digest(harness.pointer(split)))
    assert harness.pointer(split).read_bytes() == pointer_before
    assert harness.batches == batches_before


def test_entry_measurement_is_invariant_to_exit_target_values(tmp_path, monkeypatch):
    from gx1.scripts import run_unified_exit_random_access_val_v1 as val
    from tests.test_chronological_control_targets import _context

    args = _context(tmp_path, physical=True)
    original = args["dataset"]
    class Rows(torch.utils.data.Dataset):
        _entry_observed_market_binding = {
            "identity": {"target_contract": observed.entry_observed_market_contract()},
            "role": "CONTROL256",
        }
        def __len__(self):
            return len(original)
        def __getitem__(self, row):
            return {**original[row], "entry_observed_market_target_bps":
                    torch.tensor([float(row % 13) - 4., -6., 0.], dtype=torch.float32)}
    args["dataset"] = Rows()
    exit_value = [1.]
    def forward(model, seq, snap, **kwargs):
        return {val.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY: seq,
                "entry_action_q_bps": seq.expand(-1, 3).contiguous()}
    def reference(**kwargs):
        children = kwargs["child_rows"]
        target = torch.full((len(children), 2), exit_value[0])
        return ([{"entry_row_index": row, "state_index": 0, "target_hold_bps": target[i].tolist()}
                 for i, row in enumerate(children)], {"hold_target_bps": target})
    def forbidden(**kwargs):
        raise AssertionError("An Exit anchor must never construct an observed Entry target")
    monkeypatch.setattr(val, "_model_forward_fp32", forward)
    monkeypatch.setattr(val, "_multi_tf_kwargs_from_batch", lambda *a: {})
    monkeypatch.setattr(val, "_bounded_reference_exit_observations", reference)
    monkeypatch.setattr(val, "_candidate_anchor_targets", forbidden)
    monkeypatch.setattr(val, "_new_active_head_epoch_accumulator", lambda: {})
    monkeypatch.setattr(val, "_accumulate_active_head_epoch", lambda *a: None)
    monkeypatch.setattr(val, "_active_head_epoch_diagnostics", lambda *a: ({}, None))
    monkeypatch.setattr(val, "accumulate_route_diagnostics_v1", lambda *a: None)
    monkeypatch.setattr(val, "finalize_route_diagnostics_v1", lambda *a: {})
    _, before, _ = val._entry_representations(**args)
    exit_value[0] = 1000.
    _, after, _ = val._entry_representations(**args)
    assert before["bounded_entry_observations"] == after["bounded_entry_observations"]
    assert before["bounded_exit_anchor_observations"] != after["bounded_exit_anchor_observations"]
    evidence = after["candidate_active_head_evidence"]
    assert evidence["exit_model_used_for_entry_targets"] is False
    assert evidence["entry_q_target_semantics"] == observed.entry_observed_market_contract()["target"]
    for row in after["bounded_entry_observations"]:
        assert row["target_q_bps"] == [float(row["parent_entry_row_index"] % 13) - 4., -6., 0.]


def test_existing_transformer_entry_gradient_never_reaches_exit_specific_weights():
    from tests.test_entry_v10_ctx_model_shapes import _make_model, _make_inputs

    torch.manual_seed(910)
    model = _make_model(dropout=0.)
    seq, snap, cats, cont, mtf = _make_inputs()
    output = model(seq, snap, ctx_cat=cats, ctx_cont=cont, liquidation_relative_values=True, **mtf)
    target = torch.tensor([[4., -6., 0.], [-8., 6., 0.]])
    torch.nn.functional.mse_loss(output["entry_action_q_bps"], target).backward()
    observed.require_entry_only_gradients(model)
    assert any(parameter.grad is not None and bool((parameter.grad != 0).any())
               for name, parameter in model.named_parameters() if name.startswith("head_entry_action_q."))
    assert any(parameter.grad is not None and bool((parameter.grad != 0).any())
               for name, parameter in model.named_parameters() if name.startswith("family_tf_context_gate."))
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if observed.is_exit_owned_parameter(name))



def test_path_primary_population_survives_unavailable_h20_and_long_support_is_independent(tmp_path, economics):
    from gx1.contracts import entry_observed_market_horizons_v1 as hz
    from gx1.contracts.entry_causal_m1_outcomes_v1 import M5_PATH_TARGET_COLUMNS, M5_PATH_VALID_COLUMN
    from gx1.contracts.entry_model_native_aux_targets_v3 import SUPPORT_FORECAST_TARGET_BATCH_KEY, SUPPORT_FORECAST_VALID_BATCH_KEY
    dataset, scope_value = dataset_scope(tmp_path, economics)
    frame = pd.read_parquet(dataset.parquet_path)
    for side, mfe in (('long', 5.), ('short', -1.)):
        prefix = 'y_' + side + '_m5_path_'
        frame[prefix + 'mfe_bps'] = mfe
        frame[prefix + 'mae_before_first_mfe_lower_bps'] = 1.
        frame[prefix + 'mae_before_first_mfe_upper_bps'] = 2.
        frame[prefix + 'path_utility_bps'] = mfe - 2.
        frame[prefix + 'first_mfe_m1_offset'] = 0.
    frame[M5_PATH_VALID_COLUMN] = True
    for target, valid_column in zip(MODEL_NATIVE_FORECAST_TARGET_COLUMNS, MODEL_NATIVE_FORECAST_VALID_COLUMNS):
        frame[target] = 0.0 if target == MODEL_NATIVE_FORECAST_TARGET_COLUMNS[-1] else 1.0
        frame[valid_column] = target != MODEL_NATIVE_FORECAST_TARGET_COLUMNS[-1]
    dataset.aux_head_target_contract = model_native_aux_target_contract_metadata(
        emission_scope=M5_PRIMARY_OBSERVED_EMISSION_SCOPE)
    for h in hz.OBSERVATION_HORIZONS_TRADING_DAYS:
        for name in (*hz.GROSS_TARGET_COLUMNS_BY_HORIZON[h], hz.ELAPSED_COLUMN_BY_HORIZON[h]):
            frame[name] = np.nan
    for h, rows in ((1, [0, 1]), (5, [0])):
        frame.loc[rows, hz.GROSS_TARGET_COLUMNS_BY_HORIZON[h][0]] = 10.
        frame.loc[rows, hz.GROSS_TARGET_COLUMNS_BY_HORIZON[h][1]] = -12.
        frame.loc[rows, hz.ELAPSED_COLUMN_BY_HORIZON[h]] = h * 86400
    frame.to_parquet(dataset.parquet_path, index=False)
    scope_value['physical_sources']['train']['parquet'] = _bind(dataset.parquet_path)
    dataset._policy_dependent_auxiliary_binding['parent_entry_parquet'] = scope_value['physical_sources']['train']['parquet']
    scope_value['training_phase'] = observed.ENTRY_ONLY_M5_PATH_TRAINING_PHASE
    scope_value['target_contract'] = observed.entry_observed_m5_path_contract()
    observed.bind_entry_observed_market_dataset(dataset, scope_value)
    assert dataset._entry_observed_market_binding['admitted_rows'] == 3
    assert dataset._entry_observed_market_admitted_rows.tolist() == [True, True, False, True]
    assert dataset._entry_observed_market_binding['support_forecast_observed_cells'] == 6
    batch = hz.observed_entry_batch_items(dataset, 3)
    assert observed.M5_PATH_BATCH_TARGET_KEY in batch
    assert batch[M5_FORECAST_VALID_BATCH_KEY].tolist() == [True, True, True, False]
    assert not bool(batch[SUPPORT_FORECAST_VALID_BATCH_KEY].any())
    assert torch.equal(batch[SUPPORT_FORECAST_TARGET_BATCH_KEY], torch.zeros(6))
    batch = {key: value.unsqueeze(0) for key, value in batch.items()}
    target, valid = hz.observed_entry_batch_targets(batch, dataset._entry_observed_market_binding, device=torch.device('cpu'))
    assert target.tolist() == [[3., -3., 0.]] and bool(valid.all())


def test_actual_forecast_loss_backpropagates_only_observed_long_support_and_skips_legacy_heads(monkeypatch):
    from gx1.contracts.entry_model_native_aux_targets_v3 import SUPPORT_FORECAST_TARGET_BATCH_KEY, SUPPORT_FORECAST_VALID_BATCH_KEY
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import FORECAST_HEAD_DIM, FORECAST_HORIZONS
    prediction = torch.zeros(2, FORECAST_HEAD_DIM, requires_grad=True)
    batch = {f'y_forecast_ret_K{h}': torch.ones(2) for h in FORECAST_HORIZONS}
    batch[observed.M5_PATH_BATCH_TARGET_KEY] = torch.tensor([[3., -3., 0.], [3., -3., 0.]])
    batch[M5_FORECAST_VALID_BATCH_KEY] = torch.tensor([[True, True, True, False], [True, False, True, True]])
    for j, horizon in enumerate(FORECAST_HORIZONS):
        batch[f'y_forecast_ret_K{horizon}'][~batch[M5_FORECAST_VALID_BATCH_KEY][:, j]] = 0.0
    batch[SUPPORT_FORECAST_TARGET_BATCH_KEY] = torch.tensor([[2., -2., 0., 0., 0., 0.], [0., 0., 3., -3., 0., 0.]])
    batch[SUPPORT_FORECAST_VALID_BATCH_KEY] = batch[SUPPORT_FORECAST_TARGET_BATCH_KEY] != 0
    def require(out, batch, **kwargs):
        assert kwargs['output_name'] == 'forecast_pred'  # no old M5/MID risk head may be requested
        return out['forecast_pred']
    monkeypatch.setattr(trainer, '_require_active_aux_head_prediction', require)
    losses = trainer.dip_forecast_task_losses({'forecast_pred': prediction}, batch, torch.device('cpu'))
    assert set(losses) == {'forecast_return_bps'}
    losses['forecast_return_bps'].backward()
    assert torch.equal(prediction.grad[:, 4:] != 0, batch[SUPPORT_FORECAST_VALID_BATCH_KEY])
    assert torch.equal(prediction.grad[:, :4] != 0, batch[M5_FORECAST_VALID_BATCH_KEY])


def test_path_phase_inactive_auxiliary_gradients_are_rejected():
    model = SmallEntry()
    model.head_dip = nn.Linear(3, 2)
    observed.require_entry_only_gradients(model, entry_training_phase=observed.ENTRY_ONLY_M5_PATH_TRAINING_PHASE)
    model.head_dip.weight.grad = torch.zeros_like(model.head_dip.weight)
    with pytest.raises(RuntimeError, match='EXIT_GRADIENT_FORBIDDEN'):
        observed.require_entry_only_gradients(model, entry_training_phase=observed.ENTRY_ONLY_M5_PATH_TRAINING_PHASE)


def test_path_train_epoch_preserves_all_inactive_head_weights_and_ema(monkeypatch):
    torch.manual_seed(910)
    model = SmallEntry()
    for name in ('head_dip', 'head_timing', 'head_tail_risk', 'head_vol_forecast',
                 'head_side_mae', 'head_trendline_event', 'head_position_size'):
        setattr(model, name, nn.Linear(3, 2))
    loader = training_fixture(monkeypatch)
    loader.dataset._entry_observed_market_binding['identity']['training_phase'] = observed.ENTRY_ONLY_M5_PATH_TRAINING_PHASE
    class PathLoader:
        dataset = loader.dataset
        def __len__(self):
            return len(loader)
        def __iter__(self):
            for original in loader:
                batch = dict(original)
                batch[observed.M5_PATH_BATCH_TARGET_KEY] = batch.pop('entry_observed_market_target_bps')
                batch[M5_FORECAST_VALID_BATCH_KEY] = torch.ones(len(batch[observed.M5_PATH_BATCH_TARGET_KEY]), len(MODEL_NATIVE_FORECAST_TARGET_COLUMNS), dtype=torch.bool)
                batch['entry_observed_market_support_forecast_bps'] = torch.zeros(2, 6)
                batch['entry_observed_market_support_forecast_valid'] = torch.zeros(2, 6, dtype=torch.bool)
                yield batch
    def forbidden(*args, **kwargs):
        raise AssertionError('Legacy M5/MID dip, timing, risk, size and MAE targets must be inactive')
    monkeypatch.setattr(trainer, '_side_mae_auxiliary_loss', forbidden)
    monkeypatch.setattr(trainer, '_trendline_event_aux_loss', forbidden)
    monkeypatch.setattr(trainer, '_require_active_aux_head_prediction', forbidden)
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.5)
    ema = trainer._WeightEma(model, .5, entry_training_phase=observed.ENTRY_ONLY_M5_PATH_TRAINING_PHASE)
    supervised, gradients = {}, {}
    loss, stats, complete = trainer.train_epoch(model, ForbiddenTeacher().eval(), PathLoader(), optimizer,
        torch.device('cpu'), 1, supervised, gradients, weight_ema=ema)
    assert complete and np.isfinite(loss) and ema.steps == 2
    assert set(supervised) == {'entry_action_q', 'forecast_return_bps'}
    assert not torch.equal(before['shared_encoder.weight'], model.shared_encoder.weight)
    assert stats['exit_training_active'] is False and 'side_mae_loss_mean' not in stats
    for name, parameter in model.named_parameters():
        if observed.is_inactive_phase_parameter(name, observed.ENTRY_ONLY_M5_PATH_TRAINING_PHASE):
            assert parameter.grad is None and parameter not in optimizer.state
            assert torch.equal(parameter, before[name])
            assert torch.equal(ema.state_dict_clone()[name], before[name])
    # Joint training continues to average Exit-owned parameters normally.
    assert not observed.is_inactive_phase_parameter('head_exit_action.weight', 'joint_fitted_q')


def test_path_epoch_diagnostics_do_not_read_legacy_labels_or_claim_missing_support():
    import json
    from gx1.contracts.entry_model_native_aux_targets_v3 import SUPPORT_FORECAST_TARGET_BATCH_KEY, SUPPORT_FORECAST_VALID_BATCH_KEY
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import FORECAST_HORIZONS, FORECAST_HEAD_DIM
    count = 4
    index = torch.arange(count, dtype=torch.float32)
    target = torch.stack((index, -index, torch.zeros_like(index)), dim=1)
    batch = {observed.M5_PATH_BATCH_TARGET_KEY: target, 'entry_row_index': torch.arange(count),
        M5_FORECAST_VALID_BATCH_KEY: torch.ones(count, len(MODEL_NATIVE_FORECAST_TARGET_COLUMNS), dtype=torch.bool),
        SUPPORT_FORECAST_TARGET_BATCH_KEY: torch.zeros(count, 6),
        SUPPORT_FORECAST_VALID_BATCH_KEY: torch.zeros(count, 6, dtype=torch.bool)}
    batch.update({f'y_forecast_ret_K{h}': index + j for j, h in enumerate(FORECAST_HORIZONS)})
    forecast = index[:, None].expand(-1, FORECAST_HEAD_DIM).clone()
    out = {'entry_action_q_bps': target + .1, '_entry_action_q_target': target,
        '_entry_action_q_valid': torch.ones_like(target, dtype=torch.bool), 'forecast_pred': forecast}
    accumulator = trainer._new_active_head_epoch_accumulator()
    counts = trainer._accumulate_active_head_epoch(accumulator, SmallEntry(), out, batch, torch.device('cpu'))
    assert counts['dip'] == 0 and counts['side_mae'] == 0 and counts['forecast'] == count * 4
    metrics, failures = trainer._active_head_epoch_diagnostics(accumulator, minimum_supervised_rows=2)
    assert not failures
    assert metrics['active_head_contract'] == ['entry_action_q', 'forecast']
    assert metrics['active_head_diagnostics']['dip']['phase_inactive'] is True
    forecast_metrics = metrics['active_head_diagnostics']['forecast']['components']['forecast_pred']
    assert forecast_metrics['unavailable_target_columns'] == list(range(4, 10))
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize("fault", ["missing", "nonzero_masked", "no_observations"])
def test_m5_forecast_mask_errors_never_create_observations_or_fake_loss(monkeypatch, fault):
    from gx1.contracts.entry_model_native_aux_targets_v3 import (
        SUPPORT_FORECAST_TARGET_BATCH_KEY, SUPPORT_FORECAST_VALID_BATCH_KEY,
        MODEL_NATIVE_FORECAST_OUTPUT_DIM,
    )
    batch = {name: torch.zeros(2) for name in MODEL_NATIVE_FORECAST_TARGET_COLUMNS}
    batch[observed.M5_PATH_BATCH_TARGET_KEY] = torch.zeros(2, 3)
    batch[SUPPORT_FORECAST_TARGET_BATCH_KEY] = torch.zeros(2, 6)
    batch[SUPPORT_FORECAST_VALID_BATCH_KEY] = torch.zeros(2, 6, dtype=torch.bool)
    if fault != "missing":
        batch[M5_FORECAST_VALID_BATCH_KEY] = torch.zeros(2, len(MODEL_NATIVE_FORECAST_TARGET_COLUMNS), dtype=torch.bool)
    if fault == "nonzero_masked":
        batch[MODEL_NATIVE_FORECAST_TARGET_COLUMNS[0]][0] = 1.0
    prediction = torch.zeros(2, MODEL_NATIVE_FORECAST_OUTPUT_DIM, requires_grad=True)
    monkeypatch.setattr(trainer, '_require_active_aux_head_prediction', lambda out, batch, **kwargs: out['forecast_pred'])
    marker = 'NO_OBSERVED_CELLS' if fault == 'no_observations' else 'M5_FORECAST_BATCH_INVALID'
    with pytest.raises(RuntimeError, match=marker):
        trainer.dip_forecast_task_losses({'forecast_pred': prediction}, batch, torch.device('cpu'))
    assert prediction.grad is None


@pytest.fixture
def m5_physical_receipt(tmp_path, monkeypatch):
    from gx1.contracts import entry_model_native_aux_targets_v3 as targets
    from gx1.contracts import entry_causal_m1_outcomes_v1 as path
    from gx1.contracts import entry_observed_market_horizons_v1 as hz
    from gx1.features import gc_native_context_v1 as gc
    from gx1.time.session_detector import m5_decision_availability

    def prepare(fault=None):
        train_times = pd.date_range("2025-05-01T00:00Z", periods=2, freq="5min")
        registry = {"train_decision_rows": 2,
            "train_decision_clock_sha256": hashlib.sha256(m5_decision_availability(train_times).asi8.astype("<i8").tobytes()).hexdigest()}
        if fault == "registry_clock":
            registry["train_decision_rows"] = 1
        registry_binding = _write(tmp_path / "registry.json", registry)
        monkeypatch.setattr(gc, "REGISTRY_PATH", registry_binding["path"])
        monkeypatch.setattr(gc, "REGISTRY_SHA256", registry_binding["sha256"])
        sources, manifests, frames = {}, {}, {}
        for split, first, low, high in (
            ("train", "2025-05-01T00:00Z", "2025-05-01T00:00:00Z", "2025-05-03T00:00:00Z"),
            ("val", "2025-05-03T00:00Z", "2025-05-03T00:00:00Z", "2025-05-05T00:00:00Z"),
        ):
            frame = pd.DataFrame({name: np.zeros(2, np.float32) for name in trainer._MODEL_NATIVE_ACTIVE_TARGET_COLS})
            frame["time"] = pd.date_range(first, periods=2, freq="5min")
            for side, mfe in (("long", -1.), ("short", 5.)):
                prefix = "y_" + side + "_m5_path_"
                for suffix, value in (("mfe_bps", mfe), ("mae_before_first_mfe_lower_bps", 1.),
                        ("mae_before_first_mfe_upper_bps", 2.), ("path_utility_bps", mfe - 2.), ("first_mfe_m1_offset", 0.)):
                    frame[prefix + suffix] = value
            frame[path.M5_PATH_VALID_COLUMN] = True
            for j, (target, valid) in enumerate(zip(targets.MODEL_NATIVE_FORECAST_TARGET_COLUMNS, targets.MODEL_NATIVE_FORECAST_VALID_COLUMNS)):
                frame[target] = 0.0 if j == len(targets.MODEL_NATIVE_FORECAST_TARGET_COLUMNS) - 1 else 1.0
                frame[valid] = j != len(targets.MODEL_NATIVE_FORECAST_TARGET_COLUMNS) - 1
            for name in hz.HORIZON_LABEL_COLUMNS:
                frame[name] = np.nan
            # An actually observed zero return remains a valid support target.
            for name in hz.GROSS_TARGET_COLUMNS_BY_HORIZON[1]:
                frame[name] = 0.0
            frame[hz.ELAPSED_COLUMN_BY_HORIZON[1]] = 86400.0
            frame[targets.LEGACY_CAUSAL_M1_OBSERVED_COLUMN] = False
            frame[targets.LEGACY_AUX_COMPLETE_COLUMN] = False
            if split == "train":
                if fault == "nonzero_masked": frame.loc[0, targets.MODEL_NATIVE_FORECAST_TARGET_COLUMNS[-1]] = 7.0
                if fault == "observed_nonfinite": frame.loc[0, targets.MODEL_NATIVE_FORECAST_TARGET_COLUMNS[0]] = np.nan
                if fault == "primary_false": frame.loc[0, path.M5_PATH_VALID_COLUMN] = False
                if fault == "primary_utility": frame.loc[0, "y_long_m5_path_path_utility_bps"] = 9.0
                if fault == "partial_support": frame.loc[0, hz.GROSS_TARGET_COLUMNS_BY_HORIZON[20][0]] = 0.0
                if fault == "zero_elapsed": frame.loc[0, hz.ELAPSED_COLUMN_BY_HORIZON[1]] = 0.0
                if fault == "missing_mask": frame = frame.drop(columns=[targets.MODEL_NATIVE_FORECAST_VALID_COLUMNS[-1]])
            parquet = tmp_path / (split + ".parquet")
            frame.to_parquet(parquet, index=False)
            end = (pd.Timestamp(high) - pd.Timedelta(minutes=5)).isoformat()
            window = {"start": pd.Timestamp(low).isoformat(), "end": end}
            source = {"parquet": _bind(parquet), "physical_rows": len(frame), "declared_window": window,
                "clock_sha256": hashlib.sha256(frame.time.array.asi8.astype("<i8").tobytes()).hexdigest()}
            sources[split] = source
            frames[split] = frame
            manifests[split] = {"extra": {"rows": len(frame), "emission_scope": targets.M5_PRIMARY_OBSERVED_EMISSION_SCOPE,
                "gc_native_context": gc.contract(), "diagnostic_outcome_policy_sha256": "a" * 64,
                "m5_path_supervision": {"contract": observed.entry_observed_m5_path_contract(), "emitted_valid_rows": len(frame), "target_m1_source_sha256": "c" * 64},
                "aux_head_target_contract": {**targets.model_native_aux_target_contract_metadata(emission_scope=targets.M5_PRIMARY_OBSERVED_EMISSION_SCOPE),
                    "candidate_rows_before_primary_availability": len(frame), "primary_unavailable_candidate_rows_excluded": 0,
                    "primary_valid_rows_emitted": len(frame), "legacy_causal_m1_observed_rows_emitted": 0,
                    "legacy_aux_complete_rows_emitted": 0,
                    "forecast_observed_rows_by_column": dict(zip(targets.MODEL_NATIVE_FORECAST_TARGET_COLUMNS, (2, 2, 2, 0)))}}}
        windows = {key: source["declared_window"] for key, source in sources.items()}
        for split, manifest in manifests.items():
            manifest["splits"] = windows
            sources[split]["manifest"] = _write(tmp_path / (split + ".manifest.json"), manifest)
        row_bindings = {}
        for key, rows in (("TRAIN_CALENDAR_PARENT_ROWS", [0, 1]), ("CONTROL256_PARENT_ROWS", [0])):
            row_path = tmp_path / (key + ".npy")
            np.save(row_path, np.array(rows, dtype=np.int64))
            row_bindings[key] = _bind(row_path)
        design = {"schema_version": "gx1_frozen_chronological_learning_design_v1",
            "scope": {"entry_only_training_phase": observed.ENTRY_ONLY_M5_PATH_TRAINING_PHASE},
            "calendar": {"source_bindings": sources, "physical_source_splits": {"train": "train", "control": "val"},
                "physical_coordinate_namespaces_are_separate": True, "train_entry_start_inclusive": "2025-05-01T00:00:00Z",
                "train_control_cutoff": "2025-05-03T00:00:00Z", "development_control_entry_end_exclusive": "2025-05-05T00:00:00Z",
                "bindings": row_bindings, "control256": {"rows": 1}},
            "selection": {"control_entries": 1}, "budget": {"later_control_entries": 1}}
        design_binding = _write(tmp_path / "design.json", design)
        receipt = {"schema_version": targets.M5_PHYSICAL_TARGET_BINDING_SCHEMA, "design": design_binding,
            "registry": registry_binding, "training_phase": observed.ENTRY_ONLY_M5_PATH_TRAINING_PHASE,
            "emission_scope": targets.M5_PRIMARY_OBSERVED_EMISSION_SCOPE, "test_accessed": False, "forbidden_access_attempts": []}
        if fault == "phase": receipt["training_phase"] = "joint_fitted_q"
        if fault == "test_access": receipt["test_accessed"] = True
        if fault == "forbidden_access": receipt["forbidden_access_attempts"] = ["test"]
        receipt_binding = _write(tmp_path / "receipt.json", receipt)
        if fault == "changed_bytes":
            target = Path(sources["train"]["parquet"]["path"])
            target.write_bytes(target.read_bytes() + b"changed")
        return sources, frames, design_binding, receipt_binding
    return prepare


@pytest.mark.parametrize("role,split,mask", [("TRAIN", "train", [True, True]), ("CONTROL256", "val", [True, False])])
def test_actual_m5_physical_receipt_admits_primary_without_legacy_or_h20(m5_physical_receipt, role, split, mask):
    sources, frames, design, receipt = m5_physical_receipt()
    admitted, proof = trainer._physical_auxiliary_target_binding(frames[split],
        parquet_path=Path(sources[split]["parquet"]["path"]), result_binding=receipt,
        expected_design_sha256=design["sha256"], role=role)
    assert admitted.tolist() == mask
    assert proof["forecast_observed_cells"] == len(frames[split]) * (3 + 2)
    assert proof["long_support_observed_cells_by_column"] == [2, 2, 0, 0, 0, 0]
    assert proof["whole_train_registry_clock"]["exact_match"] is True
    assert proof["targets_rewritten"] is False


@pytest.mark.parametrize("fault", ["nonzero_masked", "observed_nonfinite", "primary_false", "primary_utility",
    "partial_support", "zero_elapsed", "missing_mask", "phase", "test_access", "forbidden_access", "changed_bytes", "registry_clock"])
def test_actual_m5_physical_receipt_rejects_invalid_cells_scope_and_bytes(m5_physical_receipt, fault):
    sources, frames, design, receipt = m5_physical_receipt(fault)
    with pytest.raises(RuntimeError):
        trainer._physical_auxiliary_target_binding(frames["train"],
            parquet_path=Path(sources["train"]["parquet"]["path"]), result_binding=receipt,
            expected_design_sha256=design["sha256"], role="TRAIN")


def test_new_m5_receipt_reaches_existing_dataset_binding_dispatch(m5_physical_receipt):
    sources, frames, design, receipt = m5_physical_receipt()
    dataset = SimpleNamespace(df=frames["train"].copy(), parquet_path=Path(sources["train"]["parquet"]["path"]), _advanced=True,
        _policy_dependent_auxiliary_binding=None, _policy_dependent_auxiliary_bound_rows=None)
    trainer.EntryV10CtxDataset.bind_policy_dependent_auxiliary_targets(dataset,
        result_path=Path(receipt["path"]), expected_result_sha256=receipt["sha256"],
        expected_design_sha256=design["sha256"], role="TRAIN")
    assert dataset._policy_dependent_auxiliary_binding["training_phase"] == observed.ENTRY_ONLY_M5_PATH_TRAINING_PHASE
    assert dataset._policy_dependent_auxiliary_bound_rows.tolist() == [True, True]
    pd.testing.assert_frame_equal(dataset.df, frames["train"])


def test_actual_m5_receipt_rejects_target_frame_changed_after_parquet_load(m5_physical_receipt):
    sources, frames, design, receipt = m5_physical_receipt()
    changed = frames["train"].copy()
    changed.loc[0, MODEL_NATIVE_FORECAST_TARGET_COLUMNS[0]] += 1.0
    with pytest.raises(RuntimeError, match="M5_PHYSICAL_TARGET_VALUES_MISMATCH"):
        trainer._physical_auxiliary_target_binding(changed,
            parquet_path=Path(sources["train"]["parquet"]["path"]), result_binding=receipt,
            expected_design_sha256=design["sha256"], role="TRAIN")


@pytest.fixture
def dynamic_physical_receipt(tmp_path, monkeypatch):
    """Small actual pointer parquets, with no legacy labels or expanded paths."""
    from gx1.contracts import entry_causal_m1_outcomes_v1 as quote
    from gx1.contracts import entry_model_native_aux_targets_v3 as targets
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    from gx1.contracts import unified_exit_market_closure_authority_v1 as closure
    from gx1.features import gc_native_context_v1 as gc
    registry = _write(tmp_path / "dynamic_registry.json", {})
    monkeypatch.setattr(gc, "REGISTRY_PATH", registry["path"])
    monkeypatch.setattr(gc, "REGISTRY_SHA256", registry["sha256"])
    times = pd.DatetimeIndex(list(pd.date_range("2025-05-01T00:05Z", periods=11, freq="min"))
        + list(pd.date_range("2025-05-03T00:05Z", periods=11, freq="min")))
    bid = 100. + np.arange(len(times), dtype=np.float64) / 100
    raw = pd.DataFrame({"time": times, "bid_open": bid, "ask_open": bid + .1,
        "bid_high": bid + .2, "ask_high": bid + .3,
        "bid_low": bid - .2, "ask_low": bid - .1})
    quote_path = tmp_path / "original_quotes.parquet"
    raw.to_parquet(quote_path, index=False)
    quote_binding = {**_bind(quote_path), "size_bytes": quote_path.stat().st_size,
        "rows": len(raw), "clock_sha256": closure.m1_clock_sha256(times)}

    def prepare(fault=None):
        sources, proofs = {}, {}
        for split, first, high in (("train", "2025-05-01T00:00Z", "2025-05-03T00:00Z"),
                                   ("val", "2025-05-03T00:00Z", "2025-05-05T00:00Z")):
            decisions = pd.date_range(first, periods=3, freq="5min")
            index = quote.prepare_dynamic_m1_prefix_index(raw, split_end_exclusive=high,
                source_parquet_sha256=quote_binding["sha256"])
            fill = quote.build_dynamic_m1_fill_surface(index, m5_decision_times=decisions)
            terminal = quote.query_dynamic_m1_prefix(index, fill_surface=fill,
                query_entry_rows=np.arange(3), horizon_minutes=np.ones(3, np.int64))
            path = quote.query_dynamic_m1_prefix(index, fill_surface=fill,
                query_entry_rows=np.arange(3), horizon_minutes=np.ones(3, np.int64),
                endpoint_kind="closed_m1_prefix")
            frame = pd.DataFrame({"time": decisions,
                "dynamic_entry_m1_row": fill.entry_m1_row.to_numpy(np.int64),
                "dynamic_first_prefix_observed": True,
                "dynamic_first_terminal_observed_long": terminal["terminal_gross_observed"][:, 0],
                "dynamic_first_terminal_observed_short": terminal["terminal_gross_observed"][:, 1],
                "dynamic_first_path_observed_long": path["path_observed"][:, 0],
                "dynamic_first_path_observed_short": path["path_observed"][:, 1],
                "dynamic_first_terminal_censor_reason": terminal["terminal_censor_reason"],
                "dynamic_first_path_censor_reason": path["path_censor_reason"],
            })
            if split == "train":
                if fault == "false_population": frame.loc[0, "dynamic_first_prefix_observed"] = False
                if fault == "empty_component_union":
                    frame.loc[0, list(targets.DYNAMIC_M5_POINTER_COLUMNS[2:6])] = False
                if fault == "integer_mask":
                    frame["dynamic_first_prefix_observed"] = np.ones(3, dtype=np.int64)
                if fault == "bad_pointer": frame.loc[0, "dynamic_entry_m1_row"] = len(raw)
                if fault == "bad_reason": frame.loc[0, "dynamic_first_path_censor_reason"] = "unknown_source_absence"
                if fault == "bad_clock": frame.loc[1, "time"] = frame.time.iloc[0]
                if fault == "missing_column": frame = frame.drop(columns=[targets.DYNAMIC_M5_POINTER_COLUMNS[-1]])
            parquet_path = tmp_path / ("dynamic_" + split + ".parquet")
            frame.to_parquet(parquet_path, index=False)
            meta = {"schema_version": "gx1_dynamic_m5_physical_target_pointers_v1",
                "contract": quote.dynamic_m1_prefix_contract(), "source_quote_binding": quote_binding,
                "source_quote_columns": list(targets.DYNAMIC_M5_QUOTE_COLUMNS),
                "index_binding": index.binding(), "pointer_columns": list(targets.DYNAMIC_M5_POINTER_COLUMNS),
                "pointer_row_clock": "global_row_in_bound_shared_m1_quote_parquet",
                "first_prefix_horizon_minutes": 1, "first_terminal_endpoint_kind": "terminal_open_inclusive",
                "first_path_endpoint_kind": "closed_m1_prefix",
                "physical_population_rule": "any_genuine_first_prefix_terminal_or_path_component",
                "later_horizon_availability_is_not_an_emission_gate": True,
                "closure_mode": "unknown_gaps_only_no_inferred_market_calendar", "target_only_columns": True,
                "feature_inputs_include_future_masks": False, "dense_future_paths_materialized": False}
            if split == "train":
                if fault == "stale_cutoff": meta["index_binding"]["split_end_exclusive"] = "2025-05-04T00:00:00+00:00"
                if fault == "fake_row_offset": meta["index_binding"]["source_row_offset"] = 1
                if fault == "future_masks_as_features": meta["feature_inputs_include_future_masks"] = True
            emitted = {**targets.model_native_aux_target_contract_metadata(emission_scope=targets.M5_DYNAMIC_OBSERVED_EMISSION_SCOPE),
                "candidate_rows_before_primary_availability": 3, "primary_unavailable_candidate_rows_excluded": 0,
                "primary_valid_rows_emitted": 3, "legacy_causal_m1_observed_rows_emitted": 0,
                "legacy_aux_complete_rows_emitted": 0,
                "forecast_observed_rows_by_column": dict.fromkeys(targets.MODEL_NATIVE_FORECAST_TARGET_COLUMNS, 0)}
            manifest = {"extra": {"rows": 3, "emission_scope": targets.M5_DYNAMIC_OBSERVED_EMISSION_SCOPE,
                "aux_head_target_contract": emitted, "dynamic_m5_supervision": meta,
                "gc_native_context": gc.contract()}}
            source = {"parquet": _bind(parquet_path),
                "manifest": _write(tmp_path / ("dynamic_" + split + ".manifest.json"), manifest),
                "physical_rows": 3,
                "clock_sha256": hashlib.sha256(np.asarray(decisions.asi8, dtype="<i8").tobytes()).hexdigest(),
                "declared_window": {"start": pd.Timestamp(first).isoformat(),
                                    "end": (pd.Timestamp(high) - pd.Timedelta(minutes=5)).isoformat()}}
            sources[split] = source
            proofs[split] = {"source_parquet": source["parquet"], "source_manifest": source["manifest"],
                "rows": 3, "physical_clock_sha256": source["clock_sha256"],
                "dynamic_m5_supervision": meta, "first_prefix_observed_rows": 3,
                "first_terminal_observed_rows_by_side": {"long": 2, "short": 2},
                "first_path_observed_rows_by_side": {"long": 3, "short": 3},
                "first_terminal_censor_reason_counts": {"observed": 2, "terminal_quote_missing": 1},
                "first_path_censor_reason_counts": {"observed": 3},
                "entry_m1_pointer_sha256": hashlib.sha256(np.asarray(fill.entry_m1_row, dtype="<i8").tobytes()).hexdigest()}
        design = _write(tmp_path / "dynamic_design.json", {
            "schema_version": "gx1_frozen_chronological_learning_design_v1",
            "scope": {"entry_only_training_phase": ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE},
            "calendar": {"source_bindings": sources,
                "train_control_cutoff": "2025-05-03T00:00:00Z",
                "development_control_entry_end_exclusive": "2025-05-05T00:00:00Z"}})
        receipt = {"schema_version": targets.M5_DYNAMIC_PHYSICAL_TARGET_BINDING_SCHEMA,
            "design": design, "registry": registry,
            "training_phase": ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
            "emission_scope": targets.M5_DYNAMIC_OBSERVED_EMISSION_SCOPE,
            "target_contract": observed.entry_observed_dynamic_m5_path_contract(),
            "splits": proofs, "test_accessed": False, "forbidden_access_attempts": []}
        if fault == "wrong_phase": receipt["training_phase"] = "entry_only_m5_path"
        if fault == "legacy_scope": receipt["emission_scope"] = targets.M5_PRIMARY_OBSERVED_EMISSION_SCOPE
        if fault == "wrong_contract": receipt["target_contract"] = observed.entry_observed_m5_path_contract()
        if fault == "test_access": receipt["test_accessed"] = True
        if fault == "extra_test_split": receipt["splits"]["test"] = {}
        if fault == "wrong_count": receipt["splits"]["train"]["first_terminal_observed_rows_by_side"]["long"] = 3
        if fault == "wrong_pointer_hash": receipt["splits"]["train"]["entry_m1_pointer_sha256"] = "0" * 64
        if fault == "boolean_count": receipt["splits"]["train"]["first_prefix_observed_rows"] = True
        if fault == "changed_bytes":
            path = Path(sources["train"]["parquet"]["path"])
            path.write_bytes(path.read_bytes() + b"changed")
        return receipt, design
    return prepare


def test_dynamic_physical_receipt_preserves_independent_first_prefix_components(dynamic_physical_receipt):
    from gx1.contracts.entry_model_native_aux_targets_v3 import require_m5_dynamic_physical_target_receipt
    receipt, design = dynamic_physical_receipt()
    actual = require_m5_dynamic_physical_target_receipt(receipt, design_binding=design, context="TEST_DYNAMIC")
    assert actual == receipt
    assert actual["splits"]["train"]["first_terminal_observed_rows_by_side"] == {"long": 2, "short": 2}
    assert actual["splits"]["train"]["first_path_observed_rows_by_side"] == {"long": 3, "short": 3}
    # The genuine closed one-minute prefix survives a missing terminal quote;
    # neither a 95-minute label nor all 480 horizons is an admission gate.
    assert actual["splits"]["train"]["first_prefix_observed_rows"] == 3


@pytest.mark.parametrize("fault", ["false_population", "empty_component_union", "integer_mask", "bad_pointer",
    "bad_reason", "bad_clock", "missing_column", "stale_cutoff", "fake_row_offset", "future_masks_as_features",
    "wrong_phase", "legacy_scope", "wrong_contract", "test_access", "extra_test_split", "wrong_count",
    "wrong_pointer_hash", "boolean_count", "changed_bytes"])
def test_dynamic_physical_receipt_rejects_stale_or_false_physical_proof(dynamic_physical_receipt, fault):
    from gx1.contracts.entry_model_native_aux_targets_v3 import require_m5_dynamic_physical_target_receipt
    receipt, design = dynamic_physical_receipt(fault)
    with pytest.raises(RuntimeError):
        require_m5_dynamic_physical_target_receipt(receipt, design_binding=design, context="TEST_DYNAMIC")



def _dynamic_target_arrays(parent_rows):
    from gx1.contracts.entry_causal_m1_outcomes_v1 import DYNAMIC_M1_PREFIX_HORIZON_MINUTES
    count, query_count = len(parent_rows), len(DYNAMIC_M1_PREFIX_HORIZON_MINUTES)
    terminal = np.zeros((count, query_count, 3), np.float32)
    tv = np.zeros_like(terminal, bool)
    mfe = np.zeros((count, query_count, 2), np.float32)
    mae = np.zeros_like(mfe)
    pv = np.zeros_like(mfe, bool)
    closed_mfe, closed_mae, closed_valid = mfe.copy(), mae.copy(), pv.copy()
    for j, parent in enumerate(parent_rows):
        if int(parent) % 2 == 0:
            terminal[j, 0] = [4., -6., 0.]
            tv[j, 0] = True
            mfe[j, 0], mae[j, 0], pv[j, 0] = [5., -2.], [1., 1.], True
        else:
            closed_mfe[j, 0], closed_mae[j, 0], closed_valid[j, 0] = [-1., 3.], [2., 1.], True
    return {
        observed.DYNAMIC_M5_MARKOUT_TARGET_KEY: terminal,
        observed.DYNAMIC_M5_MARKOUT_VALID_KEY: tv,
        observed.DYNAMIC_M5_MFE_TARGET_KEY: mfe,
        observed.DYNAMIC_M5_MAE_TARGET_KEY: mae,
        observed.DYNAMIC_M5_PATH_VALID_KEY: pv,
        observed.DYNAMIC_M5_CLOSED_MFE_TARGET_KEY: closed_mfe,
        observed.DYNAMIC_M5_CLOSED_MAE_TARGET_KEY: closed_mae,
        observed.DYNAMIC_M5_CLOSED_PATH_VALID_KEY: closed_valid,
    }


class SmallDynamicEntry(SmallEntry):
    def __init__(self):
        super().__init__()
        from gx1.contracts.entry_model_native_joint_task_weighting_v1 import joint_task_names_for_phase
        from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
        self.entry_training_phase = ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
        self.head_dynamic_terminal_net_markout = nn.Linear(3, 2)
        self.head_dynamic_mfe = nn.Linear(3, 2)
        self.head_dynamic_pre_mfe_mae_upper = nn.Linear(3, 2)
        self.head_forecast = nn.Linear(3, trainer.FORECAST_HEAD_DIM)
        self.task_log_variances = nn.ParameterDict({key: nn.Parameter(torch.zeros(()))
            for key in joint_task_names_for_phase(self.entry_training_phase)})

    def forward(self, seq_x, snap_x, **kwargs):
        from gx1.contracts.entry_causal_m1_outcomes_v1 import DYNAMIC_M1_PREFIX_HORIZON_MINUTES
        from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
        assert not set(kwargs).intersection(_dynamic_target_arrays([0]))
        hidden = self.shared_encoder(seq_x)
        q = self.head_dynamic_terminal_net_markout(hidden)[:, None].expand(-1, len(DYNAMIC_M1_PREFIX_HORIZON_MINUTES), -1)
        mfe = self.head_dynamic_mfe(hidden)[:, None].expand_as(q)
        mae = torch.nn.functional.softplus(self.head_dynamic_pre_mfe_mae_upper(hidden))[:, None].expand_as(q)
        ratio = EntryV10CtxHybridTransformer._dynamic_quality_ratio_candidate(q, mae).amax(dim=1)
        return {
            "dynamic_terminal_net_markout_bps": torch.cat((q, torch.zeros_like(q[..., :1])), dim=-1),
            "dynamic_mfe_bps": mfe, "dynamic_pre_mfe_mae_upper_bps": mae,
            "dynamic_closed_mfe_bps": mfe, "dynamic_closed_pre_mfe_mae_upper_bps": mae,
            "forecast_pred": self.head_forecast(hidden),
            "dynamic_entry_action_quality_candidate": torch.cat((ratio, torch.zeros_like(ratio[..., :1])), dim=-1),
        }


def _dynamic_training_fixture(monkeypatch):
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    from gx1.contracts.entry_observed_market_horizons_v1 import (
        M5_FORECAST_VALID_BATCH_KEY, SUPPORT_FORECAST_TARGET_BATCH_KEY, SUPPORT_FORECAST_VALID_BATCH_KEY,
    )
    from gx1.contracts.entry_model_native_aux_targets_v3 import M5_DYNAMIC_EMISSION_BATCH_KEY
    old = training_fixture(monkeypatch)
    dataset = old.dataset
    dataset._entry_observed_market_binding = {"identity": {"training_phase": ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE}}
    dataset.df = pd.DataFrame({"dynamic_entry_m1_row": [100, 105]})
    class Provider:
        def __init__(self): self.calls = []
        def targets_for_rows(self, rows):
            self.calls.append(rows.copy())
            return _dynamic_target_arrays(rows)
    dataset._entry_dynamic_m5_target_provider = Provider()
    batch = next(iter(old))
    batch = {key: value for key, value in batch.items() if key in (
        "seq_x", "snap_x", "ctx_cont", "ctx_cat", "entry_row_index")}
    batch.update({M5_DYNAMIC_EMISSION_BATCH_KEY: torch.ones(2, dtype=torch.bool),
        "dynamic_entry_parent_row_index": torch.tensor([0, 1]), "dynamic_entry_m1_row": torch.tensor([100, 105]),
        M5_FORECAST_VALID_BATCH_KEY: torch.tensor([[True, False, False, False], [True, False, False, False]]),
        SUPPORT_FORECAST_TARGET_BATCH_KEY: torch.tensor([[3., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0.]]),
        SUPPORT_FORECAST_VALID_BATCH_KEY: torch.tensor([[True, False, False, False, False, False], [False]*6])})
    for j, key in enumerate(trainer._FORECAST_TARGET_COLS): batch[key] = torch.full((2,), 2. if j == 0 else 0.)
    class Loader:
        def __init__(self): self.dataset = dataset
        def __len__(self): return 2
        def __iter__(self): yield batch; yield batch
    def forbidden(*a, **kw): raise AssertionError("dynamic phase used legacy/Exit loss")
    for key in ("dip_forecast_task_losses", "_side_mae_auxiliary_loss", "_trendline_event_aux_loss"):
        monkeypatch.setattr(trainer, key, forbidden)
    return Loader(), batch


def test_dynamic_m5_real_train_and_curve_eval_preserve_exit_and_query_only_current_batches(monkeypatch):
    torch.manual_seed(9029)
    loader, _ = _dynamic_training_fixture(monkeypatch)
    model = SmallDynamicEntry()
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.5)
    ema = trainer._WeightEma(model, .5, entry_training_phase=model.entry_training_phase)
    supervised, gradients = {}, {}
    loss, stats, complete = trainer.train_epoch(model, ForbiddenTeacher(), loader, optimizer,
        torch.device("cpu"), 1, supervised, gradients, weight_ema=ema)
    assert complete and np.isfinite(loss) and ema.steps == len(loader)
    assert set(supervised) == set(model.task_log_variances) == set(gradients)
    assert all(supervised.values()) and all(gradients.values())
    assert stats["exit_training_active"] is False and stats["dynamic_candidate_native_authority_admitted"] is False
    assert "entry_action_q_raw_bps_mse_mean" not in stats
    assert stats["joint_task_supervised_cells_dynamic_terminal_net_markout_bps"] == 6
    assert stats["joint_task_supervised_cells_dynamic_mfe_bps"] == 8
    assert stats["joint_task_supervised_cells_dynamic_pre_mfe_mae_upper_bps"] == 8
    assert not torch.equal(model.shared_encoder.weight, before["shared_encoder.weight"])
    for name, parameter in model.named_parameters():
        if observed.is_inactive_phase_parameter(name, model.entry_training_phase):
            assert torch.equal(parameter, before[name]) and parameter not in optimizer.state
            assert torch.equal(ema.state_dict_clone()[name], before[name])
    result = trainer.validate(model, ForbiddenTeacher(), loader, torch.device("cpu"))
    assert np.isfinite(result[0])
    assert result[-1]["curve_evaluation_rows"] == 4
    assert result[-1]["realized_cash_or_native_policy_evaluated"] is False
    assert len(loader.dataset._entry_dynamic_m5_target_provider.calls) == 4
    assert all(np.array_equal(rows, [0, 1]) for rows in loader.dataset._entry_dynamic_m5_target_provider.calls)


def test_dynamic_m5_closed_only_tail_has_real_learning_without_terminal_or_inclusive_imputation(monkeypatch):
    loader, batch = _dynamic_training_fixture(monkeypatch)
    model = SmallDynamicEntry()
    prepared = trainer._materialize_dynamic_m5_batch_targets(batch, loader.dataset)
    for key in (observed.DYNAMIC_M5_MARKOUT_TARGET_KEY, observed.DYNAMIC_M5_MFE_TARGET_KEY,
                observed.DYNAMIC_M5_MAE_TARGET_KEY): prepared[key].zero_()
    prepared[observed.DYNAMIC_M5_MARKOUT_VALID_KEY].zero_()
    prepared[observed.DYNAMIC_M5_PATH_VALID_KEY].zero_()
    from gx1.contracts.entry_observed_market_horizons_v1 import (
        M5_FORECAST_VALID_BATCH_KEY, SUPPORT_FORECAST_TARGET_BATCH_KEY, SUPPORT_FORECAST_VALID_BATCH_KEY,
    )
    prepared[M5_FORECAST_VALID_BATCH_KEY].zero_()
    prepared[SUPPORT_FORECAST_VALID_BATCH_KEY].zero_(); prepared[SUPPORT_FORECAST_TARGET_BATCH_KEY].zero_()
    for key in trainer._FORECAST_TARGET_COLS: prepared[key].zero_()
    out = model(prepared["seq_x"], prepared["snap_x"])
    losses, cells = trainer._dynamic_m5_task_loss_surfaces(out, prepared, torch.device("cpu"))
    assert set(losses) == {"dynamic_mfe_bps", "dynamic_pre_mfe_mae_upper_bps"}
    assert cells["dynamic_terminal_net_markout_bps"] == cells["forecast_return_bps"] == 0
    total, _ = trainer._joint_task_loss(model, losses)
    total.backward()
    assert model.head_dynamic_mfe.weight.grad is not None
    assert model.head_dynamic_pre_mfe_mae_upper.weight.grad is not None
    assert model.head_dynamic_terminal_net_markout.weight.grad is None
    assert model.task_log_variances["dynamic_terminal_net_markout_bps"].grad is None
    assert model.task_log_variances["forecast_return_bps"].grad is None


@pytest.mark.parametrize("fault", ["false_nonzero", "observed_nan", "closed_negative_mae", "mask_dtype", "flat_nonzero", "pointer"])
def test_dynamic_m5_batch_loss_rejects_invalid_cells_and_global_pointer_drift(monkeypatch, fault):
    loader, batch = _dynamic_training_fixture(monkeypatch)
    if fault == "pointer":
        batch["dynamic_entry_m1_row"] = torch.tensor([101, 105])
        with pytest.raises(RuntimeError, match="POINTER_MISMATCH"):
            trainer._materialize_dynamic_m5_batch_targets(batch, loader.dataset)
        return
    prepared = trainer._materialize_dynamic_m5_batch_targets(batch, loader.dataset)
    if fault == "false_nonzero": prepared[observed.DYNAMIC_M5_MFE_TARGET_KEY][1, 1, 0] = 2.
    elif fault == "observed_nan": prepared[observed.DYNAMIC_M5_MARKOUT_TARGET_KEY][0, 0, 0] = float("nan")
    elif fault == "closed_negative_mae": prepared[observed.DYNAMIC_M5_CLOSED_MAE_TARGET_KEY][1, 0, 0] = -1.
    elif fault == "mask_dtype": prepared[observed.DYNAMIC_M5_CLOSED_PATH_VALID_KEY] = prepared[observed.DYNAMIC_M5_CLOSED_PATH_VALID_KEY].float()
    else: prepared[observed.DYNAMIC_M5_MARKOUT_TARGET_KEY][0, 0, 2] = 1.
    with pytest.raises(RuntimeError, match="ENTRY_DYNAMIC_M5"):
        trainer._dynamic_m5_task_loss_surfaces(SmallDynamicEntry()(prepared["seq_x"], prepared["snap_x"]), prepared, torch.device("cpu"))


def test_dynamic_m5_all_unobserved_tasks_fail_before_optimizer_and_weights_move(monkeypatch):
    loader, batch = _dynamic_training_fixture(monkeypatch)
    model = SmallDynamicEntry()
    prepared = trainer._materialize_dynamic_m5_batch_targets(batch, loader.dataset)
    for key in _dynamic_target_arrays([0]): prepared[key].zero_()
    from gx1.contracts.entry_observed_market_horizons_v1 import M5_FORECAST_VALID_BATCH_KEY, SUPPORT_FORECAST_TARGET_BATCH_KEY, SUPPORT_FORECAST_VALID_BATCH_KEY
    for key in (*trainer._FORECAST_TARGET_COLS, M5_FORECAST_VALID_BATCH_KEY, SUPPORT_FORECAST_TARGET_BATCH_KEY, SUPPORT_FORECAST_VALID_BATCH_KEY): prepared[key].zero_()
    losses, counts = trainer._dynamic_m5_task_loss_surfaces(model(batch["seq_x"], batch["snap_x"]), prepared, torch.device("cpu"))
    assert not losses and set(counts.values()) == {0}
    with pytest.raises(RuntimeError, match="EMPTY_BATCH"): trainer._joint_task_loss(model, losses)
    assert all(p.grad is None for p in model.parameters())


def test_dynamic_m5_model_kwargs_contain_only_declared_phase_and_known_queries():
    from gx1.contracts.entry_causal_m1_outcomes_v1 import DYNAMIC_M1_PREFIX_HORIZON_MINUTES
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    assert trainer._dynamic_m5_model_kwargs(ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE) == {
        "entry_training_phase": ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        "dynamic_m5_query_minutes": DYNAMIC_M1_PREFIX_HORIZON_MINUTES}
    assert trainer._dynamic_m5_model_kwargs("joint_fitted_q") == {"entry_training_phase": "joint_fitted_q"}



def test_dynamic_m5_arrow_getitem_keeps_pointers_target_only_and_provider_lazy(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tests.test_entry_v10_ctx_dataset_memmap import _write_advanced_parquet
    from tests.entry_v10_trainer_dataset_support import install_multi_tf_stub
    from gx1.contracts.entry_exit_production_architecture_v1 import PRODUCTION_MTF_PER_TF_WINDOW_BARS
    from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SEQ_LEN
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    from gx1.contracts import entry_model_native_aux_targets_v3 as aux
    from gx1.contracts.entry_causal_m1_outcomes_v1 import M5_PATH_TARGET_COLUMNS
    parquet = tmp_path / "dynamic.parquet"
    _write_advanced_parquet(parquet)
    table = pq.read_table(parquet)
    count = len(table)
    table = table.append_column("dynamic_entry_m1_row", pa.array(np.arange(count, dtype=np.int64) + 100))
    for name in M5_PATH_TARGET_COLUMNS: table = table.append_column(name, pa.array([float("nan")] * count))
    for name in aux.MODEL_NATIVE_FORECAST_VALID_COLUMNS: table = table.append_column(name, pa.array([False] * count))
    pq.write_table(table, parquet)
    manifest_path = parquet.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["extra"]["emission_scope"] = aux.M5_DYNAMIC_OBSERVED_EMISSION_SCOPE
    manifest["extra"]["aux_head_target_contract"] = {
        **aux.model_native_aux_target_contract_metadata(emission_scope=aux.M5_DYNAMIC_OBSERVED_EMISSION_SCOPE),
        "candidate_rows_before_primary_availability": count, "primary_unavailable_candidate_rows_excluded": 0,
        "primary_valid_rows_emitted": count, "legacy_causal_m1_observed_rows_emitted": 0,
        "legacy_aux_complete_rows_emitted": 0,
        "forecast_observed_rows_by_column": {name: 0 for name in aux.MODEL_NATIVE_FORECAST_TARGET_COLUMNS}}
    manifest_path.write_text(json.dumps(manifest))
    ds = trainer.EntryV10CtxDataset(parquet, seq_len=MODEL_NATIVE_SEQ_LEN,
        m5_prebuilt_path=install_multi_tf_stub(tmp_path, monkeypatch),
        per_tf_seq_lens=dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS), multi_tf_closed_bar=True)
    ds._entry_observed_market_binding = {"identity": {"training_phase": ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE}}
    ds._entry_observed_market_admitted_rows = np.ones(count, bool)
    ds._entry_observed_market_m5_forecast_valid = np.zeros((count, len(aux.MODEL_NATIVE_FORECAST_TARGET_COLUMNS)), bool)
    ds._entry_observed_market_support_forecast = np.zeros((count, 6), np.float32)
    ds._entry_observed_market_support_forecast_valid = np.zeros((count, 6), bool)
    ds._policy_dependent_auxiliary_bound_rows = np.zeros(count, bool)
    ds._random_access_child_index_by_parent = {i: i + 10 for i in range(count)}
    class Provider:
        def __init__(self): self.calls = []
        def targets_for_rows(self, rows):
            self.calls.append(rows.copy())
            return _dynamic_target_arrays(rows)
    ds._entry_dynamic_m5_target_provider = Provider()
    row = ds[0]
    assert row["entry_row_index"].item() == 10
    assert row["dynamic_entry_parent_row_index"].item() == 0
    assert row["dynamic_entry_m1_row"].item() == 100
    assert row[aux.M5_DYNAMIC_EMISSION_BATCH_KEY].item() is True
    assert not ds._entry_dynamic_m5_target_provider.calls
    assert not any(name in row for name in M5_PATH_TARGET_COLUMNS)
    assert "y_dip_mae_long_K12" not in row
    batch = {name: value.unsqueeze(0) for name, value in row.items()}
    actual = trainer._materialize_dynamic_m5_batch_targets(batch, ds)
    assert np.array_equal(ds._entry_dynamic_m5_target_provider.calls[0], [0])
    assert actual[observed.DYNAMIC_M5_MARKOUT_TARGET_KEY].shape[0] == 1
    assert all(name not in trainer._dynamic_m5_model_kwargs(ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
        for name in _dynamic_target_arrays([0]))
