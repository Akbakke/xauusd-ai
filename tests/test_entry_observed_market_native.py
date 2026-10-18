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
    assert bool((prediction.grad[:, :4] != 0).all())


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
