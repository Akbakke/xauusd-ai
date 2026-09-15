from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from gx1.contracts import unified_exit_economics_objective_v2 as economics
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return {"path": str(path), "sha256": trainer._sha256_file(path)}


def _objective(relative, rho=0.1):
    hurdle = economics.seal_train_fitted_capital_hurdle_artifact({
        "schema_version": economics.CAPITAL_HURDLE_SCHEMA_VERSION,
        "decision": "PASS", "fitted_splits": ["train"], "validation_or_test_used": False,
        "train_split_sha256": "1" * 64, "train_fold_sha256": "2" * 64,
        "source_lineage_sha256": "3" * 64, "annual_continuous_hurdle_rate": rho,
        "rate_unit": "continuous_per_wall_clock_year", "seconds_per_year": economics.SECONDS_PER_YEAR,
        "fit_method": "train_only_capital_hurdle_fit_v1", "fit_evidence_sha256": "6" * 64,
    })
    return economics.build_unified_exit_economics_objective_contract(
        capital_hurdle_artifact=hurdle, expected_train_split_sha256="1" * 64,
        expected_train_fold_sha256="2" * 64, expected_source_lineage_sha256="3" * 64,
        policy_sha256="4" * 64,
        reward_accounting=economics.LIQUIDATION_ADVANTAGE_REWARD_ACCOUNTING if relative else "terminal_cash_v2",
    )


def _fixture(tmp_path, monkeypatch, fault=None, *, initialization=None,
             policy_initialization="same_as_origin", model=None, optimizer=None, native_calibration=None, population_scope=False, changed_owner=None):
    current = tmp_path / "CURRENT"
    current.mkdir()
    monkeypatch.setattr(trainer, "__file__", str(current / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"))
    policy_value = initialization if policy_initialization == "same_as_origin" else policy_initialization
    policy = _write(current / "NEXT_RUN_POLICY.json", {
        "training_enabled": False,
        **({"exit_value_initialization": policy_value} if policy_value is not None else {}),
        **({"train_population_scope": "latest_year_2025_2026_v1"} if population_scope else {}),
    })
    population_bindings = None
    if population_scope:
        # The selection-root owner has its own file/clock/unchanged-source tests.
        # Isolate the trainer transition against that already-validated boundary.
        import numpy as np
        import pandas as pd
        from gx1.contracts import unified_exit_random_access_index_v1 as population_owner
        parent_files = {f"entry_{split}_{kind}": {"path": str(tmp_path / f"same_{split}_{kind}"), "sha256": "8" * 64}
                        for split in ("train", "val") for kind in ("parquet", "manifest")}
        base_binding = _write(tmp_path / "full_root.json", {"root_sha256": "5" * 64})
        rows = np.arange(313399, dtype=np.int64)
        frame_path = tmp_path / "full_index.parquet"
        pd.DataFrame({"entry_row_index": rows, "parent_entry_row_index": rows}).to_parquet(frame_path, index=False)
        proof = {"index_entry_row_count": 313399, "selected_entry_row_count": 3, "parent_entry_source_rows": 313399,
                 "selected_child_entry_row_indices_sha256": hashlib.sha256(rows[-3:].tobytes()).hexdigest(),
                 "selected_parent_entry_row_indices_sha256": hashlib.sha256(rows[-3:].tobytes()).hexdigest()}
        selected_root = {"root_sha256": "6" * 64,
                         "latest_year_population": {"source_root": base_binding, "source_root_sha256": "5" * 64, "splits": {"train": proof}},
                         "splits": {"train": {"index_parquet_path": str(frame_path)}}}
        if fault == "population_source":
            selected_root["latest_year_population"]["source_root"] = {**base_binding, "sha256": "f" * 64}
        def checked_population(value, *, expected_parent_bindings):
            assert expected_parent_bindings == {split: {kind: parent_files[f"entry_{split}_{kind}"] for kind in ("parquet", "manifest")} for split in ("train", "val")}
            assert value == selected_root
            return value
        monkeypatch.setattr(population_owner, "require_latest_year_index_root", checked_population)
        monkeypatch.setattr(population_owner, "latest_year_selected_entry_rows", lambda frame, split: frame["entry_row_index"].to_numpy()[-3:])
        selected_binding = _write(tmp_path / "selection_root.json", selected_root)
        population_bindings = (parent_files, base_binding, selected_binding)
    outputs = [tmp_path / "old", tmp_path / "new"]
    source_roots = [tmp_path / "historic_source", current]
    source_names = [changed_owner or "gx1/models/entry_v10/entry_v10_ctx_train_v3.py", "gx1/features/htf_features.py"]
    sources = []
    for side in range(2):
        sources.append({name: {"path": str(source_roots[side] / name), "sha256": ("b" if side and index == 0 else "a") * 64}
                        for index, name in enumerate(source_names)})
    if fault == "unapproved_source":
        sources[1][source_names[1]]["sha256"] = "f" * 64
    if fault == "source_path":
        sources[1][source_names[1]]["path"] = str(source_roots[0] / source_names[1])
    if fault == "source_added":
        sources[1]["unexpected"] = {"path": str(current / "new.py"), "sha256": "f" * 64}
    models = [torch.nn.Linear(3, 2) if model is None else model]
    target = copy.deepcopy(models[0])
    if optimizer is None:
        optimizer = torch.optim.AdamW(models[0].parameters(), lr=0.01)
    optimizer.zero_grad()
    models[0](torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad()
    for parameter_state in optimizer.state.values():
        parameter_state["step"].fill_(19908)
    ema = trainer._WeightEma(models[0], .5)
    ema.update(models[0])
    ema._steps = 19908
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    contracts, sessions, recipes = [], [], []
    origin = None
    for side in range(2):
        relative = bool(side)
        objective = _objective(relative, rho=.2 if side and fault == "risk_changed" else .1)
        readiness = {"schema_version": "gx1_unified_exit_training_economics_readiness_v3",
                     "mode": "economics_objective_v4" if relative else "economics_objective_v2",
                     "economics_objective_contract": objective, "test_data_used": False}
        if side and fault == "objective_version":
            readiness["economics_objective_contract"] = _objective(False)
        econ_binding = _write(tmp_path / f"economics{side}.json", readiness)
        monitor = trainer.MARKED_NET_CHECKPOINT_MONITOR if relative else trainer.COUPLED_NET_CHECKPOINT_MONITOR
        limits = {"policy_batch_size": 256 if side else 128,
                  "max_wall_seconds": 10800 if side else 4200,
                  "max_model_forwards": 84049614, "max_state_views": 84049614,
                  "progress_interval_forwards": 64}
        if side:
            limits["cpu_pipeline_workers"] = 8
        if side and fault == "val_profile":
            limits["policy_batch_size"] = 128
        recipe = {"schema_version": "gx1_unified_exit_random_access_full_train_recipe_v1",
                  "source_repo": str(source_roots[side]), "source_commit": ("b" if side else "a") * 40,
                  "source_bindings": sources[side], "source_bindings_sha256": _sha(sources[side]),
                  "run_id": outputs[side].name, "out_bundle_dir": str(outputs[side]),
                  "files": {"economics_readiness": econ_binding, "train": {"path": str(tmp_path / "unchanged.parquet"), "sha256": "7" * 64}},
                  "trainer_cli": {"batch_size": 16, "checkpoint_monitor": monitor},
                  "val_limits": limits, "test_data_used": False, "dataset_run_id": "frozen"}
        if population_bindings is not None:
            parent_files, base_binding, selected_binding = population_bindings
            recipe["files"].update(parent_files, random_access_root=selected_binding if side else base_binding)
        if side:
            recipe.update(candidate_resume_origin=origin, next_run_policy=policy)
        if side and native_calibration is not None:
            recipe["native_calibration"] = native_calibration
        if side and fault == "data_changed":
            recipe["files"]["train"]["sha256"] = "9" * 64
        recipe["recipe_sha256"] = _sha(recipe)
        bound_recipe = _write(tmp_path / f"recipe{side}.json", recipe)
        contract = {"schema_version": trainer._CANDIDATE_TRAINING_SESSION_SCHEMA_VERSION,
                    "authority": {"test": False}, "source_commit": recipe["source_commit"],
                    "out_bundle_dir": recipe["out_bundle_dir"], "run_id": recipe["run_id"],
                    "recipe_source_provenance": {"recipe_audit_path": bound_recipe["path"], "recipe_audit_sha256": bound_recipe["sha256"],
                                                  "source_bindings": sources[side], "source_bindings_sha256": _sha(sources[side])},
                    "native_full_val": {"compute_limits": limits, "same_factory": "bound"},
                    "training": {"batch_size": 16, "learning_rate": .01,
                                 "checkpoint_policy": trainer.checkpoint_policy_metadata(checkpoint_monitor=monitor)}}
        if population_bindings is not None:
            contract["artifacts"] = {"unified_exit_lifecycle_manifest": recipe["files"]["random_access_root"]}
            factory = {
                "schema_version": "gx1_unified_exit_random_access_val_factory_v1",
                "decision": "PASS", "split": "val", "entry_pair_count": 5508,
                "artifact_file_sha256": {
                    "random_access_index_root": recipe["files"]["random_access_root"]["sha256"],
                    "composite_normalization": "d" * 64,
                }, "test_accessed": False,
            }
            if side and fault == "population_val_root":
                factory["artifact_file_sha256"]["random_access_index_root"] = "e" * 64
            if side and fault == "population_val_normalization":
                factory["artifact_file_sha256"]["composite_normalization"] = "e" * 64
            factory["factory_sha256"] = _sha(factory)
            if side and fault == "population_val_seal":
                factory["factory_sha256"] = "e" * 64
            contract["native_full_val"]["factory_receipt"] = factory
        if side and fault == "training_changed":
            contract["training"]["learning_rate"] = .02
        session = trainer._CandidateTrainingSession(out_bundle_dir=outputs[side], contract=contract)
        contracts.append(contract); sessions.append(session); recipes.append(recipe)
        if not side:
            progress = trainer._new_candidate_training_progress(checkpoint_monitor=monitor)
            progress["joint_task_supervision_observed"] = {name: True for name in trainer.JOINT_TASK_NAMES}
            progress["joint_task_gradient_observed"] = {name: True for name in trainer.JOINT_TASK_NAMES}
            progress["checkpoint_selection"].update(epochs_since_improve=2, last_epoch=1, last_val_stats={"old_metric": 123.0})
            state = {"schema_version": trainer._CANDIDATE_TRAINING_SESSION_SCHEMA_VERSION,
                     "session_contract_sha256": session.contract_sha256,
                     "checkpoint_index": 315, "phase": "train", "epoch_index": 1,
                     "next_batch_offset": 320, "global_optimizer_steps": 19908,
                     "epoch_order": torch.arange(313399, dtype=torch.int64).flip(0),
                     "model_state": models[0].state_dict(), "target_model_state": target.state_dict(),
                     "optimizer_state": optimizer.state_dict(), "weight_ema_state": ema.checkpoint_state(),
                     "lr_scheduler_state": scheduler.state_dict(),
                     "rng_state": trainer._attended_session_rng_state(device=torch.device("cpu")),
                     "training_progress": progress, "complete": False}
            if fault == "cursor":
                state["global_optimizer_steps"] = 19909
            session.save_checkpoint(state)
            origin = {"schema_version": trainer._CANDIDATE_ECONOMICS_TRANSITION_SCHEMA,
                      "contract": {"path": str(session._contract_path), "sha256": trainer._sha256_file(session._contract_path)},
                      "pointer": {"path": str(session._active_path), "sha256": trainer._sha256_file(session._active_path)}}
            if initialization is not None:
                origin["exit_value_initialization"] = initialization
            if population_scope:
                origin["train_population_scope"] = "latest_year_2025_2026_v1"
    return sessions, origin


def _identical(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _identical(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _identical(a, b)
    else:
        assert left == right


def test_economics_transition_preserves_states_resets_coverage_and_is_crash_idempotent(tmp_path, monkeypatch):
    (old, new), origin = _fixture(tmp_path, monkeypatch)
    before = old.load_checkpoint()
    hashes = {path.name: trainer._sha256_file(path) for path in old.directory.iterdir() if path.is_file()}
    order = torch.arange(313399, dtype=torch.int64)
    state = trainer._load_candidate_economics_successor_state(session=new, origin=origin, epoch_order=order)
    receipt_path = new.directory / trainer._CANDIDATE_ECONOMICS_TRANSITION_RECEIPT
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    assert receipt["origin_cursor"]["global_optimizer_steps"] == 19908
    assert receipt["origin_cursor"]["checkpoint_index"] == 315
    assert receipt["optimizer_internal_step_histogram"] == {"19908": 2}
    assert receipt["ema_internal_steps"] == before["weight_ema_state"]["steps"]
    for key in receipt["preserved_state_fields"]:
        _identical(state[key], before[key])
    assert state["checkpoint_index"] == 1
    assert state["epoch_index"] == state["next_batch_offset"] == state["global_optimizer_steps"] == 0
    assert torch.equal(state["epoch_order"], order)
    assert state["training_progress"] == trainer._new_candidate_training_progress(checkpoint_monitor=trainer.MARKED_NET_CHECKPOINT_MONITOR)
    repeated = trainer._load_candidate_economics_successor_state(session=new, origin=origin, epoch_order=order)
    _identical(repeated, state)
    assert receipt_path.read_bytes() == receipt_bytes
    assert {path.name: trainer._sha256_file(path) for path in old.directory.iterdir() if path.is_file()} == hashes
    new.save_checkpoint(state)
    persisted = new.load_checkpoint()
    _identical(persisted, state)
    restored_model = torch.nn.Linear(3, 2)
    restored_target = copy.deepcopy(restored_model)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=.01)
    restored_ema = trainer._WeightEma(restored_model, .5)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=30)
    restored = trainer._restore_candidate_training_checkpoint(
        persisted, session=new, model=restored_model, target_model=restored_target,
        optimizer=restored_optimizer, weight_ema=restored_ema,
        lr_scheduler=restored_scheduler, device=torch.device("cpu"), dataset_rows=313399,
    )
    assert restored["global_optimizer_steps"] == (
        restored["epoch_index"] * ((313399 + 15) // 16) + restored["next_batch_offset"]
    ) == 0
    assert all(float(item["step"]) == 19908 for item in restored_optimizer.state.values())
    assert restored_ema._steps == 19908
    _identical(restored_model.state_dict(), before["model_state"])
    _identical(restored_target.state_dict(), before["target_model_state"])
    _identical(restored_optimizer.state_dict(), before["optimizer_state"])
    _identical(restored_scheduler.state_dict(), before["lr_scheduler_state"])
    _identical(restored_ema.checkpoint_state(), before["weight_ema_state"])
    _identical(trainer._attended_session_rng_state(device=torch.device("cpu")), before["rng_state"])
    with pytest.raises(RuntimeError, match="SESSION_BINDING_INVALID"):
        trainer._load_candidate_economics_successor_state(session=new, origin=origin, epoch_order=order)


@pytest.mark.parametrize("fault,expected", [
    ("cursor", "STOPPED_CURSOR_INVALID"), ("unapproved_source", "UNAPPROVED_SOURCE_CHANGED"),
    ("source_path", "SOURCE_PATH_INVALID"), ("source_added", "SOURCE_CLOSURE_CHANGED"),
    ("risk_changed", "ECONOMIC_PARAMETERS_CHANGED"), ("objective_version", "OBJECTIVE_VERSION_INVALID"),
    ("training_changed", "MODEL_DATA_OR_TRAINING_CHANGED"), ("data_changed", "MODEL_DATA_OR_TRAINING_CHANGED"),
    ("val_profile", "VAL_PROFILE_INVALID"),
])
def test_economics_transition_rejects_unapproved_changes(tmp_path, monkeypatch, fault, expected):
    (old, new), origin = _fixture(tmp_path, monkeypatch, fault)
    with pytest.raises(RuntimeError, match=expected):
        trainer._load_candidate_economics_successor_state(session=new, origin=origin, epoch_order=torch.arange(313399))
    assert not new._active_path.exists()
    assert not (new.directory / trainer._CANDIDATE_ECONOMICS_TRANSITION_RECEIPT).exists()


def test_economics_transition_rejects_receipt_and_order_tampering(tmp_path, monkeypatch):
    (old, new), origin = _fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="FULL_EPOCH_ORDER_INVALID"):
        trainer._load_candidate_economics_successor_state(session=new, origin=origin, epoch_order=torch.zeros(313399, dtype=torch.int64))
    order = torch.arange(313399)
    trainer._load_candidate_economics_successor_state(session=new, origin=origin, epoch_order=order)
    receipt = new.directory / trainer._CANDIDATE_ECONOMICS_TRANSITION_RECEIPT
    receipt.write_bytes(receipt.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="RECEIPT_MISMATCH"):
        trainer._load_candidate_economics_successor_state(session=new, origin=origin, epoch_order=order)


def test_economics_transition_rejects_changed_bound_pointer_bytes(tmp_path, monkeypatch):
    from gx1.contracts.local_random_access_campaign_v2 import RandomAccessCampaignError

    (old, new), origin = _fixture(tmp_path, monkeypatch)
    old._active_path.write_bytes(old._active_path.read_bytes() + b" ")
    with pytest.raises(RandomAccessCampaignError, match="SHA"):
        trainer._load_candidate_economics_successor_state(session=new, origin=origin, epoch_order=torch.arange(313399))


class _EconomicHeads(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 3)
        self.head_entry_action_q = torch.nn.Linear(3, 3)
        self.head_exit_action = torch.nn.Linear(3, 2)

    def forward(self, x):
        encoded = self.encoder(x)
        return torch.cat((self.head_entry_action_q(encoded), self.head_exit_action(encoded)), dim=-1)


def _economic_model_optimizer():
    model = _EconomicHeads()
    named = dict(model.named_parameters())
    # Deliberately interleave Exit with other parameters, unlike state_dict order.
    groups = [["head_entry_action_q.bias", "head_exit_action.weight", "encoder.weight"],
              ["encoder.bias", "head_exit_action.bias", "head_entry_action_q.weight"]]
    optimizer = torch.optim.AdamW(
        [{"params": [named[name] for name in group]} for group in groups], lr=.01, amsgrad=True,
    )
    return model, optimizer


def test_exit_baseline_changes_only_bound_head_states_and_preserves_learned_values_on_resume(tmp_path, monkeypatch):
    model, optimizer = _economic_model_optimizer()
    (old, new), origin = _fixture(tmp_path, monkeypatch, initialization="close_now_baseline_v1",
                                  model=model, optimizer=optimizer)
    before = old.load_checkpoint()
    live_model_before = copy.deepcopy(model.state_dict())
    live_optimizer_before = copy.deepcopy(optimizer.state_dict())
    before_rng = torch.get_rng_state().clone()
    hashes = {path.name: trainer._sha256_file(path) for path in old.directory.iterdir() if path.is_file()}
    order = torch.arange(313399, dtype=torch.int64)
    state = trainer._load_candidate_economics_successor_state(
        session=new, origin=origin, epoch_order=order, model=model, optimizer=optimizer,
    )
    assert torch.equal(torch.get_rng_state(), before_rng)
    _identical(model.state_dict(), live_model_before)
    _identical(optimizer.state_dict(), live_optimizer_before)
    names = {"head_exit_action.weight", "head_exit_action.bias"}
    for location in ("model_state", "target_model_state"):
        for name, value in state[location].items():
            if name in names:
                assert torch.count_nonzero(value) == 0
            else:
                _identical(value, before[location][name])
    for name, value in state["weight_ema_state"]["shadow"].items():
        if name in names:
            assert torch.count_nonzero(value) == 0
        else:
            _identical(value, before["weight_ema_state"]["shadow"][name])
    assert state["weight_ema_state"]["steps"] == before["weight_ema_state"]["steps"] == 19908
    assert state["weight_ema_state"]["decay"] == before["weight_ema_state"]["decay"]
    expected_ids = {name: saved_id for live, saved in zip(optimizer.param_groups, before["optimizer_state"]["param_groups"])
                    for parameter, saved_id in zip(live["params"], saved["params"])
                    for name, named_parameter in model.named_parameters() if parameter is named_parameter and name in names}
    assert set(expected_ids.values()) == {1, 4}
    assert set(state["optimizer_state"]["state"]) == set(before["optimizer_state"]["state"]) - {1, 4}
    for identifier, value in state["optimizer_state"]["state"].items():
        _identical(value, before["optimizer_state"]["state"][identifier])
    _identical(state["optimizer_state"]["param_groups"], before["optimizer_state"]["param_groups"])
    for key in ("lr_scheduler_state", "rng_state"):
        _identical(state[key], before[key])
    receipt_path = new.directory / trainer._CANDIDATE_ECONOMICS_TRANSITION_RECEIPT
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    details = receipt["preservation_exceptions"]
    assert details["variant"] == "close_now_baseline_v1"
    assert details["encoder_reset"] is details["entry_q_reset"] is False
    assert set(details["parameter_names"]) == names
    assert details["origin_optimizer_internal_step_histogram"] == {"19908": 6}
    assert receipt["optimizer_internal_step_histogram"] == {"19908": 4}
    assert set(receipt["preserved_state_fields"]) == {"lr_scheduler_state", "rng_state"}
    for name, detail in details["removed_optimizer_states"].items():
        assert detail["optimizer_state_id"] == expected_ids[name]
        assert detail["previous_step"] == 19908
        assert set(detail["previous_state_fields"]) == {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
    for location, values in details["value_state_changes"].items():
        old_values = before["weight_ema_state"]["shadow"] if location.endswith(".shadow") else before[location]
        for name, detail in values.items():
            assert detail["before_sha256"] == trainer.canonical_model_state_sha256({name: old_values[name]})
            assert detail["after_sha256"] == trainer.canonical_model_state_sha256({name: torch.zeros_like(old_values[name])})
            assert detail["after_all_values"] == 0.0
    repeated = trainer._load_candidate_economics_successor_state(
        session=new, origin=origin, epoch_order=order, model=model, optimizer=optimizer,
    )
    _identical(repeated, state)
    assert receipt_path.read_bytes() == receipt_bytes
    assert {path.name: trainer._sha256_file(path) for path in old.directory.iterdir() if path.is_file()} == hashes
    new.save_checkpoint(state)
    target = copy.deepcopy(model)
    ema = trainer._WeightEma(model, .5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    trainer._restore_candidate_training_checkpoint(
        new.load_checkpoint(), session=new, model=model, target_model=target, optimizer=optimizer,
        weight_ema=ema, lr_scheduler=scheduler, device=torch.device("cpu"), dataset_rows=313399,
    )
    assert not any(parameter.requires_grad for parameter in target.parameters())
    from gx1.contracts.unified_exit_random_access_model_v1 import liquidation_relative_action_values
    from gx1.contracts.unified_exit_fitted_q_v1 import build_unified_exit_fitted_q_targets
    successor_q = liquidation_relative_action_values(
        target.head_exit_action(target.encoder(torch.arange(12, dtype=torch.float32).reshape(4, 3)))
    ).reshape(2, 2, 2)
    reward = torch.tensor([[[-2.0], [0.5]], [[0.0], [3.25]]])
    q = successor_q.unsqueeze(2)
    action_valid = torch.ones_like(q, dtype=torch.bool)
    state_valid = torch.ones_like(reward, dtype=torch.bool)
    bellman, valid = build_unified_exit_fitted_q_targets(
        frozen_target_q_bps=q, exit_now_reward_bps=torch.zeros_like(reward),
        action_valid_mask=action_valid, state_valid_mask=state_valid,
        terminal_mask=torch.zeros_like(state_valid), terminal_reason_index=torch.zeros_like(reward, dtype=torch.int64),
        chunk_successor_target_q_bps=successor_q,
        chunk_successor_action_valid_mask=torch.ones_like(successor_q, dtype=torch.bool),
        bellman_target_valid_mask=action_valid, successor_observed_mask=state_valid,
        hold_immediate_reward_bps=reward, transition_discount=torch.full_like(reward, .9998),
    )
    assert bool(valid.all())
    torch.testing.assert_close(bellman[..., 0], reward, rtol=0, atol=0)
    torch.testing.assert_close(bellman[..., 1], torch.zeros_like(reward), rtol=0, atol=0)
    # The ordinary AdamW update lazily restarts just the reset parameters at one.
    optimizer.zero_grad()
    model.head_exit_action(model.encoder(torch.ones(2, 3)))[:, 0].sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    ema.update(model)
    for name, parameter in model.named_parameters():
        if name in names:
            assert float(optimizer.state[parameter]["step"]) == 1
            assert torch.count_nonzero(parameter) > 0
    continued = copy.deepcopy(state)
    continued.update(checkpoint_index=2, next_batch_offset=1, global_optimizer_steps=1,
                     model_state=copy.deepcopy(model.state_dict()), target_model_state=copy.deepcopy(target.state_dict()),
                     optimizer_state=copy.deepcopy(optimizer.state_dict()), weight_ema_state=copy.deepcopy(ema.checkpoint_state()))
    new.save_checkpoint(continued)
    restored_model, restored_optimizer = _economic_model_optimizer()
    restored_target = copy.deepcopy(restored_model)
    restored_ema = trainer._WeightEma(restored_model, .5)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=30)
    trainer._restore_candidate_training_checkpoint(
        new.load_checkpoint(), session=new, model=restored_model, target_model=restored_target,
        optimizer=restored_optimizer, weight_ema=restored_ema, lr_scheduler=restored_scheduler,
        device=torch.device("cpu"), dataset_rows=313399,
    )
    _identical(restored_model.state_dict(), continued["model_state"])
    _identical(restored_optimizer.state_dict(), continued["optimizer_state"])
    _identical(restored_ema.checkpoint_state(), continued["weight_ema_state"])
    with pytest.raises(RuntimeError, match="SESSION_BINDING_INVALID"):
        trainer._load_candidate_economics_successor_state(
            session=new, origin=origin, epoch_order=order, model=restored_model, optimizer=restored_optimizer,
        )
    assert receipt_path.read_bytes() == receipt_bytes


@pytest.mark.parametrize("value", [None, True, 0, {}, "unknown"])
def test_transition_rejects_invalid_initialization_variant(tmp_path, monkeypatch, value):
    (_, new), origin = _fixture(tmp_path, monkeypatch)
    origin["exit_value_initialization"] = value
    with pytest.raises(RuntimeError, match="ORIGIN_INVALID"):
        trainer._load_candidate_economics_successor_state(session=new, origin=origin, epoch_order=torch.arange(313399))


@pytest.mark.parametrize("initialization,policy_initialization", [
    ("close_now_baseline_v1", None), (None, "close_now_baseline_v1"),
    ("close_now_baseline_v1", "unknown"), ("close_now_baseline_v1", True),
])
def test_transition_requires_policy_to_match_exact_initialization(tmp_path, monkeypatch, initialization, policy_initialization):
    (_, new), origin = _fixture(tmp_path, monkeypatch, initialization=initialization,
                               policy_initialization=policy_initialization)
    with pytest.raises(RuntimeError, match="EXIT_VALUE_INITIALIZATION_POLICY_MISMATCH"):
        trainer._load_candidate_economics_successor_state(session=new, origin=origin, epoch_order=torch.arange(313399))
    assert not (new.directory / trainer._CANDIDATE_ECONOMICS_TRANSITION_RECEIPT).exists()


def test_baseline_requires_real_model_optimizer_mapping(tmp_path, monkeypatch):
    model, optimizer = _economic_model_optimizer()
    (_, new), origin = _fixture(tmp_path, monkeypatch, initialization="close_now_baseline_v1",
                               model=model, optimizer=optimizer)
    order = torch.arange(313399)
    with pytest.raises(RuntimeError, match="MODEL_OPTIMIZER_REQUIRED"):
        trainer._load_candidate_economics_successor_state(session=new, origin=origin, epoch_order=order)
    other_model, other_optimizer = _economic_model_optimizer()
    with pytest.raises(RuntimeError, match="OPTIMIZER_MAPPING_INVALID"):
        trainer._load_candidate_economics_successor_state(
            session=new, origin=origin, epoch_order=order, model=model, optimizer=other_optimizer,
        )
    assert not (new.directory / trainer._CANDIDATE_ECONOMICS_TRANSITION_RECEIPT).exists()


@pytest.mark.parametrize("arm,report_only", [("reference", False), ("split", True)])
def test_economics_transition_calibration_arm_is_bound_operational_control(tmp_path, monkeypatch, arm, report_only):
    control = {"schema_version": "gx1_native_learning_calibration_run_v1", "arm": arm, "report_only_val": report_only}
    (old, new), origin = _fixture(tmp_path, monkeypatch, native_calibration=control)
    state = trainer._load_candidate_economics_successor_state(
        session=new, origin=origin, epoch_order=torch.arange(313399, dtype=torch.int64),
    )
    _identical(state["model_state"], old.load_checkpoint()["model_state"])
    receipt = json.loads((new.directory / trainer._CANDIDATE_ECONOMICS_TRANSITION_RECEIPT).read_text())
    assert receipt["new_native_calibration"] == control


@pytest.mark.parametrize("control", [
    {"schema_version": "gx1_native_learning_calibration_run_v1", "arm": "other", "report_only_val": True},
    {"schema_version": "gx1_native_learning_calibration_run_v1", "arm": "split", "report_only_val": 1},
])
def test_economics_transition_rejects_unbounded_calibration_controls(tmp_path, monkeypatch, control):
    (_, new), origin = _fixture(tmp_path, monkeypatch, native_calibration=control)
    with pytest.raises(RuntimeError, match="NATIVE_CALIBRATION_INVALID"):
        trainer._load_candidate_economics_successor_state(session=new, origin=origin, epoch_order=torch.arange(313399, dtype=torch.int64))


def test_economics_transition_changes_only_selection_root_and_preserves_all_states(tmp_path, monkeypatch):
    (old, new), origin = _fixture(tmp_path, monkeypatch, population_scope=True)
    before = old.load_checkpoint()
    order = torch.tensor([313398, 313396, 313397])
    state = trainer._load_candidate_economics_successor_state(session=new, origin=origin, epoch_order=order)
    receipt = json.loads((new.directory / trainer._CANDIDATE_ECONOMICS_TRANSITION_RECEIPT).read_text())
    proof = receipt["train_population_transition"]
    assert (proof["old_epoch_entry_row_count"], proof["new_epoch_entry_row_count"], proof["physical_parent_entry_row_count"]) == (313399, 3, 313399)
    assert proof["data_normalization_fold_and_val_artifacts_unchanged"] is True
    assert torch.equal(state["epoch_order"], order)
    assert state["global_optimizer_steps"] == state["epoch_index"] == state["next_batch_offset"] == 0
    for field in receipt["preserved_state_fields"]:
        _identical(state[field], before[field])
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    ema = trainer._WeightEma(model, .5)
    restored = trainer._restore_candidate_training_checkpoint(
        state, session=new, model=model, target_model=copy.deepcopy(model), optimizer=optimizer,
        weight_ema=ema, lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30),
        device=torch.device("cpu"), dataset_rows=313399, parent_population=order,
    )
    assert restored["epoch_order"].tolist() == order.tolist()
    assert all(float(item["step"]) == 19908 for item in optimizer.state.values())
    assert ema._steps == 19908


@pytest.mark.parametrize("fault,order,error", [
    ("population_source", [313398, 313396, 313397], "POPULATION_SOURCE_ROOT_CHANGED"),
    ("population_val_root", [313398, 313396, 313397], "POPULATION_VAL_FACTORY_BINDING_INVALID"),
    ("population_val_seal", [313398, 313396, 313397], "POPULATION_VAL_FACTORY_BINDING_INVALID"),
    ("population_val_normalization", [313398, 313396, 313397], "MODEL_DATA_OR_TRAINING_CHANGED"),
    ("data_changed", [313398, 313396, 313397], "MODEL_DATA_OR_TRAINING_CHANGED"),
    (None, [0, 1, 2], "FULL_EPOCH_ORDER_INVALID"),
])
def test_selection_transition_never_loosens_other_lineage_or_parent_ids(tmp_path, monkeypatch, fault, order, error):
    (_, new), origin = _fixture(tmp_path, monkeypatch, fault=fault, population_scope=True)
    with pytest.raises(RuntimeError, match=error):
        trainer._load_candidate_economics_successor_state(session=new, origin=origin, epoch_order=torch.tensor(order))


def test_economics_transition_permits_bound_val_checkpoint_history_owner(tmp_path, monkeypatch):
    owner = "gx1/contracts/unified_exit_random_access_val_checkpoint_v1.py"
    (old, new), origin = _fixture(tmp_path, monkeypatch, changed_owner=owner)
    before = old.load_checkpoint()
    before_pointer = old._active_path.read_bytes()
    state = trainer._load_candidate_economics_successor_state(
        session=new, origin=origin, epoch_order=torch.arange(313399),
    )
    receipt = json.loads((new.directory / trainer._CANDIDATE_ECONOMICS_TRANSITION_RECEIPT).read_text())
    assert receipt["changed_source_paths"] == [owner]
    assert old._active_path.read_bytes() == before_pointer
    for key in ("model_state", "target_model_state", "optimizer_state", "weight_ema_state", "lr_scheduler_state", "rng_state"):
        _identical(state[key], before[key])


def _optimizer_procedure_fixture(tmp_path, monkeypatch, fault=None, *, continuation=False):
    from gx1.contracts import unified_exit_native_candidate_campaign_v1 as scope
    from gx1.contracts import unified_exit_random_access_val_checkpoint_v1 as val_checkpoint
    current = tmp_path / "CURRENT"
    current.mkdir()
    monkeypatch.setattr(trainer, "__file__", str(current / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"))
    # The exact deployed85 origin and finite native authority are independently
    # checked by scope-owner tests. Here use real tiny serialized sessions to
    # test preservation and the trainer's recipe/source/contract boundary.
    monkeypatch.setattr(scope, "require_training_continuation_origin" if continuation else "require_optimizer_procedure_origin", lambda origin: dict(origin))
    monkeypatch.setattr(scope, "require_native_run_scope", lambda recipe: 5521 if continuation else 5265)
    cursor = scope.TRAINING_CONTINUATION_ORIGIN_CURSOR if continuation else scope.OPTIMIZER_PROCEDURE_ORIGIN_CURSOR
    inherited = {"optimizer_step_offset": 19908, "transition_receipt": {"path": "bound_old_receipt", "sha256": "a" * 64}}
    monkeypatch.setattr(val_checkpoint, "bind_candidate_weight_ema_history_v1", lambda **kwargs: inherited)
    sessions = []
    origin = None
    model = torch.nn.Linear(3, 2)
    target = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    ema = trainer._WeightEma(model, .99)
    ema._steps = cursor["global_optimizer_steps"] + 19908
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    for side in (0, 1):
        sources = {
            "trainer": {"path": str(current / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"), "sha256": str(side + 1) * 64},
            "model": {"path": str(current / "gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py"), "sha256": "3" * 64},
        }
        if continuation and fault == "unchanged_source":
            sources["trainer"]["sha256"] = "1" * 64
        if side and fault == "model_source":
            sources["model"]["sha256"] = "4" * 64
        recipe = {"source_repo": str(current), "source_commit": str(side + 1) * 40,
                  "source_bindings": sources, "source_bindings_sha256": _sha(sources),
                  "out_bundle_dir": str(tmp_path / f"clip_bundle{side}"), "run_id": f"clip{side}",
                  "test_data_used": False, "files": {"data": "unchanged"},
                  "trainer_cli": {"batch_size": 16}, "val_limits": {"policy_batch_size": 256, "cpu_pipeline_workers": 8, "max_wall_seconds": 10800}}
        if side:
            recipe.update(candidate_resume_origin=origin, next_run_policy={"path": "bound_by_scope", "sha256": "5" * 64})
        if side and fault == "data":
            recipe["files"]["data"] = "changed"
        recipe["recipe_sha256"] = _sha(recipe)
        bound_recipe = _write(tmp_path / f"clip_recipe{side}.json", recipe)
        contract = {"schema_version": trainer._CANDIDATE_TRAINING_SESSION_SCHEMA_VERSION,
                    "authority": {"test": False}, "source_commit": recipe["source_commit"],
                    "out_bundle_dir": recipe["out_bundle_dir"], "run_id": recipe["run_id"],
                    "recipe_source_provenance": {"recipe_audit_path": bound_recipe["path"], "recipe_audit_sha256": bound_recipe["sha256"], "source_bindings": sources, "source_bindings_sha256": _sha(sources)},
                    "native_full_val": {"compute_limits": recipe["val_limits"]},
                    "training": {"batch_size": 16, "grad_clip_norm": 1.0, "checkpoint_policy": trainer.checkpoint_policy_metadata(checkpoint_monitor=trainer.MARKED_NET_CHECKPOINT_MONITOR)}}
        if side or continuation:
            contract["training"]["gradient_clipping_policy"] = trainer._GRAD_CLIP_POLICY
        if continuation and not side and fault == "old_clipping_missing":
            contract["training"].pop("gradient_clipping_policy")
        if side and fault == "clip_cap":
            contract["training"]["grad_clip_norm"] = 2.0
        if side and fault == "batch":
            contract["training"]["batch_size"] = 8
        session = trainer._CandidateTrainingSession(out_bundle_dir=Path(recipe["out_bundle_dir"]), contract=contract)
        sessions.append(session)
        if side == 0:
            progress = trainer._new_candidate_training_progress(checkpoint_monitor=trainer.MARKED_NET_CHECKPOINT_MONITOR)
            progress["checkpoint_selection"].update(last_epoch=1, last_val_stats={"negative_marked_bps": -1235.8772})
            state = {"schema_version": trainer._CANDIDATE_TRAINING_SESSION_SCHEMA_VERSION,
                     "session_contract_sha256": session.contract_sha256,
                     **cursor,
                     "epoch_order": torch.arange(65295, dtype=torch.int64).flip(0),
                     "model_state": model.state_dict(), "target_model_state": target.state_dict(),
                     "optimizer_state": optimizer.state_dict(), "weight_ema_state": ema.checkpoint_state(),
                     "lr_scheduler_state": scheduler.state_dict(),
                     "rng_state": trainer._attended_session_rng_state(device=torch.device("cpu")),
                     "training_progress": progress}
            if fault == "cursor": state["next_batch_offset"] += 1
            if fault == "ema_steps": state["weight_ema_state"]["steps"] -= 1
            session.save_checkpoint(state)
            pointer = json.loads(session._active_path.read_text())
            monkeypatch.setattr(scope, "TRAINING_CONTINUATION_ORIGIN_STATE_SHA256" if continuation else "OPTIMIZER_PROCEDURE_ORIGIN_STATE_SHA256", pointer["state_sha256"])
            origin = {"schema_version": scope.TRAINING_CONTINUATION_SCHEMA if continuation else scope.OPTIMIZER_PROCEDURE_TRANSITION_SCHEMA,
                      "contract": {"path": str(session._contract_path), "sha256": trainer._sha256_file(session._contract_path)},
                      "pointer": {"path": str(session._active_path), "sha256": trainer._sha256_file(session._active_path)},
                      "exit_value_initialization": "close_now_baseline_v1", "train_population_scope": "latest_year_2025_2026_v1",
                      "gradient_clipping_policy": trainer._GRAD_CLIP_POLICY}
    return sessions, origin


def test_optimizer_transition_preserves_entire_stopped_state_and_crash_retry(tmp_path, monkeypatch):
    (old, new), origin = _optimizer_procedure_fixture(tmp_path, monkeypatch)
    before = old.load_checkpoint()
    hashes = {p.name: trainer._sha256_file(p) for p in old.directory.iterdir() if p.is_file()}
    state = trainer._load_candidate_optimizer_procedure_successor_state(session=new, origin=origin)
    assert state["session_contract_sha256"] == new.contract_sha256
    for key in before.keys() - {"session_contract_sha256"}:
        _identical(state[key], before[key])
    receipt_path = new.directory / "CANDIDATE_OPTIMIZER_PROCEDURE_TRANSITION.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["state_preserved"] and receipt["optimizer_procedure_changed"]
    assert receipt["identical_future_trajectory_claimed"] is False
    assert receipt["inherited_weight_ema_history"]["optimizer_step_offset"] == 19908
    assert receipt["ema_internal_steps"] == 25141
    repeated = trainer._load_candidate_optimizer_procedure_successor_state(session=new, origin=origin)
    _identical(state, repeated)
    new.save_checkpoint(state)
    _identical(new.load_checkpoint(), state)
    assert hashes == {p.name: trainer._sha256_file(p) for p in old.directory.iterdir() if p.is_file()}


@pytest.mark.parametrize("fault,expected", [
    ("model_source", "UNAPPROVED_SOURCE_CHANGED"), ("data", "MODEL_DATA_OR_TRAINING_CHANGED"),
    ("batch", "MODEL_DATA_OR_TRAINING_CHANGED"), ("clip_cap", "CLIPPING_POLICY_INVALID"),
    ("cursor", "STOPPED_STATE_INVALID"), ("ema_steps", "EMA_HISTORY_INVALID"),
])
def test_optimizer_transition_rejects_unrelated_change(tmp_path, monkeypatch, fault, expected):
    (old, new), origin = _optimizer_procedure_fixture(tmp_path, monkeypatch, fault)
    with pytest.raises(RuntimeError, match=expected):
        trainer._load_candidate_optimizer_procedure_successor_state(session=new, origin=origin)
    assert not new._active_path.exists()
    assert not (new.directory / "CANDIDATE_OPTIMIZER_PROCEDURE_TRANSITION.json").exists()



@pytest.mark.parametrize("fault", [None, "unchanged_source"])
def test_training_continuation_preserves_all_state_and_crash_retry(tmp_path, monkeypatch, fault):
    from gx1.contracts import unified_exit_native_candidate_campaign_v1 as scope
    (old, new), origin = _optimizer_procedure_fixture(tmp_path, monkeypatch, fault, continuation=True)
    before = old.load_checkpoint()
    old_hashes = {p.name: trainer._sha256_file(p) for p in old.directory.iterdir() if p.is_file()}
    state = trainer._load_candidate_optimizer_procedure_successor_state(session=new, origin=origin)
    assert state["session_contract_sha256"] == new.contract_sha256
    assert state["global_optimizer_steps"] == 5265 and state["next_batch_offset"] == 1184
    for key in before.keys() - {"session_contract_sha256"}:
        _identical(state[key], before[key])
    receipt_path = new.directory / scope.TRAINING_CONTINUATION_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text())
    assert receipt["schema_version"] == scope.TRAINING_CONTINUATION_RECEIPT_SCHEMA
    assert receipt["state_preserved"] is True
    assert receipt["optimizer_procedure_changed"] is False
    assert receipt["identical_future_trajectory_claimed"] is False
    assert receipt["ema_internal_steps"] == 25173
    assert receipt["inherited_weight_ema_history"]["optimizer_step_offset"] == 19908
    repeated = trainer._load_candidate_optimizer_procedure_successor_state(session=new, origin=origin)
    _identical(repeated, state)
    receipt_path.write_text(json.dumps({**receipt, "optimizer_procedure_changed": True}))
    with pytest.raises(RuntimeError, match="RECEIPT_MISMATCH"):
        trainer._load_candidate_optimizer_procedure_successor_state(session=new, origin=origin)
    receipt_path.write_bytes(trainer._candidate_training_session_json_bytes(receipt))
    new.save_checkpoint(state)
    _identical(new.load_checkpoint(), state)
    with pytest.raises(RuntimeError, match="SESSION_BINDING_INVALID"):
        trainer._load_candidate_optimizer_procedure_successor_state(session=new, origin=origin)
    assert old_hashes == {p.name: trainer._sha256_file(p) for p in old.directory.iterdir() if p.is_file()}


@pytest.mark.parametrize("fault,expected", [
    ("model_source", "UNAPPROVED_SOURCE_CHANGED"), ("data", "MODEL_DATA_OR_TRAINING_CHANGED"),
    ("batch", "MODEL_DATA_OR_TRAINING_CHANGED"), ("clip_cap", "CLIPPING_POLICY_INVALID"),
    ("old_clipping_missing", "CLIPPING_POLICY_INVALID"),
    ("cursor", "STOPPED_STATE_INVALID"), ("ema_steps", "EMA_HISTORY_INVALID"),
])
def test_training_continuation_rejects_any_learning_or_lineage_change(tmp_path, monkeypatch, fault, expected):
    from gx1.contracts import unified_exit_native_candidate_campaign_v1 as scope
    (_, new), origin = _optimizer_procedure_fixture(tmp_path, monkeypatch, fault, continuation=True)
    with pytest.raises(RuntimeError, match=expected):
        trainer._load_candidate_optimizer_procedure_successor_state(session=new, origin=origin)
    assert not new._active_path.exists()
    assert not (new.directory / scope.TRAINING_CONTINUATION_RECEIPT_NAME).exists()
