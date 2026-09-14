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


def _fixture(tmp_path, monkeypatch, fault=None):
    current = tmp_path / "CURRENT"
    current.mkdir()
    monkeypatch.setattr(trainer, "__file__", str(current / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"))
    policy = _write(current / "NEXT_RUN_POLICY.json", {"training_enabled": False})
    outputs = [tmp_path / "old", tmp_path / "new"]
    source_roots = [tmp_path / "historic_source", current]
    source_names = ["gx1/models/entry_v10/entry_v10_ctx_train_v3.py", "gx1/features/htf_features.py"]
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
    models = [torch.nn.Linear(3, 2)]
    target = copy.deepcopy(models[0])
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
        if side:
            recipe.update(candidate_resume_origin=origin, next_run_policy=policy)
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
