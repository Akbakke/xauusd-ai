from __future__ import annotations

import copy

import pytest
import torch

from gx1.contracts.entry_candidate_checkpoint_policy_v1 import (
    COUPLED_NET_CHECKPOINT_MONITOR,
    checkpoint_policy_metadata,
)
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.unified_exit_random_access_model_v1 import RANDOM_ACCESS_MODEL_SCHEMA_VERSION, RANDOM_ACCESS_MODEL_SCHEMA_SHA256
from gx1.contracts.unified_exit_random_access_val_checkpoint_v1 import (
    bind_candidate_weight_ema_validation_checkpoint_v1,
    require_candidate_weight_ema_val_binding_v1,
)
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from tests.test_candidate_training_session import _contract, _state, _step


def _epoch_boundary(tmp_path):
    contract = _contract()
    contract["training"] = {"checkpoint_policy": checkpoint_policy_metadata(checkpoint_monitor=COUPLED_NET_CHECKPOINT_MONITOR)}
    session = trainer._CandidateTrainingSession(out_bundle_dir=tmp_path.resolve() / "CANDIDATE", contract=contract)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4), torch.nn.Dropout(.25), torch.nn.Linear(4, 2))
    # Protocol stand-in for serialization, not a production architecture proof.
    model.unified_exit_random_access_architecture_version = RANDOM_ACCESS_MODEL_SCHEMA_VERSION
    model.register_buffer("unified_exit_random_access_architecture_sha256", torch.tensor(list(bytes.fromhex(RANDOM_ACCESS_MODEL_SCHEMA_SHA256)), dtype=torch.uint8))
    target = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    ema = trainer._WeightEma(model, .5)
    _step(model, optimizer)
    ema.update(model)
    state = _state(session, model, target, optimizer, ema, scheduler)
    state.update(phase="validation", next_batch_offset=0, global_optimizer_steps=1)
    state["training_progress"] = trainer._new_candidate_training_progress(checkpoint_monitor=COUPLED_NET_CHECKPOINT_MONITOR)
    session.save_checkpoint(state)
    return session, model, target, ema, state


def test_native_val_snapshot_survives_resume_slot_advance(tmp_path):
    session, model, _target, ema, state = _epoch_boundary(tmp_path)
    snapshot = session.save_validation_checkpoint(model=model)
    assert session.save_validation_checkpoint(model=model) == snapshot
    model.eval()
    with ema.evaluating(model):
        binding = bind_candidate_weight_ema_validation_checkpoint_v1(snapshot=snapshot, model=model)
    saved = torch.load(snapshot["path"], map_location="cpu", weights_only=True)
    assert all(torch.equal(saved["model_state"][name], value) for name, value in saved["online_buffers"].items())
    state.update(checkpoint_index=2, phase="train", epoch_index=1)
    session.save_checkpoint(state)
    assert require_candidate_weight_ema_val_binding_v1(binding) == binding
    assert session.load_checkpoint()["epoch_index"] == 1
    assert saved["epoch_index"] == 0


def test_native_val_pause_restores_online_weights_and_training_mode(tmp_path, monkeypatch):
    from gx1.scripts import run_unified_exit_random_access_val_v1 as cli

    session, model, target, ema, _state_value = _epoch_boundary(tmp_path)
    before_pointer = session._active_path.read_bytes()
    before_model = canonical_model_state_sha256(model.state_dict())
    before_target = canonical_model_state_sha256(target.state_dict())
    modes = [module.training for module in model.modules()]
    calls = []

    def paused(**kwargs):
        assert not any(module.training for module in kwargs["model"].modules())
        assert canonical_model_state_sha256(kwargs["model"].state_dict()) == canonical_model_state_sha256(ema._shadow)
        assert kwargs["candidate_target_model"] is target
        assert kwargs["checkpoint_binding"]["target_model_state_sha256"] == before_target
        calls.append(kwargs["result_path"])
        return {"decision": "PAUSED_RESUMABLE"}

    monkeypatch.setattr(cli, "evaluate_bound_full_val_v1", paused)
    context = {"frame": None, "state_factory": None, "parent_coordinate_evidence": {}, "val_sequence_audit": tmp_path / "audit.json", "max_model_forwards": 100, "max_state_views": 100, "max_wall_seconds": 60, "progress_interval_forwards": 16}
    result = trainer._native_candidate_epoch_validation(session=session, model=model, target_model=target, weight_ema=ema, val_ds=None, device=torch.device("cpu"), batch_size=2, epoch_index=0, context=context)
    assert result["decision"] == "PAUSED_RESUMABLE"
    assert calls == [session.directory / "native_val" / "epoch_0001" / "VAL_RESULT.json"]
    assert session._active_path.read_bytes() == before_pointer
    assert canonical_model_state_sha256(model.state_dict()) == before_model
    assert canonical_model_state_sha256(target.state_dict()) == before_target
    assert [module.training for module in model.modules()] == modes
    progress = session.load_checkpoint()["training_progress"]["checkpoint_selection"]
    assert progress["best_epoch"] == -1
    assert progress["last_epoch"] == 0
    assert progress["epochs_since_improve"] == 0


@pytest.mark.parametrize("censoring", ["none", "first_epoch", "all_epochs"])
def test_native_coordinator_resumes_val_before_next_epoch_and_selects_net(tmp_path, monkeypatch, censoring):
    """Exercise the real coordinator/session, using small optimizer steps and a rollout stub."""
    import hashlib
    import json
    import time
    from pathlib import Path
    from gx1.scripts import run_unified_exit_random_access_val_v1 as cli

    class Rows(torch.utils.data.Dataset):
        def __init__(self, count):
            self.count = count

        def __len__(self):
            return self.count

        def __getitem__(self, index):
            return torch.tensor(index, dtype=torch.int64)

        def set_unified_exit_lifecycle_v2_epoch(self, epoch):
            self.epoch = epoch

    policy = checkpoint_policy_metadata(checkpoint_monitor=COUPLED_NET_CHECKPOINT_MONITOR)
    contract = {**_contract(), "training": {"checkpoint_policy": policy}, "native_full_val": {"fixture": True}}
    monkeypatch.setattr(trainer, "_candidate_training_session_contract", lambda **kwargs: copy.deepcopy(contract))
    monkeypatch.setattr(trainer, "_native_candidate_val_context_binding", lambda context: {"fixture": True})
    monkeypatch.setattr(trainer, "_resolve_train_out_bundle_dir", lambda out, override: Path(out).resolve())
    monkeypatch.setattr(trainer, "_native_candidate_validation_stats", lambda result: result["fixture_stats"])
    train_epochs = []
    evaluated_epochs = []
    paused_once = False

    def train(model, target, loader, optimizer, device, **kwargs):
        train_epochs.append(loader.dataset.epoch)
        for offset, rows in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            model(rows.float()[:, None].expand(-1, 3)).square().mean().backward()
            optimizer.step()
            kwargs["weight_ema"].update(model)
            kwargs["session_checkpoint_hook"](next_batch_offset=kwargs["session_batch_offset"] + offset, complete_epoch=offset == len(loader))

    def rollout(**kwargs):
        nonlocal paused_once
        epoch = kwargs["checkpoint_binding"]["epoch_index"]
        evaluated_epochs.append(epoch)
        if not paused_once:
            paused_once = True
            path = kwargs["rollout_progress_path"]
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"fixture": "paused rollout"}))
            return {"decision": "PAUSED_RESUMABLE"}
        available = not (censoring == "all_epochs" or (censoring == "first_epoch" and epoch == 0))
        best_epoch_index = 1 if censoring == "first_epoch" else 0
        return {"decision": "PASS_COMPLETE" if available else "COMPLETE_WITH_RIGHT_CENSORING", "fixture_stats": {
            "entry_exit_policy_metrics": {"mean_net_bps_per_entry": (2.0 if epoch == best_epoch_index else 1.0) if available else None},
            "native_checkpoint_metric_available": available,
            "checkpoint_selection_unavailable_reason": None if available else "selected_trade_right_censored_at_val_end",
            # Gross ranking deliberately prefers later epochs.
            "entry_policy_realized_gross_spread_inclusive_pnl_bps_mean": 1.0 if epoch == 0 else 100.0,
            "active_head_health_ok": True, "cooperation_gate_health_ok": True,
            "exit_cooperation_gate_health_ok": True,
            "entry_action_q_raw_bps_mse_mean": 1.0,
            "entry_unique_target_action_agreement": .75,
            "unified_exit_full_trajectory_validation": {"schema_version": "fixture"},
        }}

    monkeypatch.setattr(trainer, "train_epoch", train)
    monkeypatch.setattr(cli, "evaluate_bound_full_val_v1", rollout)
    artifacts = {}
    for name in ("train", "val", "m5", "lifecycle"):
        path = tmp_path / (name + ".bin")
        path.write_bytes(name.encode())
        artifacts[name] = path
    output = tmp_path.resolve() / "CANDIDATE"
    pointer = None
    pause_reasons = []
    boundary_pointer = None
    for invocation in range(9):
        model = torch.nn.Linear(3, 2)
        model.unified_exit_random_access_architecture_version = RANDOM_ACCESS_MODEL_SCHEMA_VERSION
        model.register_buffer("unified_exit_random_access_architecture_sha256", torch.tensor(list(bytes.fromhex(RANDOM_ACCESS_MODEL_SCHEMA_SHA256)), dtype=torch.uint8))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        budget = {"stop_after_optimizer_steps": None, "stop_after_completed_val_epochs": None,
                  "max_invocation_seconds": 1000,
                  "expected_active_pointer_sha256": hashlib.sha256(pointer.read_bytes()).hexdigest() if pointer else None}
        try:
            result = trainer._run_resumable_candidate_training(
                model=model, optimizer=optimizer, weight_ema=trainer._WeightEma(model, .5),
                lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30), device=torch.device("cpu"),
                train_ds=Rows(8), val_ds=Rows(5509), effective_train_rows=8, batch_size=2,
                num_workers=0, pin_memory=False, persistent_workers=False, prefetch_factor=None,
                epochs=30, early_stopping_patience=5, early_stopping_min_delta=0.0, minimum_epochs_before_stop=1, save_top_k=1,
                out_bundle_dir=output, gx1_data_override="", run_id="V46_NATIVE_FIXTURE", dataset_run_id="V46_DATA_FIXTURE",
                train_parquet=artifacts["train"], val_parquet=artifacts["val"], m5_prebuilt_path=artifacts["m5"],
                unified_exit_lifecycle_manifest_path=artifacts["lifecycle"], input_normalization={"contract_sha256": "a" * 64},
                seed=1337, grad_accum_steps=1, grad_clip_norm=1.0, weight_decay=.0001, lr=.001, dropout=0.0, seq_len=96,
                per_tf_seq_lens={"M5": 16, "M15": 64, "H1": 96, "H4": 96, "D1": 252},
                multi_tf_num_layers=2, specialist_num_layers=1, multi_tf_scale=.5, specialist_fusion_scale=.25, cross_family_fusion_scale=.25,
                unified_exit_lifecycle_evidence={"splits": {"train": {"lifecycle_manifest_sha256": "b" * 64}}, "root_manifest_sha256": "c" * 64},
                recipe_source_provenance={}, precision_policy=trainer.DETERMINISTIC_FP32,
                execution_budget=budget, execution_budget_sha256=hashlib.sha256(str(invocation).encode()).hexdigest(), invocation_started_monotonic=time.monotonic(),
                checkpoint_monitor=COUPLED_NET_CHECKPOINT_MONITOR,
                native_val_context={"frame": None, "state_factory": None, "parent_coordinate_evidence": {}, "val_sequence_audit": artifacts["val"], "max_model_forwards": 100, "max_state_views": 100, "max_wall_seconds": 10, "progress_interval_forwards": 16},
            )
        except trainer._CandidateExecutionPaused as pause:
            evidence = pause.evidence
            pointer = Path(evidence["session_directory"]) / trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME
            pause_reasons.append(evidence["reason"])
            if len(pause_reasons) == 1:
                boundary_pointer = pointer.read_bytes()
            if evidence["reason"] == "native_full_val_window_complete":
                assert pointer.read_bytes() == boundary_pointer
                assert evidence["completed_val_epochs"] == 0
                assert train_epochs == [0]
            continue
        break
    else:
        raise AssertionError("Coordinator failed to finish the patience-5 fixture")
    epoch_count = 5 if censoring == "all_epochs" else 7 if censoring == "first_epoch" else 6
    assert train_epochs == list(range(epoch_count))
    assert evaluated_epochs == [0, 0, *range(1, epoch_count)]
    assert pause_reasons.count("native_full_val_phase_boundary") == epoch_count
    if censoring == "all_epochs":
        assert result["best_epoch"] == -1 and result["best_checkpoint"] is None
        assert result["top_k_checkpoints"] == []
    else:
        assert result["best_epoch"] == (2 if censoring == "first_epoch" else 1)
        assert result["best_policy_pnl"] == 2.0
    assert result["last_epoch"] == epoch_count and result["epochs_since_improve"] == 5
    assert result["early_stopped"] is True
