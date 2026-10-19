from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from gx1.contracts.entry_candidate_checkpoint_policy_v1 import (
    COUPLED_NET_CHECKPOINT_MONITOR,
    MARKED_NET_CHECKPOINT_MONITOR,
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


def _epoch_boundary(tmp_path, *, native_calibration=None):
    contract = _contract()
    if native_calibration is not None:
        recipe_path = tmp_path / "report-recipe.json"
        recipe_path.write_text(json.dumps({
            "native_calibration": native_calibration,
            "val_limits": {"policy_batch_size": 256, "cpu_pipeline_workers": 8,
                           "max_wall_seconds": 10800, "progress_interval_forwards": 64},
        }))
        contract["recipe_source_provenance"] = {
            "recipe_audit_path": str(recipe_path), "recipe_audit_sha256": trainer._sha256_file(recipe_path),
        }
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
    if native_calibration is not None:
        state.update(phase="train", next_batch_offset=32, global_optimizer_steps=32)
        state["weight_ema_state"]["steps"] = ema._steps = 32
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
        assert kwargs["cpu_pipeline_workers"] == 8
        assert kwargs["exit_policy_batch_size"] == 256
        assert kwargs["candidate_target_model"] is target
        assert kwargs["checkpoint_binding"]["target_model_state_sha256"] == before_target
        calls.append(kwargs["result_path"])
        return {"decision": "PAUSED_RESUMABLE"}

    monkeypatch.setattr(cli, "evaluate_bound_full_val_v1", paused)
    context = {"frame": None, "state_factory": None, "parent_coordinate_evidence": {}, "val_sequence_audit": tmp_path / "audit.json", "max_model_forwards": 100, "max_state_views": 100, "max_wall_seconds": 60, "progress_interval_forwards": 16}
    context.update(policy_batch_size=256, cpu_pipeline_workers=8)
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


@pytest.mark.parametrize("monitor", [COUPLED_NET_CHECKPOINT_MONITOR, MARKED_NET_CHECKPOINT_MONITOR])
@pytest.mark.parametrize("censoring", ["none", "first_epoch", "all_epochs"])
def test_native_coordinator_resumes_val_before_next_epoch_and_selects_net(tmp_path, monkeypatch, censoring, monitor):
    """Exercise the real coordinator/session, using small optimizer steps and a rollout stub."""
    import hashlib
    import json
    import time
    from pathlib import Path
    from gx1.scripts import run_unified_exit_random_access_val_v1 as cli

    class Rows(torch.utils.data.Dataset):
        def __init__(self, count):
            self.count = count
            self._unified_exit_lifecycle_v2 = object()

        def __len__(self):
            return self.count

        def __getitem__(self, index):
            return torch.tensor(index, dtype=torch.int64)

        def set_unified_exit_lifecycle_v2_epoch(self, epoch):
            self.epoch = epoch

    policy = checkpoint_policy_metadata(checkpoint_monitor=monitor)
    contract = {**_contract(), "training": {"checkpoint_policy": policy}, "native_full_val": {"fixture": True}}
    monkeypatch.setattr(trainer, "_candidate_training_session_contract", lambda **kwargs: copy.deepcopy(contract))
    monkeypatch.setattr(trainer, "_native_candidate_val_context_binding", lambda context: {"fixture": True})
    monkeypatch.setattr(trainer, "_resolve_train_out_bundle_dir", lambda out, override: Path(out).resolve())
    monkeypatch.setattr(trainer, "_native_candidate_validation_stats", lambda result: result["fixture_stats"])
    train_epochs = []
    evaluated_epochs = []
    paused_once = False

    def train(model, target, loader, optimizer, device, **kwargs):
        assert kwargs["session_exit_action_forward_chunk_rows"] is None
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
            "entry_exit_policy_metrics": {"long_count": 1, "short_count": 1, "flat_count": 2, "mean_net_bps_per_entry": (2.0 if epoch == best_epoch_index else (100.0 if monitor == MARKED_NET_CHECKPOINT_MONITOR else 1.0)) if available else None},
            "native_checkpoint_monitor": monitor,
            "marked_policy_evaluation": {"single_position_replay": {
                "full_cohort_authoritative": available,
                "net_cash_plus_open_value_bps_sum": (2.0 if epoch == best_epoch_index else -10.0) if available else None,
            }},
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
                checkpoint_monitor=monitor,
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


@pytest.mark.parametrize("step_limit", [None, 1])
def test_cuda_lifecycle_dispatch_preserves_no_nested_chunk(monkeypatch, step_limit):
    """CPU fixture exercises CUDA control flow, not CUDA computation."""
    from types import SimpleNamespace

    class LifecycleReached(Exception):
        pass

    from tests.test_unified_exit_dataset_adapter_v2 import _readiness
    from gx1.contracts.unified_exit_fitted_q_v1 import require_unified_exit_unbounded_training_readiness

    readiness = require_unified_exit_unbounded_training_readiness(
        _readiness(), context="CUDA_LIFECYCLE_DISPATCH_FIXTURE",
    )
    dataset = object.__new__(trainer.EntryV10CtxDataset)
    dataset._unified_exit_lifecycle_v2 = SimpleNamespace(
        _readiness=readiness,
        random_access_training_bindings_v1=lambda: {
            "economics_objective_contract_sha256": readiness["economics_objective_contract"]["contract_sha256"],
        },
    )
    value = torch.zeros(1, 2)

    class Transfer:
        def to(self, device, **kwargs):
            assert device.type == "cuda"
            return value

    class Loader:
        def __init__(self):
            self.dataset = dataset

        def __len__(self):
            return 1

        def __iter__(self):
            yield {
                **{key: Transfer() for key in ("seq_x", "snap_x", "ctx_cont", "ctx_cat")},
                "entry_row_index": torch.tensor([0]),
            }

    model = torch.nn.Linear(2, 2)
    target = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    monkeypatch.setattr(trainer, "_synchronized_exit_profile_clock", lambda device: 0.0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(trainer, "_multi_tf_kwargs_from_batch", lambda *args: {})
    monkeypatch.setattr(
        trainer, "_model_forward_fp32",
        lambda *args, **kwargs: {trainer.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY: value},
    )

    def reached(**kwargs):
        assert kwargs["dataset"] is dataset
        raise LifecycleReached

    monkeypatch.setattr(trainer, "_episode_native_exit_train_v2", reached)
    with pytest.raises(LifecycleReached):
        trainer.train_epoch(
            model, target, Loader(), optimizer, SimpleNamespace(type="cuda"),
            grad_accum_steps=1, task_supervision_observed={}, task_gradient_observed={},
            session_max_optimizer_steps=step_limit,
            session_checkpoint_hook=lambda **kwargs: None,
            session_checkpoint_interval_optimizer_steps=64,
            session_exit_action_forward_chunk_rows=None,
            session_log_label="NATIVE_CUDA_DISPATCH_TEST",
        )


def test_capacity_limits_are_bound_to_native_session(tmp_path):
    import pandas as pd
    from gx1.contracts.unified_exit_random_access_val_factory_v1 import RandomAccessValStateFactoryV1
    factory = RandomAccessValStateFactoryV1.__new__(RandomAccessValStateFactoryV1)
    factory.factory_receipt = {"fixture": True}
    audit = tmp_path / "audit.json"
    audit.write_text("{}")
    context = dict(frame=pd.DataFrame({"entry_row_index": range(5508), "parent_entry_row_index": range(5508)}),
                   state_factory=factory, parent_coordinate_evidence={}, val_sequence_audit=audit,
                   max_model_forwards=100, max_state_views=100, max_wall_seconds=10800,
                   progress_interval_forwards=64, policy_batch_size=256, cpu_pipeline_workers=8)
    bound = trainer._native_candidate_val_context_binding(context)
    assert bound["compute_limits"]["cpu_pipeline_workers"] == 8
    assert bound["compute_limits"]["policy_batch_size"] == 256
    assert bound["compute_limits"]["max_wall_seconds"] == 10800
    for key, value in (("cpu_pipeline_workers", 9), ("policy_batch_size", 512)):
        with pytest.raises(RuntimeError, match="LIMITS_INVALID"):
            trainer._native_candidate_val_context_binding({**context, key: value})


def _complete_selection_result(marked):
    import numpy as np
    from gx1.contracts.unified_exit_entry_policy_evaluation_v1 import (
        build_entry_policy_decisions, coupled_entry_exit_policy_metrics, marked_entry_exit_policy_metrics,
    )
    from gx1.contracts.unified_exit_random_access_val_evaluator_v1 import RESULT_SCHEMA_VERSION, MARKED_RESULT_SCHEMA_VERSION
    binding = "a" * 64
    policy = build_entry_policy_decisions(
        predicted_q_bps=np.tile(np.array([[2, 1, 0]], dtype=np.float32), (5508, 1)),
        entry_row_indices=list(range(5508)), checkpoint_binding_sha256=binding,
    )
    outcomes = []
    for row in range(5508):
        for side in (0, 1):
            closed = side == 1
            outcomes.append({"entry_row_index": row, "side_index": side,
                "status": "EXITED" if closed else "RIGHT_CENSORED_SPLIT_END",
                "entry_fill_time_ns": 100 + row * 10,
                "exit_decision_time_ns": 110 + row * 10 if closed else None,
                "undiscounted_net_cash_pnl_bps": 200.0 if closed else -1.0,
                "valuation": {"remaining_liquidation_value_bps": 0.0 if closed else -29.0,
                    "net_cash_plus_open_value_bps": 200.0 if closed else -30.0,
                    "valuation_time_ns": 110 + row * 10 if closed else 100000,
                    "model_exit_executed": closed}})
    result = {"schema_version": MARKED_RESULT_SCHEMA_VERSION if marked else RESULT_SCHEMA_VERSION,
        "decision": "COMPLETE_WITH_RIGHT_CENSORING", "test_data_used": False,
        "rollout_execution_complete": True, "entry_pair_cohort_size": 5508,
        "side_trade_count": 11016, "exited_side_trade_count": 5508,
        "right_censored_side_trade_count": 5508, "compute_truncated_side_trade_count": 0,
        "checkpoint_binding_sha256": binding, "checkpoint_binding": {"target_model_state_sha256": binding},
        "entry_gate_and_feature_route_diagnostics": {"candidate_active_head_evidence": {"target_model_state_sha256": binding}},
        "trade_outcomes": outcomes, "entry_policy_decisions": policy,
        "entry_exit_policy_metrics": coupled_entry_exit_policy_metrics(entry_policy=policy, trade_outcomes=outcomes, full_cohort_authoritative=True),
        "full_cohort_policy_metrics_authoritative": False}
    if marked:
        result["marked_policy_evaluation"] = marked_entry_exit_policy_metrics(
            entry_policy=policy, trade_outcomes=outcomes, full_cohort_authoritative=True)
    return result


@pytest.mark.parametrize("marked", [False, True])
def test_native_selection_reconstructs_complete_value_without_forcing_exit(monkeypatch, marked):
    from gx1.contracts.entry_candidate_checkpoint_policy_v1 import checkpoint_metric
    monkeypatch.setattr(trainer, "_native_val_gate_health_stats", lambda result: {})
    result = _complete_selection_result(marked)
    stats = trainer._native_candidate_validation_stats(result)
    assert stats["native_checkpoint_metric_available"] is marked
    assert result["entry_exit_policy_metrics"]["mean_net_bps_per_entry"] is None
    monitor = MARKED_NET_CHECKPOINT_MONITOR if marked else COUPLED_NET_CHECKPOINT_MONITOR
    assert stats["native_checkpoint_monitor"] == monitor
    if marked:
        assert checkpoint_metric(stats, checkpoint_monitor=monitor) == -30.0
        replay = stats["marked_policy_evaluation"]["single_position_replay"]
        assert replay["executed_trade_count"] == 1 and replay["skipped_while_position_open_count"] == 5507
        assert replay["open_position_count"] == 1 and replay["model_exited_count"] == 0
        assert stats["marked_policy_evaluation"]["used_for_early_stopping"] is True
        assert result["marked_policy_evaluation"]["used_for_early_stopping"] is False
        result["marked_policy_evaluation"]["single_position_replay"]["net_cash_plus_open_value_bps_sum"] = 200.0
        with pytest.raises(RuntimeError, match="MARKED_VAL_METRICS_INVALID"):
            trainer._native_candidate_validation_stats(result)
    else:
        result["marked_policy_evaluation"] = {}
        with pytest.raises(RuntimeError, match="MARKED_VAL_SCHEMA_MISMATCH"):
            trainer._native_candidate_validation_stats(result)


@pytest.mark.parametrize("marked", [False, True])
def test_native_campaign_recipe_binds_monitor_to_economics(tmp_path, marked):
    import hashlib
    import json
    from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_native_recipe_metadata, native_sha256
    from tests.test_unified_exit_economics_objective_v2 import _contract as objective_contract
    def bound(name, data, digest=None):
        if digest:
            data[digest] = native_sha256(data)
        path = tmp_path / name
        path.write_text(json.dumps(data))
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    economics = bound("economics.json", {"economics_objective_contract": objective_contract(
        reward_accounting="liquidation_value_increments_v1" if marked else "terminal_cash_v2")})
    manifest = {"entry_row_count": 313399, "parent_entry_source_rows": 313399}
    manifest_binding = bound("train.json", manifest, "manifest_sha256")
    root = bound("root.json", {"decision": "PASS", "allowed_splits": ["train", "val"], "test_accessed": False,
        "splits": {"train": {"manifest_path": manifest_binding["path"], "manifest_sha256": manifest["manifest_sha256"]}}}, "root_sha256")
    monitor = MARKED_NET_CHECKPOINT_MONITOR if marked else COUPLED_NET_CHECKPOINT_MONITOR
    recipe = {"schema_version": "gx1_unified_exit_random_access_full_train_recipe_v1", "profile": "candidate",
        "test_data_used": False, "source_repo": str(tmp_path), "source_commit": "b" * 40,
        "val_limits": {"policy_batch_size": 256, "cpu_pipeline_workers": 8, "max_wall_seconds": 10800, "progress_interval_forwards": 64},
        "trainer_cli": {"epochs": 30, "batch_size": 16, "early_stopping_patience": 5, "checkpoint_monitor": monitor},
        "files": {"economics_readiness": economics, "random_access_root": root}}
    binding = bound("recipe.json", recipe, "recipe_sha256")
    assert require_native_recipe_metadata(binding, source_repo=tmp_path, source_commit="b" * 40)[1] == 313399
    recipe["trainer_cli"]["checkpoint_monitor"] = COUPLED_NET_CHECKPOINT_MONITOR if marked else MARKED_NET_CHECKPOINT_MONITOR
    del recipe["recipe_sha256"]
    binding = bound("recipe.json", recipe, "recipe_sha256")
    with pytest.raises(RuntimeError, match="RECIPE_METADATA_INVALID"):
        require_native_recipe_metadata(binding, source_repo=tmp_path, source_commit="b" * 40)


def _transition_epoch_boundary(tmp_path, monkeypatch, initialization):
    from tests.test_candidate_economics_transition import _fixture, _economic_model_optimizer

    model, optimizer = _economic_model_optimizer()
    model.unified_exit_random_access_architecture_version = RANDOM_ACCESS_MODEL_SCHEMA_VERSION
    model.register_buffer("unified_exit_random_access_architecture_sha256", torch.tensor(
        list(bytes.fromhex(RANDOM_ACCESS_MODEL_SCHEMA_SHA256)), dtype=torch.uint8,
    ))
    (old, session), origin = _fixture(tmp_path, monkeypatch, initialization=initialization,
                                     model=model, optimizer=optimizer)
    state = trainer._load_candidate_economics_successor_state(
        session=session, origin=origin, epoch_order=torch.arange(313399), model=model, optimizer=optimizer,
    )
    state.update(phase="validation", next_batch_offset=0, global_optimizer_steps=1)
    state["weight_ema_state"]["steps"] += 1
    model.load_state_dict(state["model_state"])
    ema = trainer._WeightEma(model, .5)
    ema.restore_checkpoint_state(state["weight_ema_state"], model=model)
    session.save_checkpoint(state)
    return old, session, model, ema, state


@pytest.mark.parametrize("initialization", [None, "close_now_baseline_v1"])
def test_transition_ema_offset_is_exact_and_bound_through_snapshot_and_reader(tmp_path, monkeypatch, initialization):
    old, session, model, ema, state = _transition_epoch_boundary(tmp_path, monkeypatch, initialization)
    origin_hashes = {path.name: trainer._sha256_file(path) for path in old.directory.iterdir() if path.is_file()}
    snapshot = session.save_validation_checkpoint(model=model)
    model.eval()
    with ema.evaluating(model):
        binding = bind_candidate_weight_ema_validation_checkpoint_v1(snapshot=snapshot, model=model)
    assert binding["weight_ema_steps"] == 19909
    assert binding["global_step"] == 1
    assert binding["weight_ema_history"]["optimizer_step_offset"] == 19908
    assert binding["immutable_epoch_snapshot"] is True
    receipt = session.directory / trainer._CANDIDATE_ECONOMICS_TRANSITION_RECEIPT
    assert binding["weight_ema_history"]["transition_receipt"] == {
        "path": str(receipt), "sha256": trainer._sha256_file(receipt),
    }
    saved = torch.load(snapshot["path"], map_location="cpu", weights_only=True)
    assert saved["weight_ema_history"] == binding["weight_ema_history"]
    state.update(checkpoint_index=2, phase="train", epoch_index=1)
    session.save_checkpoint(state)
    assert require_candidate_weight_ema_val_binding_v1(binding) == binding
    assert {path.name: trainer._sha256_file(path) for path in old.directory.iterdir() if path.is_file()} == origin_hashes
    from gx1.contracts.unified_exit_random_access_checkpoint_v1 import canonical_sha256
    for delta in (-1, 1):
        changed = copy.deepcopy(binding)
        changed["weight_ema_steps"] += delta
        changed["binding_sha256"] = canonical_sha256({key: value for key, value in changed.items() if key != "binding_sha256"})
        with pytest.raises(RuntimeError, match="VAL_BINDING_INVALID"):
            require_candidate_weight_ema_val_binding_v1(changed)
    # Removing the history and making the counters equal cannot downgrade a transitioned snapshot.
    changed = copy.deepcopy(binding)
    changed.pop("weight_ema_history")
    changed["weight_ema_steps"] = changed["global_step"]
    changed["binding_sha256"] = canonical_sha256({key: value for key, value in changed.items() if key != "binding_sha256"})
    with pytest.raises(RuntimeError, match="EMA_HISTORY_INVALID"):
        require_candidate_weight_ema_val_binding_v1(changed)


@pytest.mark.parametrize("field,value", [("ema_internal_steps", 19907), ("destination_session_contract_sha256", "f" * 64)])
def test_snapshot_rejects_history_not_bound_to_origin_or_destination(tmp_path, monkeypatch, field, value):
    _, session, model, _, _ = _transition_epoch_boundary(tmp_path, monkeypatch, None)
    receipt_path = session.directory / trainer._CANDIDATE_ECONOMICS_TRANSITION_RECEIPT
    receipt = json.loads(receipt_path.read_text())
    receipt[field] = value
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="EMA_HISTORY_INVALID"):
        session.save_validation_checkpoint(model=model)


def test_snapshot_without_transition_keeps_strict_equal_ema_counter(tmp_path):
    session, model, _, _, state = _epoch_boundary(tmp_path)
    state["weight_ema_state"]["steps"] += 1
    state["checkpoint_index"] = 2
    session.save_checkpoint(state)
    with pytest.raises(RuntimeError, match="CHECKPOINT_EMA_INVALID"):
        session.save_validation_checkpoint(model=model)


@pytest.mark.parametrize("arm", ["reference", "split"])
def test_report_only_train32_snapshot_is_bound_immutable_and_not_selectable(tmp_path, arm):
    calibration = {"schema_version": "gx1_native_learning_calibration_run_v1", "arm": arm, "report_only_val": True}
    session, model, _target, ema, state = _epoch_boundary(tmp_path, native_calibration=calibration)
    before_pointer = session._active_path.read_bytes()
    before_model = canonical_model_state_sha256(model.state_dict())
    before_rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="CHECKPOINT_PHASE_INVALID"):
        session.save_validation_checkpoint(model=model)
    snapshot = session.save_validation_checkpoint(model=model, report_only=True)
    assert Path(snapshot["path"]).name == "calibration_step_0032.pt"
    assert session.save_validation_checkpoint(model=model, report_only=True) == snapshot
    saved = torch.load(snapshot["path"], map_location="cpu", weights_only=True)
    assert saved["report_only"] is True and saved["immutable_epoch_snapshot"] is False
    assert saved["snapshot_purpose"] == "native_calibration_capacity"
    assert saved["training_pointer"]["value"] == json.loads(before_pointer)
    assert session._active_path.read_bytes() == before_pointer
    assert canonical_model_state_sha256(model.state_dict()) == before_model
    assert torch.equal(torch.get_rng_state(), before_rng)
    model.eval()
    with ema.evaluating(model):
        binding = bind_candidate_weight_ema_validation_checkpoint_v1(snapshot=snapshot, model=model)
    assert binding["report_only"] is True and binding["immutable_epoch_snapshot"] is False
    with pytest.raises(RuntimeError, match="REPORT_ONLY_VAL_NOT_SELECTABLE"):
        trainer._native_candidate_validation_stats({"checkpoint_binding": binding})
    state.update(checkpoint_index=2, next_batch_offset=33, global_optimizer_steps=33)
    state["weight_ema_state"]["steps"] = 33
    session.save_checkpoint(state)
    # The embedded actual TRAIN-32 pointer remains verifiable after resume advances its source.
    assert require_candidate_weight_ema_val_binding_v1(binding) == binding
    from gx1.contracts.unified_exit_random_access_checkpoint_v1 import canonical_sha256
    changed = copy.deepcopy(binding)
    changed["training_pointer"]["value"]["global_optimizer_steps"] = 31
    changed["binding_sha256"] = canonical_sha256({key: value for key, value in changed.items() if key != "binding_sha256"})
    with pytest.raises(RuntimeError, match="REPORT_ONLY_VAL_SCOPE_INVALID"):
        require_candidate_weight_ema_val_binding_v1(changed)


@pytest.mark.parametrize("field,value", [("arm", "other"), ("report_only_val", False), ("schema_version", "other")])
def test_report_only_snapshot_requires_exact_recipe_scope(tmp_path, field, value):
    calibration = {"schema_version": "gx1_native_learning_calibration_run_v1", "arm": "reference", "report_only_val": True}
    calibration[field] = value
    session, model, _, _, _ = _epoch_boundary(tmp_path, native_calibration=calibration)
    with pytest.raises(RuntimeError, match="REPORT_ONLY_VAL_SCOPE_INVALID"):
        session.save_validation_checkpoint(model=model, report_only=True)


@pytest.mark.parametrize("field,value", [("phase", "validation"), ("epoch_index", 1),
                                         ("global_optimizer_steps", 31), ("next_batch_offset", 31)])
def test_report_only_snapshot_requires_exact_saved_train32_cursor(tmp_path, field, value):
    calibration = {"schema_version": "gx1_native_learning_calibration_run_v1", "arm": "split", "report_only_val": True}
    session, model, _, _, state = _epoch_boundary(tmp_path, native_calibration=calibration)
    state[field] = value
    state["checkpoint_index"] = 2
    session.save_checkpoint(state)
    with pytest.raises(RuntimeError, match="CHECKPOINT_PHASE_INVALID"):
        session.save_validation_checkpoint(model=model, report_only=True)


@pytest.fixture
def dynamic_full_train_recipe_scope(tmp_path, monkeypatch):
    """Runner metadata unit boundary; no physical admission or model construction proof.

    Previously qualified physical/prefix owners are isolated, while file seals,
    current source-owned geometry/routes, env, typed controls and profile are real.
    """
    import numpy as np
    from gx1.scripts import run_unified_exit_random_access_full_train_v1 as runner
    from gx1.contracts import unified_exit_native_candidate_campaign_v1 as campaign
    from gx1.contracts import entry_observed_market_horizons_v1 as horizons
    from gx1.contracts.entry_observed_market_v1 import entry_observed_dynamic_m5_path_contract
    from gx1.contracts.entry_candidate_checkpoint_policy_v1 import DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR
    from gx1.contracts.entry_model_native_train_recipe_v1 import MODEL_NATIVE_RECIPE_ENV
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE as phase
    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_SEQ_LEN, MODEL_NATIVE_MANDATORY_SELECTED_FIELDS, MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
        MODEL_NATIVE_CTX_CONT_FIELDS, MODEL_NATIVE_CTX_CAT_FIELDS, ordered_model_native_signal_fields,
    )
    from gx1.contracts.entry_exit_production_architecture_v1 import PRODUCTION_MTF_PER_TF_WINDOW_BARS
    from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4
    from gx1.features.entry_specialist_feature_groups_v1 import (
        MODEL_NATIVE_TRAINING_SPECIALISTS, classify_entry_specialist_feature,
        MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT, model_native_context_temporal_alias_policy,
    )
    def write(path, value):
        path.write_text(json.dumps(value, sort_keys=True))
        return {"path": str(path), "sha256": runner.val.file_sha256(path)}
    selected = list(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS + MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS)
    signal = list(ordered_model_native_signal_fields(selected))
    mtf_names = list(MULTI_TF_PER_BAR_FEATURES_V4)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_binding = write(cache_dir / "manifest.json", {"feature_names": mtf_names, "cache_identity_sha256": "a" * 64})
    cache = {"cache_dir": str(cache_dir), "manifest_path": cache_binding["path"],
        "manifest_sha256": cache_binding["sha256"], "cache_identity_sha256": "a" * 64}
    surfaces = {"signal": {"field_names": signal}, "ctx_cont": {"field_names": list(MODEL_NATIVE_CTX_CONT_FIELDS)}}
    surfaces.update({f"mtf_{tf.lower()}": {"field_names": mtf_names} for tf, _ in PRODUCTION_MTF_PER_TF_WINDOW_BARS})
    files = {name: tmp_path / (name + ".json") for name in runner._DYNAMIC_M5_DATA_FILES}
    for name, path in files.items():
        write(path, {"fixture": name})
    normalization = {"contract": {"surfaces": surfaces, "ctx_cat": {"field_names": list(MODEL_NATIVE_CTX_CAT_FIELDS)},
        "lineage": {"mtf_cache_manifest_path": cache_binding["path"], "mtf_cache_manifest_sha256": cache_binding["sha256"],
            "per_tf_seq_lens": dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS), "dataset_run_id": "DYNAMIC_DATA",
            "m5_prebuilt_path": str(files["m5_prebuilt"]), "m5_prebuilt_sha256": runner.val.file_sha256(files["m5_prebuilt"])}}}
    write(files["entry_only_base_normalization"], normalization)
    manifest = {"feature_contract": {"ctx_cont_names": list(MODEL_NATIVE_CTX_CONT_FIELDS), "ctx_cat_names": list(MODEL_NATIVE_CTX_CAT_FIELDS)},
        "extra": {"seq_len": MODEL_NATIVE_SEQ_LEN, "model_native_signal_contract": {"fields": signal, "selected_fields": selected}, "multi_tf_cache_binding": cache}}
    write(files["entry_train_manifest"], manifest)
    routing = copy.deepcopy(MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT)
    routing["temporal_alias_policy"] = model_native_context_temporal_alias_policy(signal)
    indices = {name: [] for name in MODEL_NATIVE_TRAINING_SPECIALISTS}
    for i, name in enumerate(signal):
        indices[classify_entry_specialist_feature(name)].append(i)
    metadata = {"schema_version": "gx1_native_fresh_constructor_metadata_v1", "seq_input_dim": len(signal),
        "snap_input_dim": len(signal), "ctx_cont_dim": len(MODEL_NATIVE_CTX_CONT_FIELDS),
        "ctx_cat_dim": len(MODEL_NATIVE_CTX_CAT_FIELDS), "seq_len": MODEL_NATIVE_SEQ_LEN, "dropout": 0.05,
        "multi_tf": {"multi_tf_num_layers": 2, "multi_tf_scale": 0.5, "feature_names": mtf_names, "shared_cache_dir": cache["cache_dir"],
            "shared_cache_manifest_path": cache_binding["path"], "shared_cache_manifest_sha256": cache_binding["sha256"],
            "shared_cache_identity_sha256": cache["cache_identity_sha256"]},
        "specialist_fusion": {"num_layers": 1, "fusion_scale": 0.25, "cross_family_fusion_scale": 0.25,
            "input_indices": indices, "context_routing": routing}}
    for tf, length in PRODUCTION_MTF_PER_TF_WINDOW_BARS:
        metadata["multi_tf"].update({f"{tf.lower()}_seq_dim": len(mtf_names), f"{tf.lower()}_seq_len": length})
    write(files["source_bundle_metadata"], metadata)
    design_binding = write(tmp_path / "design.json", {"scope": {"entry_only_training_phase": phase}})
    witness_binding = write(tmp_path / "witness.json", {"fixture": "isolated physical witness boundary"})
    artifacts = {"design": design_binding, "normalization_result": {"path": str(files["entry_only_base_normalization"]),
        "sha256": runner.val.file_sha256(files["entry_only_base_normalization"])},
        "labels_result": write(tmp_path / "labels.json", {"fixture": "isolated target receipt boundary"}),
        "native_coordinates": write(tmp_path / "coordinates.json", {"fixture": "isolated source order boundary"})}
    physical = {"entry_training_phase": phase, "physical_sources": {
        split: {kind: {"path": str(files[f"entry_{split}_{kind}"]), "sha256": runner.val.file_sha256(files[f"entry_{split}_{kind}"])}
                for kind in ("parquet", "manifest")} for split in ("train", "val")},
        "entry_only_base_normalization": artifacts["normalization_result"],
        "sequence_source_audits": {split: {"path": str(files[key]), "sha256": runner.val.file_sha256(files[key])}
            for split, key in (("train", "sequence_source_audit"), ("val", "val_sequence_source_audit"))}}
    sampler = {"fixture": "unchanged canonical native sampler boundary"}
    target_contract = entry_observed_dynamic_m5_path_contract()
    physical["train_rows"] = 3
    prefix = {"artifacts": artifacts, "bindings": artifacts, "normalization": normalization, "train_rows": 3,
        "physical_preprocessing": physical, "physical_coordinates": {"native_sampler_contract": sampler,
            "population_witness": witness_binding},
        "entry_m5_path_single_epoch": {"binding": witness_binding, "train_rows": 3, "witness": {
            "training_phase": phase, "target_contract": target_contract, "native_sampler_contract": sampler}},
        "eligible_parent_rows": np.arange(3, dtype=np.int64)}
    observed_binding = write(tmp_path / "observed-entry-plan.json", {"fixture": "isolated observed scope boundary"})
    write(files["entry_measurement_coordinates"], {"schema_version": "gx1_dynamic_m5_native_coordinate_publication_v1",
        "chronological_prefix": artifacts, "population_witness": witness_binding, "native_sampler_contract": sampler,
        "native_coordinates_complete": True, "entry_measurement_complete": False,
        "native_launch_admitted": False, "test_data_used": False})
    env = {**MODEL_NATIVE_RECIPE_ENV, "GX1_V10_CKPT_MONITOR": DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR}
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    recipe = {"schema_version": runner.NATIVE_FULL_TRAIN_RECIPE_SCHEMA, "profile": "candidate",
        "source_repo": str(Path(runner.__file__).resolve().parents[2]), "source_commit": "c" * 40,
        "source_bindings": {}, "source_bindings_sha256": runner.val.canonical_sha256({}),
        "run_id": "DYNAMIC_METADATA_UNIT", "dataset_run_id": "DYNAMIC_DATA", "gx1_data_root": str(tmp_path),
        "out_bundle_dir": str(tmp_path / "out"), "files": {}, "seed_launch": None, "seed_authority": None,
        "smoke_full_val": None, "recipe_env": env, "val_limits": {"max_model_forwards": 16,
            "max_state_views": 256, "max_wall_seconds": 10800, "progress_interval_forwards": 16},
        "initialization": "fresh_existing_model_constructor_no_checkpoint_weights", "test_data_used": False,
        "next_run_policy": {}, "chronological_prefix": artifacts, "entry_m5_path_single_epoch": witness_binding,
        "entry_observed_market": observed_binding,
        "trainer_cli": {"epochs": 1, "batch_size": 16, "num_workers": 0, "grad_accum_steps": 1,
            "early_stopping_patience": 5, "early_stopping_min_delta": 0.0, "minimum_epochs_before_stop": 1,
            "save_top_k": 0, "precision_policy": "deterministic_fp32", "seed": 1337,
            "learning_rate": 0.0001, "weight_decay": 0.0001, "grad_clip_norm": float(trainer._GRAD_CLIP_NORM),
            "checkpoint_monitor": DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR}}
    calls = []
    monkeypatch.setattr(runner.val, "_assert_clean_source", lambda value: calls.append("clean"))
    monkeypatch.setattr(runner, "_native_recipe_source_bindings", lambda repo: {})
    monkeypatch.setattr(runner, "_require_component_sampler", lambda **kwargs: calls.append("sampler"))
    monkeypatch.setattr(runner, "_require_prefix_component_bindings", lambda *args, **kwargs: prefix)
    monkeypatch.setattr(campaign, "require_chronological_prefix_recipe", lambda value: prefix)
    def observed_scope(value):
        assert value["entry_observed_market"] == observed_binding
        assert runner.val._read(runner._bound_artifact(value["entry_observed_market"])) == {"fixture": "isolated observed scope boundary"}
        calls.append("observed_scope")
        return {"training_phase": phase, "physical_sources": physical["physical_sources"],
            "target_contract": target_contract, "query_minutes": tuple(target_contract["query_contract"]["horizon_minutes"])}
    monkeypatch.setattr(horizons, "require_observed_entry_scope", observed_scope)
    monkeypatch.setattr(campaign, "require_prefix_measurement_source_binding", lambda value: calls.append("measurement_source"))
    monkeypatch.setattr(runner.trainer, "_resolve_train_out_bundle_dir", lambda *args: tmp_path / "out")
    monkeypatch.setattr(runner.launch_owner, "require_training_recipe_source_provenance_metadata", lambda value, **kwargs: value)
    def seal():
        recipe["files"] = {key: {"path": str(path), "sha256": runner.val.file_sha256(path)} for key, path in files.items()}
        recipe["recipe_sha256"] = runner.val.canonical_sha256({key: value for key, value in recipe.items() if key != "recipe_sha256"})
        return write(tmp_path / "recipe.json", recipe)
    return runner, recipe, files, prefix, calls, seal


def test_dynamic_full_train_recipe_reuses_actual_metadata_owner_without_exit(dynamic_full_train_recipe_scope):
    runner, recipe, files, prefix, calls, seal = dynamic_full_train_recipe_scope
    binding = seal()
    checked, bound_files, provenance = runner._require_native_full_train_recipe(Path(binding["path"]), binding["sha256"])
    assert checked == recipe and bound_files == files
    assert calls == ["clean", "sampler", "observed_scope", "measurement_source"]
    assert provenance["source_commit"] == recipe["source_commit"]
    assert "economics_readiness" not in bound_files and "selected_sampler" not in bound_files
    assert "train_cost_authority" in bound_files


@pytest.mark.parametrize("fault", ["epochs", "topk", "monitor", "profile", "profile_type", "extra_exit_profile",
    "extra_exit_file", "missing_file", "missing_cost_file", "source", "environment", "phase", "count", "clip"])
def test_dynamic_full_train_recipe_rejects_mixed_scope(dynamic_full_train_recipe_scope, fault):
    runner, recipe, files, prefix, calls, seal = dynamic_full_train_recipe_scope
    if fault == "epochs": recipe["trainer_cli"]["epochs"] = 2
    elif fault == "topk": recipe["trainer_cli"]["save_top_k"] = 1
    elif fault == "monitor": recipe["trainer_cli"]["checkpoint_monitor"] = COUPLED_NET_CHECKPOINT_MONITOR
    elif fault == "profile": recipe["val_limits"]["max_state_views"] += 1
    elif fault == "profile_type": recipe["val_limits"]["max_model_forwards"] = 16.0
    elif fault == "extra_exit_profile": recipe["val_limits"]["policy_batch_size"] = 256
    elif fault == "extra_exit_file": files["economics_readiness"] = files["sequence_source_audit"]
    elif fault == "missing_file": files.pop("val_sequence_source_audit")
    elif fault == "missing_cost_file": files.pop("train_cost_authority")
    elif fault == "source": recipe["source_bindings_sha256"] = "d" * 64
    elif fault == "environment": recipe["recipe_env"]["GX1_V10_CKPT_MONITOR"] = COUPLED_NET_CHECKPOINT_MONITOR
    elif fault == "phase":
        path = Path(recipe["chronological_prefix"]["design"]["path"])
        path.write_text(json.dumps({"scope": {"entry_only_training_phase": "joint_fitted_q"}}))
        recipe["chronological_prefix"]["design"]["sha256"] = runner.val.file_sha256(path)
    elif fault == "count": prefix["eligible_parent_rows"] = prefix["eligible_parent_rows"][:-1]
    else: recipe["trainer_cli"]["grad_clip_norm"] *= 2.0
    binding = seal()
    with pytest.raises(RuntimeError):
        runner._require_native_full_train_recipe(Path(binding["path"]), binding["sha256"])


@pytest.mark.parametrize("fault", ["schema", "signal_width", "typed_width", "context_width", "window", "mtf_width",
    "mtf_order", "cache_identity", "signal_order", "context_order", "specialist", "context_route", "mask_field"])
def test_dynamic_constructor_metadata_rejects_stale_current_geometry(dynamic_full_train_recipe_scope, fault):
    runner, recipe, files, prefix, calls, seal = dynamic_full_train_recipe_scope
    path = files["source_bundle_metadata"]
    metadata = json.loads(path.read_text())
    if fault == "schema": metadata["schema_version"] = "legacy"
    elif fault == "signal_width": metadata["seq_input_dim"] -= 1
    elif fault == "typed_width": metadata["seq_input_dim"] = float(metadata["seq_input_dim"])
    elif fault == "context_width": metadata["ctx_cont_dim"] -= 1
    elif fault == "window": metadata["multi_tf"]["d1_seq_len"] -= 1
    elif fault == "mtf_width": metadata["multi_tf"]["h1_seq_dim"] -= 1
    elif fault == "mtf_order": metadata["multi_tf"]["feature_names"] = list(reversed(metadata["multi_tf"]["feature_names"]))
    elif fault == "cache_identity": metadata["multi_tf"]["shared_cache_identity_sha256"] = "b" * 64
    elif fault == "signal_order": prefix["normalization"]["contract"]["surfaces"]["signal"]["field_names"] = list(reversed(prefix["normalization"]["contract"]["surfaces"]["signal"]["field_names"]))
    elif fault == "context_order": prefix["normalization"]["contract"]["surfaces"]["ctx_cont"]["field_names"] = list(reversed(prefix["normalization"]["contract"]["surfaces"]["ctx_cont"]["field_names"]))
    elif fault == "specialist": next(iter(metadata["specialist_fusion"]["input_indices"].values())).pop()
    elif fault == "context_route": metadata["specialist_fusion"]["context_routing"]["ctx_cont_dim"] -= 1
    else: prefix["normalization"]["contract"]["surfaces"]["mtf_m5"]["field_names"] = prefix["normalization"]["contract"]["surfaces"]["mtf_m5"]["field_names"][:-1]
    path.write_text(json.dumps(metadata))
    binding = seal()
    with pytest.raises(RuntimeError, match="CONSTRUCTOR_|CONTEXT_SPECIALIST_ROUTING"):
        runner._require_native_full_train_recipe(Path(binding["path"]), binding["sha256"])


@pytest.mark.parametrize("artifact", ["source_bundle_metadata", "train_cost_authority"])
def test_dynamic_full_train_recipe_keeps_file_bytes_and_clean_source_guards(dynamic_full_train_recipe_scope, monkeypatch, artifact):
    runner, recipe, files, prefix, calls, seal = dynamic_full_train_recipe_scope
    binding = seal()
    files[artifact].write_text("{}")
    with pytest.raises(RuntimeError, match="ARTIFACT_BYTES_INVALID"):
        runner._require_native_full_train_recipe(Path(binding["path"]), binding["sha256"])
    binding = seal()
    def dirty(value):
        raise RuntimeError("DIRTY_SOURCE_UNIT_BOUNDARY")
    monkeypatch.setattr(runner.val, "_assert_clean_source", dirty)
    with pytest.raises(RuntimeError, match="DIRTY_SOURCE_UNIT_BOUNDARY"):
        runner._require_native_full_train_recipe(Path(binding["path"]), binding["sha256"])
