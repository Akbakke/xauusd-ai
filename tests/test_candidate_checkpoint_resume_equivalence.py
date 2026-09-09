from __future__ import annotations

import copy
import argparse
import hashlib
import json
from pathlib import Path

import pytest
import torch

from gx1.scripts import verify_candidate_checkpoint_resume_v1 as recovery

from gx1.scripts.verify_candidate_checkpoint_resume_v1 import (
    _guard_recovery_timing,
    _migrate_exact_state_successor_checkpoint,
    _migrate_source_state_successor_checkpoint,
    _require_exact_state_successor_recipe_transition,
    _require_guard_only_recipe_transition,
    _require_finite_recovery_tensors,
    _require_source_state_successor_recipe_transition,
    _SOURCE_SUCCESSOR_EXPECTED_ADDED_ROLES,
    _SOURCE_SUCCESSOR_EXPECTED_CHANGED_ROLES,
    _SOURCE_SUCCESSOR_OLD_GROUP_IDS,
    _SOURCE_SUCCESSOR_RETIRED_PARAMETER_IDS,
    _SOURCE_SUCCESSOR_RETIRED_STATE_KEYS,
    _state_component_sha256,
    main,
    run_equivalence,
)


def test_candidate_resume_is_exact_across_a_fresh_python_process() -> None:
    report = run_equivalence()
    assert report["decision"] == "PASS"
    assert report["global_optimizer_steps"] == 8
    assert report["max_abs_model_weight_difference"] <= 1e-6
    assert report["max_abs_optimizer_state_difference"] <= 1e-6
    assert report["max_abs_prediction_difference"] <= 1e-6


def _transition() -> tuple[dict, dict]:
    original = {
        "profile": "candidate", "run_id": "same-run", "dataset_dir": "/data",
        "trainer_cli": {"batch_size": 8, "epochs": 30},
        "artifact_bindings": {"train": {"path": "/data/train", "sha256": "a"}},
        "out_bundle_dir": "/data/original", "source_commit": "old",
        "source_bindings": {
            "trainer_safety_guard": {"path": "/repo/guard", "sha256": "old", "size_bytes": 1},
            "trainer": {"path": "/repo/trainer", "sha256": "same", "size_bytes": 2},
        },
    }
    successor = copy.deepcopy(original)
    successor["source_commit"] = "new"
    successor["out_bundle_dir"] = "/data/successor"
    successor["source_bindings"]["trainer_safety_guard"]["sha256"] = "new"
    return original, successor


def test_guard_recovery_admits_only_guard_source_change() -> None:
    old, new = _transition()
    _require_guard_only_recipe_transition(old, new)
    new["source_bindings"]["trainer"]["mtime_ns"] = 123
    _require_guard_only_recipe_transition(old, new)


@pytest.mark.parametrize("change", ["model", "data", "batch", "run_id", "same_output", "closure", "guard_path", "no_repair"])
def test_guard_recovery_rejects_semantic_and_identity_changes(change: str) -> None:
    old, new = _transition()
    if change == "model":
        new["source_bindings"]["trainer"]["sha256"] = "changed"
    elif change == "data":
        new["artifact_bindings"]["train"]["sha256"] = "changed"
    elif change == "batch":
        new["trainer_cli"]["batch_size"] = 16
    elif change == "run_id":
        new["run_id"] = "different"
    elif change == "same_output":
        new["out_bundle_dir"] = old["out_bundle_dir"]
    elif change == "closure":
        del new["source_bindings"]["trainer"]
    elif change == "guard_path":
        new["source_bindings"]["trainer_safety_guard"]["path"] = "/other/guard"
    elif change == "no_repair":
        new["source_bindings"] = copy.deepcopy(old["source_bindings"])
    with pytest.raises(RuntimeError, match="GUARD_RECOVERY"):
        _require_guard_only_recipe_transition(old, new)


def test_guard_recovery_component_hash_is_bit_exact_and_typed() -> None:
    original = {"weight": torch.tensor([0.0, 1.0]), "rng": torch.tensor([3, 7], dtype=torch.uint8), "steps": 7936, "nested": (None, [True, -float("inf")])}
    assert _state_component_sha256(original) == _state_component_sha256(copy.deepcopy(original))
    changed = copy.deepcopy(original)
    changed["weight"][0] = -0.0
    assert _state_component_sha256(original) != _state_component_sha256(changed)
    changed = copy.deepcopy(original)
    changed["weight"] = changed["weight"].to(torch.float64)
    assert _state_component_sha256(original) != _state_component_sha256(changed)
    assert _state_component_sha256([1]) != _state_component_sha256((1,))
    assert _state_component_sha256(1) != _state_component_sha256(True)


def test_guard_recovery_rejects_nonfinite_learning_tensors() -> None:
    _require_finite_recovery_tensors({"online": [torch.tensor([1.0, 2.0])], "ema": None})
    for invalid in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(RuntimeError, match="NONFINITE_LEARNING_STATE"):
            _require_finite_recovery_tensors({"optimizer": {0: {"exp_avg": torch.tensor([invalid])}}})



def test_guard_recovery_contract_preserves_recipe_precision_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    same_names = (
        "seed", "batch_size", "epochs", "grad_accum_steps", "grad_clip_norm",
        "weight_decay", "dropout", "minimum_epochs_before_stop", "save_top_k",
        "seq_len", "multi_tf_num_layers", "specialist_num_layers",
        "multi_tf_scale", "specialist_fusion_scale",
        "cross_family_fusion_scale", "execution_tier",
    )
    cli = dict.fromkeys(same_names, 1)
    cli.update({
        "learning_rate": 1e-4, "early_stop_patience": 5,
        "early_stop_min_delta": 0.0, "device": "cuda",
        "precision_policy": "experimental_fp32_3090_no_uninitialized_fill",
        **{
            f"per_tf_seq_len_{name.lower()}": 16
            for name in recovery.trainer.MULTI_TF_TIMEFRAMES
        },
    })
    recipe = {
        "trainer_cli": cli,
        "artifact_bindings": {
            name: {"path": f"/data/{name}"}
            for name in (
                "train_parquet", "val_parquet", "m5_prebuilt",
                "unified_exit_lifecycle_manifest",
            )
        },
        "out_bundle_dir": "/data/out", "run_id": "run",
        "dataset_run_id": "dataset",
    }
    observed: dict[str, object] = {}

    def capture(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return dict(kwargs)

    monkeypatch.setattr(
        recovery.trainer, "_candidate_training_session_contract", capture
    )
    recovery._guard_recovery_session_contract(
        recipe, {"source": "bound"}, "a" * 64
    )
    assert observed["precision_policy"] == cli["precision_policy"]

def _source_state_successor_checkpoint() -> dict:
    model_state = {
        "active_before.weight": torch.tensor([1.0], dtype=torch.float32),
        "exit_side_embedding.weight": torch.arange(
            256,
            dtype=torch.float32,
        ).reshape(2, 128),
        "active_after.weight": torch.tensor([2.0], dtype=torch.float32),
        **{
            name: torch.tensor([float(index)], dtype=torch.float32)
            for index, name in enumerate(sorted(_SOURCE_SUCCESSOR_RETIRED_STATE_KEYS))
        },
    }
    active_parameter_ids = [
        parameter_id
        for group in _SOURCE_SUCCESSOR_OLD_GROUP_IDS
        for parameter_id in group
        if parameter_id not in _SOURCE_SUCCESSOR_RETIRED_PARAMETER_IDS
    ]
    optimizer_state = {}
    for parameter_id in active_parameter_ids:
        shape = (2, 128) if parameter_id == 603 else ()
        optimizer_state[parameter_id] = {
            "step": torch.tensor(9664.0, dtype=torch.float32),
            "exp_avg": torch.full(shape, parameter_id / 1000, dtype=torch.float32),
            "exp_avg_sq": torch.full(shape, parameter_id / 10000, dtype=torch.float32),
        }
    return {
        "schema_version": "candidate_training_session_v1",
        "session_contract_sha256": "a" * 64,
        "checkpoint_index": 152,
        "phase": "train",
        "epoch_index": 0,
        "next_batch_offset": 9664,
        "global_optimizer_steps": 9664,
        "epoch_order": torch.arange(16, dtype=torch.int64),
        "model_state": model_state,
        "target_model_state": copy.deepcopy(model_state),
        "optimizer_state": {
            "state": optimizer_state,
            "param_groups": [
                {"params": list(group), "lr": 1e-4}
                for group in _SOURCE_SUCCESSOR_OLD_GROUP_IDS
            ],
        },
        "weight_ema_state": {
            "decay": 0.999,
            "steps": 9664,
            "shadow": copy.deepcopy(model_state),
        },
        "lr_scheduler_state": {"last_epoch": 0},
        "rng_state": {"torch_cpu": torch.arange(4, dtype=torch.uint8)},
        "training_progress": {"best": None},
        "complete": False,
    }


def test_exact_state_successor_changes_only_contract_identity() -> None:
    original = _source_state_successor_checkpoint()
    before = {
        key: _state_component_sha256(value)
        for key, value in original.items()
        if key != "session_contract_sha256"
    }

    migrated, preserved = _migrate_exact_state_successor_checkpoint(
        original,
        successor_contract_sha256="b" * 64,
    )

    assert original["session_contract_sha256"] == "a" * 64
    assert migrated["session_contract_sha256"] == "b" * 64
    assert preserved == before
    assert {
        key: _state_component_sha256(value)
        for key, value in migrated.items()
        if key != "session_contract_sha256"
    } == before


def test_exact_state_successor_rejects_nonfinite_state() -> None:
    original = _source_state_successor_checkpoint()
    original["model_state"]["active_before.weight"][0] = float("nan")
    with pytest.raises(RuntimeError, match="NONFINITE_LEARNING_STATE"):
        _migrate_exact_state_successor_checkpoint(
            original,
            successor_contract_sha256="b" * 64,
        )


def test_exact_state_successor_rejects_learning_recipe_change() -> None:
    repo = Path(__file__).resolve().parents[1]
    original, successor = _transition()
    original["source_commit"] = "f47445a44b1566f0cee2bc1dc73c815d257fe6bb"
    successor["source_commit"] = recovery.subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    successor["run_id"] = "successor-run"
    successor["source_bindings"] = copy.deepcopy(original["source_bindings"])
    successor["source_bindings"]["trainer"]["sha256"] = "changed"
    _require_exact_state_successor_recipe_transition(
        original, successor, repo=repo
    )
    successor["trainer_cli"]["batch_size"] = 16
    with pytest.raises(RuntimeError, match="LEARNING_RECIPE_CHANGED"):
        _require_exact_state_successor_recipe_transition(
            original, successor, repo=repo
        )


def test_source_state_successor_preserves_active_state_and_remaps_ids() -> None:
    original = _source_state_successor_checkpoint()
    original_sha256 = _state_component_sha256(original)

    migrated = _migrate_source_state_successor_checkpoint(
        original,
        successor_contract_sha256="b" * 64,
    )

    assert _state_component_sha256(original) == original_sha256
    assert migrated["session_contract_sha256"] == "b" * 64
    assert not (_SOURCE_SUCCESSOR_RETIRED_STATE_KEYS & set(migrated["model_state"]))
    assert not (
        _SOURCE_SUCCESSOR_RETIRED_STATE_KEYS
        & set(migrated["target_model_state"])
    )
    assert not (
        _SOURCE_SUCCESSOR_RETIRED_STATE_KEYS
        & set(migrated["weight_ema_state"]["shadow"])
    )
    assert set(migrated["optimizer_state"]["state"]) == set(range(722))
    assert migrated["optimizer_state"]["param_groups"][0]["params"] == list(
        range(712)
    )
    assert migrated["optimizer_state"]["param_groups"][1]["params"] == list(
        range(712, 722)
    )
    assert torch.equal(
        migrated["optimizer_state"]["state"][579]["exp_avg"],
        original["optimizer_state"]["state"][603]["exp_avg"],
    )


@pytest.mark.parametrize(
    "corruption",
    ["retired_model_key_missing", "retired_optimizer_state", "side_moment_shape"],
)
def test_source_state_successor_rejects_ambiguous_mapping(corruption: str) -> None:
    state = _source_state_successor_checkpoint()
    if corruption == "retired_model_key_missing":
        state["model_state"].pop(next(iter(_SOURCE_SUCCESSOR_RETIRED_STATE_KEYS)))
    elif corruption == "retired_optimizer_state":
        state["optimizer_state"]["state"][579] = copy.deepcopy(
            state["optimizer_state"]["state"][578]
        )
    else:
        state["optimizer_state"]["state"][603]["exp_avg"] = torch.zeros(1)
        state["optimizer_state"]["state"][603]["exp_avg_sq"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="SOURCE_STATE_SUCCESSOR"):
        _migrate_source_state_successor_checkpoint(
            state,
            successor_contract_sha256="b" * 64,
        )


def test_source_state_successor_recipe_changes_only_run_and_source_identity() -> None:
    original, successor = _transition()
    successor["run_id"] = "successor-run"
    original["source_bindings"] = {
        role: {
            "path": f"/repo/{index}",
            "sha256": "a" * 64,
            "size_bytes": 1,
        }
        for index, role in enumerate(
            sorted(_SOURCE_SUCCESSOR_EXPECTED_CHANGED_ROLES | {"stable"})
        )
    }
    successor["source_bindings"] = copy.deepcopy(original["source_bindings"])
    for role in _SOURCE_SUCCESSOR_EXPECTED_CHANGED_ROLES:
        successor["source_bindings"][role]["sha256"] = "b" * 64
    successor["source_bindings"].update(
        {
            role: {
                "path": f"/repo/added/{index}",
                "sha256": "c" * 64,
                "size_bytes": 1,
            }
            for index, role in enumerate(
                sorted(_SOURCE_SUCCESSOR_EXPECTED_ADDED_ROLES)
            )
        }
    )

    _require_source_state_successor_recipe_transition(original, successor)

    wrong_delta = copy.deepcopy(successor)
    wrong_delta["source_bindings"]["stable"]["sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="SOURCE_STATE_SUCCESSOR_SOURCE_DELTA"):
        _require_source_state_successor_recipe_transition(original, wrong_delta)

    successor["artifact_bindings"]["train"]["sha256"] = "changed"
    with pytest.raises(RuntimeError, match="SOURCE_STATE_SUCCESSOR"):
        _require_source_state_successor_recipe_transition(original, successor)


def test_source_state_successor_adamw_one_step_parity() -> None:
    old_parameters = [
        torch.nn.Parameter(
            torch.linspace(-0.5, 0.5, 256, dtype=torch.float32).reshape(2, 128)
            if parameter_id == 603
            else torch.tensor(parameter_id / 1000, dtype=torch.float32)
        )
        for parameter_id in range(758)
    ]
    old_optimizer = torch.optim.AdamW(
        [
            {"params": old_parameters[:748], "weight_decay": 0.01},
            {"params": old_parameters[748:], "weight_decay": 0.0},
        ],
        lr=1e-4,
    )
    active_old_parameters = [
        parameter
        for parameter_id, parameter in enumerate(old_parameters)
        if parameter_id not in _SOURCE_SUCCESSOR_RETIRED_PARAMETER_IDS
    ]
    for ordinal, parameter in enumerate(active_old_parameters):
        parameter.grad = torch.full_like(parameter, (ordinal + 1) / 10000)
    old_optimizer.step()
    old_optimizer.zero_grad(set_to_none=True)

    checkpoint = _source_state_successor_checkpoint()
    checkpoint["optimizer_state"] = old_optimizer.state_dict()
    migrated = _migrate_source_state_successor_checkpoint(
        checkpoint,
        successor_contract_sha256="b" * 64,
    )
    new_parameters = [
        torch.nn.Parameter(parameter.detach().clone())
        for parameter in active_old_parameters
    ]
    new_optimizer = torch.optim.AdamW(
        [
            {"params": new_parameters[:712], "weight_decay": 0.01},
            {"params": new_parameters[712:], "weight_decay": 0.0},
        ],
        lr=1e-4,
    )
    new_optimizer.load_state_dict(migrated["optimizer_state"])

    old_loss = torch.stack(
        [parameter.square().mean() for parameter in active_old_parameters]
    ).sum()
    new_loss = torch.stack(
        [parameter.square().mean() for parameter in new_parameters]
    ).sum()
    assert torch.equal(old_loss, new_loss)
    old_loss.backward()
    new_loss.backward()
    for old_parameter, new_parameter in zip(
        active_old_parameters,
        new_parameters,
        strict=True,
    ):
        assert torch.equal(old_parameter.grad, new_parameter.grad)
    old_optimizer.step()
    new_optimizer.step()
    for old_parameter, new_parameter in zip(
        active_old_parameters,
        new_parameters,
        strict=True,
    ):
        assert torch.equal(old_parameter, new_parameter)


def _actual_next_batch_payload() -> dict:
    scales = {
        f"tf_input_scale_{timeframe}": torch.tensor(1.0)
        for timeframe in ("d1", "h1", "h4", "m15", "m5")
    }
    gradients = {"active.weight": torch.tensor([0.25]), **scales}
    model_state = {"active.weight": torch.tensor([2.0]), **scales}
    optimizer_state = {
        name: {
            "step": torch.tensor(9665.0),
            "exp_avg": value.clone(),
            "exp_avg_sq": value.square(),
        }
        for name, value in model_state.items()
    }
    return {
        "schema_version": recovery._ACTUAL_NEXT_BATCH_CHILD_SCHEMA,
        "source_commit": "old",
        "recipe": {},
        "contract": {},
        "pointer": {},
        "state": {},
        "checkpoint_index": 152,
        "global_optimizer_steps_before": 9664,
        "next_batch_offset_before": 9664,
        "next_batch_indices": [1, 2, 3, 4, 5, 6, 7, 8],
        "batch_manifest": {"seq_x": {"sha256": "a" * 64}},
        "checkpoint_cuda_rng_manifest": {"sha256": "c" * 64},
        "entry_forwards": [
            {"entry_action_q_bps": torch.tensor([[1.0, 2.0, 3.0]])},
            {"entry_action_q_bps": torch.tensor([[1.5, 2.5, 3.5]])},
        ],
        "exit": {
            "entry_representation_gradients": torch.tensor([[0.5]]),
            "stats": {"raw_loss": 1.0},
            "entry_action_q_targets": torch.tensor([[1.0, 2.0, 0.0]]),
            "entry_action_q_valid": torch.tensor([[True, True, True]]),
        },
        "joint": {
            "task_losses": {"entry_action_q": torch.tensor(0.75)},
            "joint_loss": torch.tensor(0.75),
            "stats": {"active": True},
        },
        "raw_gradients": copy.deepcopy(gradients),
        "clipped_gradients": copy.deepcopy(gradients),
        "model_state_after": copy.deepcopy(model_state),
        "target_model_state_after": copy.deepcopy(model_state),
        "optimizer_state_after": optimizer_state,
        "optimizer_groups_after": [
            {
                "lr": 0.0003,
                "weight_decay": 1e-5,
                "parameter_names": list(model_state),
            }
        ],
        "weight_ema_state_after": {
            "decay": 0.99,
            "steps": 9665,
            "shadow": copy.deepcopy(model_state),
        },
        "lr_scheduler_state_after": {"last_epoch": 0},
        "task_supervision_observed": {"entry_action_q": True},
        "task_gradient_observed": {"entry_action_q": True},
        "rng_state_after_manifest": {"torch": {"sha256": "b" * 64}},
        "cuda_started": False,
        "test_accessed": False,
    }


def test_actual_next_batch_comparison_excludes_only_reviewed_retired_state() -> None:
    original = _actual_next_batch_payload()
    successor = copy.deepcopy(original)
    original["model_state_after"]["exit_fuse.0.weight"] = torch.ones(2, 2)
    original["target_model_state_after"]["exit_fuse.0.weight"] = torch.ones(2, 2)
    original["weight_ema_state_after"]["shadow"]["exit_fuse.0.weight"] = torch.ones(2, 2)
    original["optimizer_groups_after"][0]["parameter_names"].insert(
        0, "exit_fuse.0.weight"
    )
    report = recovery._require_actual_next_batch_equivalence(original, successor)
    assert report["raw_gradients"]["tolerance_limited_to_tf_input_scales"] == [
        "tf_input_scale_d1",
        "tf_input_scale_h1",
        "tf_input_scale_h4",
        "tf_input_scale_m15",
        "tf_input_scale_m5",
    ]



def test_exact_state_next_batch_admits_only_bounded_fp32_order_noise() -> None:
    original = _actual_next_batch_payload()
    successor = copy.deepcopy(original)
    original["joint"]["stats"]["raw"] = 0.25
    successor["joint"]["stats"]["raw"] = 0.25 + 1e-8
    successor["exit"]["entry_representation_gradients"] += 1e-8
    successor["exit"]["entry_action_q_targets"] += 1e-8
    successor["joint"]["task_losses"]["entry_action_q"] += 1e-8
    successor["raw_gradients"]["active.weight"] += 1e-7
    successor["clipped_gradients"]["active.weight"] += 1e-9
    successor["optimizer_state_after"]["active.weight"]["exp_avg"] += 1e-9
    successor["weight_ema_state_after"]["shadow"][
        "tf_input_scale_m5"
    ] += 1e-11

    report = recovery._require_actual_next_batch_equivalence(
        original, successor, exact_state=True
    )

    assert report["entry_forward_0"]["max_abs_difference"] == 0.0
    assert report["raw_gradients"]["max_abs_difference"] > 0.0
    assert report["model_state_after"]["relative_l2_difference"] == 0.0


def test_exact_state_next_batch_rejects_material_numerical_change() -> None:
    original = _actual_next_batch_payload()
    successor = copy.deepcopy(original)
    successor["raw_gradients"]["active.weight"] += 1e-3

    with pytest.raises(RuntimeError, match="NUMERICAL_TOLERANCE_EXCEEDED"):
        recovery._require_actual_next_batch_equivalence(
            original, successor, exact_state=True
        )

@pytest.mark.parametrize(
    ("surface", "name"),
    [
        ("entry", "entry output"),
        ("ordinary_gradient", "ordinary gradient"),
        ("scale_gradient", "scale gradient beyond tolerance"),
        ("retired_optimizer", "retired optimizer state"),
        ("batch", "next batch identity"),
    ],
)
def test_actual_next_batch_comparison_fails_closed(
    surface: str, name: str
) -> None:
    original = _actual_next_batch_payload()
    successor = copy.deepcopy(original)
    if surface == "entry":
        successor["entry_forwards"][0]["entry_action_q_bps"][0, 0] += 1e-7
    elif surface == "ordinary_gradient":
        successor["raw_gradients"]["active.weight"] += 1e-7
    elif surface == "scale_gradient":
        successor["raw_gradients"]["tf_input_scale_m5"] += 1e-3
    elif surface == "retired_optimizer":
        original["optimizer_state_after"]["exit_fuse.0.weight"] = {
            "step": torch.tensor(9665.0),
            "exp_avg": torch.tensor(0.0),
            "exp_avg_sq": torch.tensor(0.0),
        }
    else:
        successor["next_batch_indices"][0] = 9
    with pytest.raises((RuntimeError, AssertionError), match="SOURCE_STATE|Tensor"):
        recovery._require_actual_next_batch_equivalence(original, successor)


def _incident_logs(tmp_path: Path, *, step_time: str = "19:56:30,184") -> tuple[Path, Path, dict]:
    guard = tmp_path / "guard.log"
    child = tmp_path / "trainer.log"
    guard.write_text("2026-09-05T19:56:32Z event=stop reason=guard_exit pid=601955 stage=canonical\n")
    child.write_text(
        f"2026-09-05 {step_time} [INFO] [TRAIN_STEP] batch=7936 step_done\n"
        "2026-09-05 19:56:32,468 [INFO] [CANDIDATE_TRAINING_CHECKPOINT] directory=/original checkpoint_index=125 phase=train epoch_index=0 next_batch_offset=7936 global_optimizer_steps=7936 complete=0\n"
    )
    return guard, child, {"global_optimizer_steps": 7936, "checkpoint_index": 125}


def test_guard_recovery_distinguishes_update_from_later_serialization(tmp_path: Path) -> None:
    guard, child, pointer = _incident_logs(tmp_path)
    proof = _guard_recovery_timing(guard_log=guard, trainer_log=child, pointer=pointer, session_dir=Path("/original"))
    assert proof["saved_update_precedes_guard_exit"] is True
    assert proof["telemetry_after_guard_exit_proven"] is False


@pytest.mark.parametrize("change", ["late_update", "wrong_steps", "wrong_session", "later_batch", "different_stop"])
def test_guard_recovery_rejects_unproven_incident_state(tmp_path: Path, change: str) -> None:
    guard, child, pointer = _incident_logs(tmp_path, step_time="19:56:33,000" if change == "late_update" else "19:56:30,184")
    session = Path("/wrong") if change == "wrong_session" else Path("/original")
    if change == "wrong_steps":
        pointer["global_optimizer_steps"] = 7872
    elif change == "later_batch":
        child.write_text(child.read_text() + "2026-09-05 19:56:33,000 [INFO] [TRAIN_STEP] batch=7937 begin\n")
    elif change == "different_stop":
        guard.write_text(guard.read_text().replace("reason=guard_exit", "reason=thermal_limit"))
    with pytest.raises(RuntimeError, match="GUARD_RECOVERY"):
        _guard_recovery_timing(guard_log=guard, trainer_log=child, pointer=pointer, session_dir=session)


def test_guard_recovery_cli_rejects_incomplete_or_mixed_modes() -> None:
    with pytest.raises(SystemExit) as error:
        main(["--prepare-guard-recovery"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(["--prepare-source-state-successor"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(["--prepare-guard-recovery", "--prepare-source-state-successor"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(["--original-pointer-sha256", "a" * 64, "--out-json", "/unused"])
    assert error.value.code == 2


def test_guard_recovery_publishes_exact_copy_without_mutating_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mechanical CPU transfer proof; mocked recipe/Git gates are not data evidence."""
    from gx1.contracts import entry_model_native_pretest_technical_recipe_v1 as recipes
    from gx1.contracts import entry_model_native_train_launch_v1 as launch
    from tests.test_candidate_training_session import _contract, _state

    repo = Path(recovery.__file__).resolve().parents[2]
    old, new = _transition()
    for recipe, label in ((old, "old"), (new, "new")):
        recipe["dataset_run_id"] = "dataset"
        recipe["out_bundle_dir"] = str(tmp_path / label)
        recipe["source_bindings"] = {"trainer_safety_guard": {
            "path": str(repo / "scripts/gx1_guarded_trainer_exec.sh"),
            "sha256": hashlib.sha256(label.encode()).hexdigest(), "size_bytes": 3,
        }}
    old_path, new_path = tmp_path / "old.json", tmp_path / "new.json"
    old_path.write_text(json.dumps(old))
    new_path.write_text(json.dumps(new))

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    old_provenance = {"recipe_audit_path": str(old_path), "recipe_audit_sha256": sha(old_path), "source_bindings": old["source_bindings"]}
    new_provenance = {"recipe_audit_path": str(new_path), "recipe_audit_sha256": sha(new_path), "source_commit": "new", "source_bindings": new["source_bindings"]}
    contract = {**_contract(), "recipe_source_provenance": old_provenance,
                "run_id": old["run_id"], "source_commit": "old",
                "out_bundle_dir": old["out_bundle_dir"], "input_normalization_sha256": "a" * 64}
    trainer = recovery.trainer
    original = trainer._CandidateTrainingSession(out_bundle_dir=tmp_path / "old", contract=contract)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    target = copy.deepcopy(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    ema = trainer._WeightEma(model, 0.5)
    original.save_checkpoint(_state(original, model, target, optimizer, ema, scheduler))
    original_files = {path.name: path.read_bytes() for path in original.directory.iterdir()}
    pointer_path = original.directory / trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME
    guard, child, _ = _incident_logs(tmp_path)
    child.write_text(child.read_text().replace("batch=7936", "batch=17").replace("checkpoint_index=125", "checkpoint_index=1").replace("=7936", "=17").replace("directory=/original", f"directory={original.directory}"))
    monkeypatch.setattr(recipes, "require_pretest_technical_recipe_metadata", lambda value, **kwargs: value)
    monkeypatch.setattr(launch, "require_training_recipe_source_provenance", lambda **kwargs: new_provenance)
    monkeypatch.setattr(recovery.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(recovery.subprocess, "check_output", lambda command, **kwargs: "a" * 40 if "rev-parse" in command else (b"old" if command[-1].startswith("old:") else b"new"))
    monkeypatch.setattr(recovery, "_guard_recovery_session_contract", lambda recipe, provenance, normalization: {
        **contract, "out_bundle_dir": recipe["out_bundle_dir"],
        "source_commit": recipe["source_commit"], "recipe_source_provenance": provenance,
    })
    args = argparse.Namespace(
        original_recipe_json=old_path, original_recipe_sha256=sha(old_path),
        successor_recipe_json=new_path, successor_recipe_sha256=sha(new_path),
        original_pointer_sha256=sha(pointer_path), incident_guard_log=guard,
        incident_trainer_log=child, guard_repair_commit="a" * 40, out_dir=tmp_path / "reports",
    )
    report = recovery.prepare_guard_recovery(args)
    assert report["decision"] == "PASS_EXACT_STATE_TRANSFER_NOT_CUDA_AUTHORITY"
    assert {path.name: path.read_bytes() for path in original.directory.iterdir()} == original_files
    successor_dir = Path(report["successor_session_dir"])
    pointer = json.loads((successor_dir / trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME).read_text())
    assert pointer["global_optimizer_steps"] == 17
    assert pointer["checkpoint_index"] == 1
    assert pointer["session_contract_sha256"] != original.contract_sha256
    origin = json.loads((successor_dir / "CANDIDATE_GUARD_RECOVERY_ORIGIN.json").read_text())
    assert sha(Path(origin["path"])) == origin["sha256"]
    assert not (tmp_path / "new").exists()
    with pytest.raises(RuntimeError, match="SUCCESSOR_ALREADY_EXISTS"):
        recovery.prepare_guard_recovery(args)
    assert {path.name: path.read_bytes() for path in original.directory.iterdir()} == original_files
