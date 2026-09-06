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
    _require_guard_only_recipe_transition,
    _require_finite_recovery_tensors,
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
