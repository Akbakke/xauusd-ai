"""Integrated coordinator/storage tests. Not real Entry/Exit or CUDA resume proof."""

import copy
import hashlib
import json
from types import SimpleNamespace, FunctionType

import pytest
import torch

from gx1.contracts import entry_model_native_train_launch_v1 as budget_owner
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.contracts.entry_candidate_checkpoint_policy_v1 import (
    MAX_EPOCHS,
    EARLY_STOP_PATIENCE,
    EARLY_STOP_MIN_DELTA,
    MINIMUM_EPOCHS_BEFORE_STOP,
    SAVE_TOP_K,
)
from tests.test_candidate_training_session import (
    _recipe_source_provenance,
    _validation_snapshot_fixture,
)


class Rows(torch.utils.data.Dataset):
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return torch.tensor(index, dtype=torch.int64)


class Harness:
    def __init__(self, root, *, ema=True, existing=False):
        root.mkdir(exist_ok=existing)
        self.root = root
        self.clock = 100.0
        self.batches = []
        self.validation_batches = 0
        self.pause_in_validation = False
        self.use_ema = ema
        self.artifacts = {}
        for name in ["train", "val", "m5", "lifecycle"]:
            p = root / (name + ".bin")
            p.write_bytes(name.encode())
            self.artifacts[name] = p
        self.ns = dict(vars(trainer))
        for name in [
            "candidate_execution_pause_reason",
            "require_candidate_execution_pointer",
        ]:
            self.ns[name] = getattr(budget_owner, name)
        self.ns["time"] = SimpleNamespace(monotonic=lambda: self.clock)
        owner = trainer._run_resumable_candidate_training
        actual = FunctionType(
            owner.__code__,
            self.ns,
            owner.__name__,
            owner.__defaults__,
            owner.__closure__,
        )
        actual.__kwdefaults__ = owner.__kwdefaults__
        self.ns[owner.__name__] = actual
        self.ns["train_epoch"] = self.train
        self.ns["validate"] = self.validate

    def train(self, model, teacher, loader, optimizer, device, **kwargs):
        cap = kwargs.get("session_max_optimizer_steps")
        for index, batch in enumerate(loader, 1):
            self.batches.append(batch.tolist())
            optimizer.zero_grad(set_to_none=True)
            inputs = torch.nn.functional.dropout(
                batch.float()[:, None].expand(-1, 3), p=0.25, training=True
            )
            model(inputs).square().mean().backward()
            optimizer.step()
            if kwargs["weight_ema"] is not None:
                kwargs["weight_ema"].update(model)
            final = index == len(loader)
            if (
                final
                or index % kwargs["session_checkpoint_interval_optimizer_steps"] == 0
                or (cap is not None and index >= cap)
            ):
                kwargs["session_checkpoint_hook"](
                    next_batch_offset=kwargs["session_batch_offset"] + index,
                    complete_epoch=final,
                )
            if cap is not None and index >= cap and not final:
                pytest.fail(
                    "integrated pause did not unwind after durable step ceiling"
                )
        return 0.0, {}, True

    def validate(self, model, teacher, loader, device, **kwargs):
        saved = kwargs["resume_validation_state"]
        rows = int(saved["rows"]) if saved is not None else 0
        for index, batch in enumerate(loader, 1):
            self.validation_batches += 1
            rows += len(batch)
            if self.pause_in_validation and index == 1:
                self.clock += 6000
                kwargs["validation_checkpoint_hook"](
                    next_batch_offset=kwargs["validation_batch_offset"] + index,
                    validation_snapshot=_validation_snapshot_fixture(rows=rows),
                )
                pytest.fail(
                    "integrated wall pause did not unwind validation checkpoint"
                )
        assert rows == len(loader.dataset)
        return (
            1.0,
            float("nan"),
            0.75,
            float("nan"),
            {
                "entry_policy_realized_gross_spread_inclusive_pnl_bps_mean": 2.0,
                "active_head_health_ok": True,
                "cooperation_gate_health_ok": True,
                "exit_cooperation_gate_health_ok": True,
                "unified_exit_full_trajectory_validation": {
                    "schema_version": "test",
                    "decision": "PASS",
                },
            },
        )

    def pointer(self, output):
        directory = output.parent / (
            trainer._CANDIDATE_TRAINING_SESSION_DIR_PREFIX + output.name
        )
        path = directory / trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME
        return path

    def state(self, output):
        pointer = self.pointer(output)
        active = json.loads(pointer.read_text())
        contract = json.loads(
            (pointer.parent / trainer._CANDIDATE_TRAINING_CONTRACT_FILENAME).read_text()
        )
        session = trainer._CandidateTrainingSession(
            out_bundle_dir=output, contract=contract, read_only=True
        )
        state = session.load_checkpoint()
        assert active["global_optimizer_steps"] == state["global_optimizer_steps"]
        return state

    def run(self, output, *, steps=None, epochs=None, expected_pointer=None):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        ema = trainer._WeightEma(model, 0.5) if self.use_ema else None
        budget = {
            "stop_after_optimizer_steps": steps,
            "stop_after_completed_val_epochs": epochs,
            "max_invocation_seconds": 5400,
            "expected_active_pointer_sha256": expected_pointer,
        }
        kwargs = dict(
            precision_policy=trainer.DETERMINISTIC_FP32,
            model=model,
            optimizer=optimizer,
            weight_ema=ema,
            lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=MAX_EPOCHS
            ),
            device=torch.device("cpu"),
            train_ds=Rows(20),
            val_ds=Rows(6),
            effective_train_rows=20,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
            epochs=MAX_EPOCHS,
            early_stopping_patience=EARLY_STOP_PATIENCE,
            early_stopping_min_delta=EARLY_STOP_MIN_DELTA,
            minimum_epochs_before_stop=MINIMUM_EPOCHS_BEFORE_STOP,
            save_top_k=SAVE_TOP_K,
            out_bundle_dir=output,
            gx1_data_override="",
            run_id="V46_20260825T170935Z_CANDIDATE",
            dataset_run_id="V46_20260825T170935Z",
            train_parquet=self.artifacts["train"],
            val_parquet=self.artifacts["val"],
            m5_prebuilt_path=self.artifacts["m5"],
            unified_exit_lifecycle_manifest_path=self.artifacts["lifecycle"],
            input_normalization={"contract_sha256": "c" * 64},
            seed=1337,
            grad_accum_steps=1,
            grad_clip_norm=1.0,
            weight_decay=1e-5,
            lr=0.001,
            dropout=0.0,
            seq_len=96,
            per_tf_seq_lens={"M5": 16, "M15": 64, "H1": 96, "H4": 96, "D1": 252},
            multi_tf_num_layers=2,
            specialist_num_layers=1,
            multi_tf_scale=0.5,
            specialist_fusion_scale=0.25,
            cross_family_fusion_scale=0.25,
            unified_exit_lifecycle_evidence={
                "splits": {"train": {"lifecycle_manifest_sha256": "a" * 64}},
                "root_manifest_sha256": "b" * 64,
            },
            recipe_source_provenance=_recipe_source_provenance(source_commit="1" * 40),
            execution_budget=budget,
            execution_budget_sha256="e" * 64,
            invocation_started_monotonic=self.clock,
        )
        with pytest.raises(self.ns["_CandidateExecutionPaused"]) as pause:
            self.ns["_run_resumable_candidate_training"](**kwargs)
        assert not output.exists()
        return model, pause.value.evidence


def equal_tree(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_integrated_eight_steps_equal_four_plus_new_model_and_restore(tmp_path):
    h = Harness(tmp_path / "h")
    continuous = h.root / "CONTINUOUS"
    split = h.root / "SPLIT"
    torch.manual_seed(1337)
    _, first = h.run(continuous, steps=8)
    reference = h.state(continuous)
    reference_batches = list(h.batches)
    h.batches.clear()
    torch.manual_seed(1337)
    _, paused = h.run(split, steps=4)
    assert (
        paused["global_optimizer_steps"] == 4
        and paused["reason"] == "optimizer_step_ceiling"
    )
    _, resumed = h.run(split, steps=8, expected_pointer=digest(h.pointer(split)))
    assert resumed["global_optimizer_steps"] == 8
    actual = h.state(split)
    for key in [
        "model_state",
        "target_model_state",
        "optimizer_state",
        "weight_ema_state",
        "lr_scheduler_state",
        "rng_state",
        "epoch_order",
        "training_progress",
    ]:
        equal_tree(reference[key], actual[key])
    assert h.batches == reference_batches
    assert h.validation_batches == 0


def test_integrated_completed_val_stage_stops_before_next_epoch_update(tmp_path):
    h = Harness(tmp_path / "h")
    output = h.root / "STAGED"
    torch.manual_seed(1337)
    _, paused = h.run(output, epochs=1)
    assert (
        paused["completed_val_epochs"] == 1 and paused["global_optimizer_steps"] == 10
    )
    assert (
        paused["phase"] == "train"
        and paused["epoch_index"] == 1
        and paused["next_batch_offset"] == 0
    )
    assert (
        paused["complete"] is False
        and h.validation_batches == 3
        and len(h.batches) == 10
    )
    before = h.pointer(output).read_bytes()
    _, again = h.run(output, epochs=1, expected_pointer=digest(h.pointer(output)))
    assert again["reason"] == "completed_val_epoch_ceiling"
    assert h.pointer(output).read_bytes() == before and len(h.batches) == 10


@pytest.mark.parametrize("ema", [False, True])
def test_integrated_mid_val_wall_pause_saves_raw_model_and_unwinds_ema(tmp_path, ema):
    h = Harness(tmp_path / "h", ema=ema)
    output = h.root / "MIDVAL"
    h.pause_in_validation = True
    torch.manual_seed(1337)
    model, paused = h.run(output, epochs=1)
    assert (
        paused["reason"] == "invocation_wall_limit" and paused["phase"] == "validation"
    )
    assert paused["next_batch_offset"] == 1 and paused["completed_val_epochs"] == 0
    state = h.state(output)
    equal_tree(model.state_dict(), state["model_state"])
    assert state["training_progress"]["validation_snapshot"]["rows"] == 2
    h.pause_in_validation = False
    _, resumed = h.run(output, epochs=1, expected_pointer=digest(h.pointer(output)))
    assert resumed["completed_val_epochs"] == 1 and h.validation_batches == 3


def test_failed_durable_save_propagates_instead_of_claiming_pause(tmp_path):
    h = Harness(tmp_path / "h")

    def fail_save(*_args, **_kwargs):
        raise OSError("injected durable storage failure")

    class FailingSession(trainer._CandidateTrainingSession):
        save_checkpoint = fail_save

    h.ns["_CandidateTrainingSession"] = FailingSession
    with pytest.raises(OSError, match="injected durable storage failure"):
        h.run(h.root / "FAILED", steps=4)
    assert not h.pointer(h.root / "FAILED").exists()
    assert h.batches == []


@pytest.mark.parametrize("expected", [None, "9" * 64])
def test_stale_pointer_rejected_before_checkpoint_load_or_write(tmp_path, expected):
    h = Harness(tmp_path / "h")
    output = h.root / "STAGED"
    h.run(output, steps=4)
    before = h.pointer(output).read_bytes()

    class MustNotLoad(trainer._CandidateTrainingSession):
        def load_checkpoint(self):
            pytest.fail("stale invocation must fail before tensor load")

    h.ns["_CandidateTrainingSession"] = MustNotLoad
    with pytest.raises(ValueError, match="ACTIVE_POINTER_MISMATCH"):
        h.run(output, steps=8, expected_pointer=expected)
    assert h.pointer(output).read_bytes() == before
    assert len(h.batches) == 4


def test_private_pause_receipt_is_bound_and_does_not_export(tmp_path):
    h = Harness(tmp_path / "h")
    output = h.root / "STAGED"
    _, evidence = h.run(output, steps=4)
    receipt = trainer._write_candidate_execution_pause_receipt(
        evidence, out_bundle_dir=output, gx1_data_override=""
    )
    before = receipt.read_bytes()
    assert json.loads(before) == evidence
    assert not output.exists()
    assert (
        trainer._write_candidate_execution_pause_receipt(
            evidence, out_bundle_dir=output, gx1_data_override=""
        )
        == receipt
    )
    assert receipt.read_bytes() == before
    changed = dict(evidence)
    changed["reason"] = "invocation_wall_limit"
    with pytest.raises(RuntimeError, match="RECEIPT_CONFLICT"):
        trainer._write_candidate_execution_pause_receipt(
            changed, out_bundle_dir=output, gx1_data_override=""
        )
    changed["active_pointer_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="RECEIPT_INVALID"):
        trainer._write_candidate_execution_pause_receipt(
            changed, out_bundle_dir=output, gx1_data_override=""
        )
    assert receipt.read_bytes() == before


def test_epoch_final_train_step_pause_resumes_scheduler_once(tmp_path):
    h = Harness(tmp_path / "h")
    direct = h.root / "DIRECT"
    split = h.root / "SPLIT"
    torch.manual_seed(1337)
    h.run(direct, epochs=1)
    reference = h.state(direct)
    torch.manual_seed(1337)
    _, paused = h.run(split, steps=10)
    assert paused["phase"] == "train" and paused["next_batch_offset"] == 10
    assert paused["completed_val_epochs"] == 0
    # Installed PyTorch checks process-local optimizer._opt_called, which is
    # not serialized. At an epoch-end resume it warns, but its step code still
    # uses the restored scheduler state. Assert the actual LR/state below.
    with pytest.warns(UserWarning, match="lr_scheduler.step.*before.*optimizer.step"):
        h.run(split, epochs=1, expected_pointer=digest(h.pointer(split)))
    actual = h.state(split)
    for key in [
        "model_state",
        "target_model_state",
        "optimizer_state",
        "weight_ema_state",
        "lr_scheduler_state",
        "rng_state",
        "epoch_order",
    ]:
        equal_tree(reference[key], actual[key])

    left = copy.deepcopy(reference["training_progress"])
    right = copy.deepcopy(actual["training_progress"])
    # Each top-k file embeds its own session contract and torch.save archive
    # name. Verify each declared hash and compare loaded payloads, exempting
    # only the independently verified session identity.
    for selection, output, state in [(left, direct, reference), (right, split, actual)]:
        checkpoints = selection["checkpoint_selection"]
        records = [*checkpoints["top_k_checkpoints"], checkpoints["best_checkpoint"]]
        seen = set()
        for record in records:
            if id(record) in seen:
                continue
            seen.add(id(record))
            artifact = h.pointer(output).parent / record["path"]
            assert digest(artifact) == record.pop("sha256")
            payload = torch.load(artifact, map_location="cpu", weights_only=True)
            assert (
                payload.pop("session_contract_sha256")
                == state["session_contract_sha256"]
            )
            record["verified_payload"] = payload
    equal_tree(left, right)


def test_cpu_fresh_process_resume_preserves_state_rng_and_order(tmp_path):
    import subprocess
    import sys

    h = Harness(tmp_path / "h")
    continuous = h.root / "CONTINUOUS"
    split = h.root / "SPLIT"
    torch.manual_seed(1337)
    h.run(continuous, steps=8)
    reference = h.state(continuous)
    torch.manual_seed(1337)
    h.run(split, steps=4)
    code = """from pathlib import Path
import torch, sys
from tests.test_candidate_execution_staging import Harness, digest
torch.manual_seed(97531)
h = Harness(Path(sys.argv[1]), existing=True)
output = h.root / "SPLIT"
h.run(output, steps=8, expected_pointer=digest(h.pointer(output)))
"""
    subprocess.run(
        [sys.executable, "-c", code, str(h.root)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    actual = h.state(split)
    for key in [
        "model_state",
        "target_model_state",
        "optimizer_state",
        "weight_ema_state",
        "lr_scheduler_state",
        "rng_state",
        "epoch_order",
        "training_progress",
    ]:
        equal_tree(reference[key], actual[key])
