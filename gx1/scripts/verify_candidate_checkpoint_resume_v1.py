#!/usr/bin/env python3
"""Deterministic, process-bound resume equivalence probe for candidate state.

This is deliberately a tiny FP32 checkpoint integration probe.  It exercises
the exact two-slot candidate session, optimizer, scheduler, RNG restoration and
the persisted global optimizer-step counter without training on market data.
It never reads a dataset, makes a prediction, or opens TEST.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


SCHEMA_VERSION = "gx1_candidate_checkpoint_resume_equivalence_v1"
_STEPS = 8
_HALF = _STEPS // 2


def _contract() -> dict[str, Any]:
    return {
        "schema_version": trainer._CANDIDATE_TRAINING_SESSION_SCHEMA_VERSION,
        "authority": {
            "candidate_training": True,
            "bundle": False,
            "validation": False,
            "test": False,
            "promotion": False,
            "paper": False,
            "live": False,
        },
        "purpose": "deterministic_checkpoint_resume_probe_only",
    }


def _reset_rng() -> None:
    random.seed(9137)
    np.random.seed(9137)
    torch.manual_seed(9137)


def _new_components() -> tuple[
    torch.nn.Module,
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    model = torch.nn.Linear(3, 2)
    target = copy.deepcopy(model)
    target.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_STEPS, eta_min=0.0
    )
    return model, target, optimizer, scheduler


def _step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> float:
    # RNG-generated inputs make restoration of Python/NumPy/Torch state
    # meaningful.  CPU FP32 has an exact <=1e-6 acceptance threshold.
    x = torch.randn(7, 3)
    y = torch.from_numpy(np.random.standard_normal((7, 2)).astype(np.float32))
    random_scale = 0.75 + random.random() * 0.5
    optimizer.zero_grad(set_to_none=True)
    loss = ((model(x) - y).square().mean()) * random_scale
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach().cpu())


def _state(
    session: trainer._CandidateTrainingSession,
    model: torch.nn.Module,
    target: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> dict[str, Any]:
    return {
        "schema_version": trainer._CANDIDATE_TRAINING_SESSION_SCHEMA_VERSION,
        "session_contract_sha256": session.contract_sha256,
        "checkpoint_index": 1,
        "phase": "train",
        "epoch_index": 0,
        "next_batch_offset": _HALF,
        "global_optimizer_steps": _HALF,
        "epoch_order": torch.arange(_STEPS, dtype=torch.int64),
        "model_state": model.state_dict(),
        "target_model_state": target.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "weight_ema_state": None,
        "lr_scheduler_state": scheduler.state_dict(),
        "rng_state": trainer._attended_session_rng_state(device=torch.device("cpu")),
        "training_progress": trainer._new_candidate_training_progress(),
        "complete": False,
    }


def _tensor_state_max_abs(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> float:
    if set(a) != set(b):
        raise RuntimeError("[RESUME_EQUIVALENCE_STATE_KEYS_MISMATCH]")
    return max(
        float((a[name].detach().cpu() - b[name].detach().cpu()).abs().max().item())
        for name in a
    )


def _optimizer_state_max_abs(a: Any, b: Any) -> float:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError("[RESUME_EQUIVALENCE_OPTIMIZER_SHAPE_MISMATCH]")
        return float((a.detach().cpu() - b.detach().cpu()).abs().max().item())
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a) != set(b):
            raise RuntimeError("[RESUME_EQUIVALENCE_OPTIMIZER_KEYS_MISMATCH]")
        return max((_optimizer_state_max_abs(a[key], b[key]) for key in a), default=0.0)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise RuntimeError("[RESUME_EQUIVALENCE_OPTIMIZER_LENGTH_MISMATCH]")
        return max((_optimizer_state_max_abs(x, y) for x, y in zip(a, b)), default=0.0)
    if a != b:
        raise RuntimeError("[RESUME_EQUIVALENCE_OPTIMIZER_VALUE_MISMATCH]")
    return 0.0


def _child_resume(out_bundle: Path) -> dict[str, Any]:
    session = trainer._CandidateTrainingSession(out_bundle_dir=out_bundle, contract=_contract())
    model, target, optimizer, scheduler = _new_components()
    state = session.load_checkpoint()
    if state is None:
        raise RuntimeError("[RESUME_EQUIVALENCE_CHECKPOINT_MISSING]")
    restored = trainer._restore_candidate_training_checkpoint(
        state,
        session=session,
        model=model,
        target_model=target,
        optimizer=optimizer,
        weight_ema=None,
        lr_scheduler=scheduler,
        device=torch.device("cpu"),
        dataset_rows=_STEPS,
    )
    if restored["global_optimizer_steps"] != _HALF:
        raise RuntimeError("[RESUME_EQUIVALENCE_GLOBAL_STEP_RESTORE_INVALID]")
    losses = [_step(model, optimizer, scheduler) for _ in range(_HALF, _STEPS)]
    reference = torch.linspace(-1.0, 1.0, 12, dtype=torch.float32).reshape(4, 3)
    return {
        "global_optimizer_steps": _STEPS,
        "losses": losses,
        "lr": float(optimizer.param_groups[0]["lr"]),
        "scheduler_last_epoch": int(scheduler.last_epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "prediction": model(reference).detach().cpu(),
    }


def run_equivalence() -> dict[str, Any]:
    _reset_rng()
    reference_model, _, reference_optimizer, reference_scheduler = _new_components()
    reference_losses = [
        _step(reference_model, reference_optimizer, reference_scheduler)
        for _ in range(_STEPS)
    ]
    reference_input = torch.linspace(-1.0, 1.0, 12, dtype=torch.float32).reshape(4, 3)
    reference_prediction = reference_model(reference_input).detach().cpu()

    with tempfile.TemporaryDirectory(prefix="gx1-candidate-resume-") as temporary:
        root = Path(temporary)
        out_bundle = root / "candidate_bundle_never_published"
        _reset_rng()
        model, target, optimizer, scheduler = _new_components()
        first_losses = [_step(model, optimizer, scheduler) for _ in range(_HALF)]
        session = trainer._CandidateTrainingSession(
            out_bundle_dir=out_bundle, contract=_contract()
        )
        session.save_checkpoint(_state(session, model, target, optimizer, scheduler))
        child = subprocess.run(
            [
                sys.executable,
                "-m",
                "gx1.scripts.verify_candidate_checkpoint_resume_v1",
                "--resume-child",
                "--out-bundle",
                str(out_bundle),
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
        if child.returncode != 0:
            raise RuntimeError(
                "[RESUME_EQUIVALENCE_CHILD_FAILED] " + child.stderr.strip()
            )
        resumed = torch.load(
            root / "child_result.pt", map_location="cpu", weights_only=True
        )
        # The child writes to the deterministic session parent so the parent
        # can compare exact state without entrusting hidden process memory.

    model_diff = _tensor_state_max_abs(
        reference_model.state_dict(), resumed["model_state"]
    )
    optimizer_diff = _optimizer_state_max_abs(
        reference_optimizer.state_dict(), resumed["optimizer_state"]
    )
    prediction_diff = float((reference_prediction - resumed["prediction"]).abs().max().item())
    loss_diff = max(
        abs(expected - observed)
        for expected, observed in zip(reference_losses[_HALF:], resumed["losses"], strict=True)
    )
    if (
        model_diff > 1e-6
        or optimizer_diff > 1e-6
        or prediction_diff > 1e-6
        or loss_diff > 1e-6
        or int(resumed["global_optimizer_steps"]) != _STEPS
        or int(resumed["scheduler_last_epoch"]) != int(reference_scheduler.last_epoch)
        or float(resumed["lr"]) != float(reference_optimizer.param_groups[0]["lr"])
    ):
        raise RuntimeError("[RESUME_EQUIVALENCE_FP32_TOLERANCE_EXCEEDED]")
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS",
        "test_accessed": False,
        "precision": "fp32",
        "steps_continuous": _STEPS,
        "steps_before_resume": _HALF,
        "steps_after_resume": _HALF,
        "global_optimizer_steps": _STEPS,
        "scheduler_last_epoch": int(reference_scheduler.last_epoch),
        "learning_rate": float(reference_optimizer.param_groups[0]["lr"]),
        "max_abs_model_weight_difference": model_diff,
        "max_abs_optimizer_state_difference": optimizer_diff,
        "max_abs_prediction_difference": prediction_diff,
        "max_abs_loss_difference": loss_diff,
        "tolerance": 1e-6,
        "amp_grad_scaler": "not_applicable_candidate_precision_is_deterministic_fp32",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-child", action="store_true")
    parser.add_argument("--out-bundle", type=Path)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args(argv)
    if args.resume_child:
        if args.out_bundle is None:
            parser.error("--resume-child requires --out-bundle")
        result = _child_resume(args.out_bundle.resolve())
        target = args.out_bundle.resolve().parent / "child_result.pt"
        torch.save(result, target)
        return 0
    if args.out_json is None:
        parser.error("--out-json is required for the parent equivalence probe")
    if args.out_bundle is not None:
        parser.error("--out-bundle is child-only; the parent probe uses a temporary session")
    try:
        report = run_equivalence()
    except RuntimeError as exc:
        print(f"FATAL: candidate resume equivalence failed: {exc}", file=sys.stderr)
        return 2
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
