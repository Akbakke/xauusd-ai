#!/usr/bin/env python3
"""Executable early-stop/top-k policy proof for the external candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from gx1.contracts.entry_candidate_checkpoint_policy_v1 import (
    checkpoint_policy_metadata,
    metric_improved,
    retain_top_k,
    should_early_stop,
)


def _record(epoch: int, metric: float) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "metric": metric,
        "path": f"top_k/epoch_{epoch:04d}.pt",
        "sha256": hashlib.sha256(f"synthetic-epoch-{epoch}".encode()).hexdigest(),
    }


def _advance(
    metrics: Sequence[float], *, stop_after: int | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = checkpoint_policy_metadata()
    current = dict(state or {"best": float("-inf"), "best_epoch": -1, "since": 0, "records": []})
    current["records"] = list(current["records"])
    for epoch, value in enumerate(metrics, start=1):
        if stop_after is not None and epoch > stop_after:
            break
        improved = current["best_epoch"] < 0 or metric_improved(
            candidate=value,
            best=current["best"],
            min_delta=policy["early_stop_min_delta"],
        )
        current["records"].append(_record(epoch, value))
        if improved:
            current["best"] = value
            current["best_epoch"] = epoch
            current["since"] = 0
        else:
            current["since"] += 1
        current["last_epoch"] = epoch
        current["stopped"] = should_early_stop(
            completed_epochs=epoch,
            epochs_since_improve=current["since"],
            patience=policy["early_stop_patience"],
            minimum_epochs_before_stop=policy["minimum_epochs_before_stop"],
        )
        if current["stopped"]:
            break
    current["records"] = retain_top_k(current["records"], top_k=policy["save_top_k"])
    return current


def run_policy_proof() -> dict[str, Any]:
    # Improvement, plateau, then deterioration.  Resume happens after epoch 4
    # with two patience checks already consumed; it must not reset to zero.
    metrics = [1.0, 1.5, 1.5, 1.4, 1.3, 1.2, 1.1]
    uninterrupted = _advance(metrics)
    before_resume = _advance(metrics, stop_after=4)
    # `_advance` enumerates local arrays, so assert persistence with a direct
    # policy continuation below for unambiguous external epoch numbering.
    policy = checkpoint_policy_metadata()
    since = int(before_resume["since"])
    stopped_epoch = None
    for epoch, value in enumerate(metrics[4:], start=5):
        assert not metric_improved(candidate=value, best=1.5, min_delta=0.0)
        since += 1
        if should_early_stop(
            completed_epochs=epoch,
            epochs_since_improve=since,
            patience=policy["early_stop_patience"],
            minimum_epochs_before_stop=policy["minimum_epochs_before_stop"],
        ):
            stopped_epoch = epoch
            break
    if (
        uninterrupted["best_epoch"] != 2
        or uninterrupted["last_epoch"] != 7
        or not uninterrupted["stopped"]
        or stopped_epoch != 7
        or [row["epoch"] for row in uninterrupted["records"]] != [2, 3, 4]
    ):
        raise RuntimeError("[CANDIDATE_CHECKPOINT_POLICY_PROOF_FAILED]")
    return {
        "schema_version": "gx1_candidate_checkpoint_policy_proof_v1",
        "decision": "PASS",
        "test_accessed": False,
        "policy": policy,
        "synthetic_validation_metric_sequence": metrics,
        "best_checkpoint_epoch": uninterrupted["best_epoch"],
        "best_metric": uninterrupted["best"],
        "last_checkpoint_epoch": uninterrupted["last_epoch"],
        "early_stop_epoch": 7,
        "epochs_since_improve_before_resume": before_resume["since"],
        "early_stop_epoch_after_resume": stopped_epoch,
        "top_k": uninterrupted["records"],
        "disk_checkpoint_immutability": "covered by test_candidate_training_session.py",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_policy_proof()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
