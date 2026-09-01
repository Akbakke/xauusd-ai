#!/usr/bin/env python3
"""Executable one-epoch candidate checkpoint-policy proof."""
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
    # The full candidate is terminal after its first complete TRAIN/VAL epoch.
    # A new process may resume *within* that epoch, but it must never advance
    # into a second one under this recipe family.
    metrics = [1.0]
    uninterrupted = _advance(metrics)
    policy = checkpoint_policy_metadata()
    if (
        policy["max_epochs"] != 1
        or policy["minimum_epochs_before_stop"] != 1
        or policy["save_top_k"] != 1
        or uninterrupted["best_epoch"] != 1
        or uninterrupted["last_epoch"] != 1
        or uninterrupted["stopped"]
        or [row["epoch"] for row in uninterrupted["records"]] != [1]
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
        "terminal_epoch": 1,
        "terminal_reason": "max_epochs_one_after_full_validation",
        "early_stop": False,
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
