"""Frozen checkpoint-selection policy for the external Entry candidate.

This module owns *selection mechanics*, not model behaviour.  In particular it
does not alter features, labels, architecture, the optimizer or any task loss.
Keeping the policy as a small pure contract lets the trainer, resume state and
remote recipe share one fail-closed definition of ``best``, ``last`` and the
three retained checkpoint records.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "gx1_entry_candidate_checkpoint_policy_v2"
# The external candidate objective is one complete full-TRAIN pass followed by
# one complete VAL pass. Keeping that terminal boundary in the frozen policy
# prevents a resumed process from silently spending a second 31,004-batch
# epoch after the requested candidate result has been produced.
MAX_EPOCHS = 1
VALIDATION_FREQUENCY_EPOCHS = 1
EARLY_STOP_PATIENCE = 5
MINIMUM_EPOCHS_BEFORE_STOP = 1
SAVE_TOP_K = 1
EARLY_STOP_MIN_DELTA = 0.0
CHECKPOINT_MONITOR = "entry_policy_realized_gross_spread_inclusive_pnl_bps_mean"
CHECKPOINT_MODE = "max"


def checkpoint_policy_metadata() -> dict[str, Any]:
    """Return the exact external-candidate policy in JSON-safe form."""

    return {
        "schema_version": SCHEMA_VERSION,
        "max_epochs": MAX_EPOCHS,
        "validation_frequency_epochs": VALIDATION_FREQUENCY_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "minimum_epochs_before_stop": MINIMUM_EPOCHS_BEFORE_STOP,
        "save_top_k": SAVE_TOP_K,
        "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        "checkpoint_monitor": CHECKPOINT_MONITOR,
        "checkpoint_mode": CHECKPOINT_MODE,
    }


def require_checkpoint_policy(
    value: Mapping[str, Any], *, context: str
) -> dict[str, Any]:
    """Reject a recipe/session that drifts from the frozen candidate policy."""

    expected = checkpoint_policy_metadata()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise RuntimeError(f"[{context}_CHECKPOINT_POLICY_INVALID]")
    return expected


def metric_improved(*, candidate: float, best: float, min_delta: float) -> bool:
    """Apply the sole MAX monitor with a finite-value requirement."""

    if (
        not math.isfinite(float(candidate))
        or not math.isfinite(float(best))
        or not math.isfinite(float(min_delta))
        or float(min_delta) < 0.0
    ):
        return False
    return (float(candidate) - float(best)) > float(min_delta)


def should_early_stop(
    *,
    completed_epochs: int,
    epochs_since_improve: int,
    patience: int,
    minimum_epochs_before_stop: int,
) -> bool:
    """Return whether a completed validation may terminate the candidate."""

    if (
        isinstance(completed_epochs, bool)
        or isinstance(epochs_since_improve, bool)
        or isinstance(patience, bool)
        or isinstance(minimum_epochs_before_stop, bool)
        or int(completed_epochs) < 0
        or int(epochs_since_improve) < 0
        or int(patience) < 1
        or int(minimum_epochs_before_stop) < 1
    ):
        raise RuntimeError("[ENTRY_CANDIDATE_EARLY_STOP_ARGUMENT_INVALID]")
    return (
        int(completed_epochs) >= int(minimum_epochs_before_stop)
        and int(epochs_since_improve) >= int(patience)
    )


def retain_top_k(
    records: Sequence[Mapping[str, Any]], *, top_k: int
) -> list[dict[str, Any]]:
    """Keep the highest finite metrics, deterministically breaking ties by epoch.

    Records deliberately contain only immutable checkpoint identity metadata;
    their tensor payloads live in separately hash-bound files owned by the
    candidate session.
    """

    if isinstance(top_k, bool) or int(top_k) < 1:
        raise RuntimeError("[ENTRY_CANDIDATE_TOP_K_INVALID]")
    normalized: list[dict[str, Any]] = []
    seen_epochs: set[int] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise RuntimeError("[ENTRY_CANDIDATE_TOP_K_RECORD_INVALID]")
        record = dict(raw)
        if set(record) != {"epoch", "metric", "path", "sha256"}:
            raise RuntimeError("[ENTRY_CANDIDATE_TOP_K_RECORD_INVALID]")
        epoch = record["epoch"]
        metric = record["metric"]
        path = record["path"]
        sha256 = record["sha256"]
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 1
            or epoch in seen_epochs
            or not isinstance(metric, (int, float))
            or isinstance(metric, bool)
            or not math.isfinite(float(metric))
            or not isinstance(path, str)
            or not path
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in sha256)
        ):
            raise RuntimeError("[ENTRY_CANDIDATE_TOP_K_RECORD_INVALID]")
        seen_epochs.add(epoch)
        normalized.append(
            {
                "epoch": int(epoch),
                "metric": float(metric),
                "path": path,
                "sha256": sha256,
            }
        )
    return sorted(normalized, key=lambda item: (-item["metric"], item["epoch"]))[
        : int(top_k)
    ]


__all__ = [
    "CHECKPOINT_MODE",
    "CHECKPOINT_MONITOR",
    "EARLY_STOP_MIN_DELTA",
    "EARLY_STOP_PATIENCE",
    "MAX_EPOCHS",
    "MINIMUM_EPOCHS_BEFORE_STOP",
    "SAVE_TOP_K",
    "SCHEMA_VERSION",
    "VALIDATION_FREQUENCY_EPOCHS",
    "checkpoint_policy_metadata",
    "metric_improved",
    "require_checkpoint_policy",
    "retain_top_k",
    "should_early_stop",
]
