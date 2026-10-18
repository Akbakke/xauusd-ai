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


SCHEMA_VERSION = "gx1_entry_candidate_checkpoint_policy_v6"
COUPLED_NET_SCHEMA_VERSION = "gx1_entry_candidate_checkpoint_policy_v7"
MARKED_NET_SCHEMA_VERSION = "gx1_entry_candidate_checkpoint_policy_v8"
M5_SINGLE_EPOCH_SCHEMA_VERSION = "gx1_entry_candidate_checkpoint_policy_m5_single_epoch_v1"
# A validation epoch whose Entry argmax is one action for every row (all
# FLAT, all LONG or all SHORT) is a constant policy, never a selectable
# optimum (docs/LEARNING_GATE.md: constant action choice is not PASS).
# All-FLAT scores exactly 0.0 on every net monitor and would otherwise absorb
# selection whenever each trading epoch is negative; this rule removes that
# absorbing state without introducing any new numeric threshold.
CONSTANT_ACTION_CHECKPOINT_ELIGIBLE = False
CONSTANT_ACTION_INELIGIBLE_REASON = "constant_validation_entry_action_distribution"
ENTRY_ACTION_COUNT_KEYS = ("long_count", "short_count", "flat_count")
# The external candidate is permitted at most thirty complete TRAIN/VAL
# epochs. This upper bound belongs in the frozen policy so a resumed process
# cannot silently exceed the hash-bound candidate budget.
MAX_EPOCHS = 30
VALIDATION_FREQUENCY_EPOCHS = 1
EARLY_STOP_PATIENCE = 5
MINIMUM_EPOCHS_BEFORE_STOP = 1
SAVE_TOP_K = 1
EARLY_STOP_MIN_DELTA = 0.0
CHECKPOINT_MONITOR = "entry_policy_realized_gross_spread_inclusive_pnl_bps_mean"
COUPLED_NET_CHECKPOINT_MONITOR = "entry_exit_policy_metrics.mean_net_bps_per_entry"
MARKED_NET_CHECKPOINT_MONITOR = "marked_policy_evaluation.single_position_replay.net_cash_plus_open_value_bps_sum"
CHECKPOINT_MODE = "max"


def _m5_single_epoch_checkpoint_scope(prefix_training_binding):
    """Join only the declared actual-population successor, never infer it from max_epochs."""
    if prefix_training_binding is None:
        return None
    if not isinstance(prefix_training_binding, Mapping):
        raise RuntimeError("[ENTRY_CANDIDATE_M5_EPOCH_BINDING_INVALID]")
    if "entry_m5_path_single_epoch" not in prefix_training_binding:
        return None
    from pathlib import Path
    from gx1.contracts.local_random_access_campaign_v2 import require_binding, read_bound_json
    from gx1.contracts.entry_observed_market_horizons_v1 import require_m5_single_epoch_population
    binding = require_binding(prefix_training_binding.get("artifacts", {}).get("design"),
        label="M5 checkpoint actual design", verify_file=True)
    single = require_m5_single_epoch_population(read_bound_json(Path(binding["path"]), binding["sha256"]))
    if single is None:
        raise RuntimeError("[ENTRY_CANDIDATE_M5_EPOCH_BINDING_INVALID]")
    expected = {"population_witness":single["binding"], "train_rows":single["train_rows"],
        "last_batch_rows":single["last_batch_rows"], "drop_last":False}
    observed = prefix_training_binding.get("entry_observed_market", {})
    if (prefix_training_binding.get("schema_version") != "gx1_candidate_prefix_training_binding_v1"
            or prefix_training_binding["entry_m5_path_single_epoch"] != expected
            or prefix_training_binding.get("train_rows") != single["train_rows"]
            or prefix_training_binding.get("maximum_optimizer_steps") != single["optimizer_steps"]
            or prefix_training_binding.get("maximum_completed_epochs") != 1
            or prefix_training_binding.get("physical_sources", {}).get("train", {}).get("parquet") != single["train_parquet"]
            or not isinstance(observed, Mapping) or set(observed) != {"train","control"}
            or any(not isinstance(observed[role], Mapping)
                or observed[role].get("identity", {}).get("training_phase") != single["witness"]["training_phase"]
                for role in observed)):
        raise RuntimeError("[ENTRY_CANDIDATE_M5_EPOCH_BINDING_INVALID]")
    return {**expected, "optimizer_steps":single["optimizer_steps"]}


def checkpoint_policy_metadata(
    *, checkpoint_monitor: str = CHECKPOINT_MONITOR,
    prefix_training_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact external-candidate policy in JSON-safe form."""

    if checkpoint_monitor not in {
        CHECKPOINT_MONITOR, COUPLED_NET_CHECKPOINT_MONITOR, MARKED_NET_CHECKPOINT_MONITOR,
    }:
        raise RuntimeError("[ENTRY_CANDIDATE_CHECKPOINT_MONITOR_INVALID]")
    single = _m5_single_epoch_checkpoint_scope(prefix_training_binding)
    if single is not None and checkpoint_monitor not in {COUPLED_NET_CHECKPOINT_MONITOR, MARKED_NET_CHECKPOINT_MONITOR}:
        raise RuntimeError("[ENTRY_CANDIDATE_M5_EPOCH_MONITOR_INVALID]")
    return {
        "schema_version": (M5_SINGLE_EPOCH_SCHEMA_VERSION if single is not None else
            MARKED_NET_SCHEMA_VERSION if checkpoint_monitor == MARKED_NET_CHECKPOINT_MONITOR
            else COUPLED_NET_SCHEMA_VERSION if checkpoint_monitor == COUPLED_NET_CHECKPOINT_MONITOR
            else SCHEMA_VERSION
        ),
        "max_epochs": 1 if single is not None else MAX_EPOCHS,
        **({"entry_m5_path_single_epoch": single} if single is not None else {}),
        "validation_frequency_epochs": VALIDATION_FREQUENCY_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "minimum_epochs_before_stop": MINIMUM_EPOCHS_BEFORE_STOP,
        "save_top_k": SAVE_TOP_K,
        "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
        "checkpoint_monitor": checkpoint_monitor,
        "checkpoint_mode": CHECKPOINT_MODE,
        "constant_action_checkpoint_eligible": CONSTANT_ACTION_CHECKPOINT_ELIGIBLE,
    }


def validation_entry_action_counts(stats: Mapping[str, Any]) -> list[int]:
    """Read [LONG, SHORT, FLAT] argmax counts from either validation owner.

    The in-trainer ``validate`` emits ``entry_action_counts``; the native full
    VAL evaluator emits ``entry_exit_policy_metrics.{long,short,flat}_count``.
    Missing or malformed counts fail closed: selection may never assume a
    non-constant policy it has not observed.
    """

    if not isinstance(stats, Mapping):
        raise RuntimeError("[ENTRY_CANDIDATE_VALIDATION_STATS_INVALID]")
    counts: Any = stats.get("entry_action_counts")
    if counts is None:
        metrics = stats.get("entry_exit_policy_metrics")
        if isinstance(metrics, Mapping) and all(
            key in metrics for key in ENTRY_ACTION_COUNT_KEYS
        ):
            counts = [metrics[key] for key in ENTRY_ACTION_COUNT_KEYS]
    if (
        not isinstance(counts, (list, tuple))
        or len(counts) != 3
        or any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in counts)
        or sum(int(c) for c in counts) <= 0
    ):
        raise RuntimeError("[ENTRY_CANDIDATE_VALIDATION_ACTION_COUNTS_MISSING]")
    return [int(c) for c in counts]


def constant_action_checkpoint_ineligible(action_counts: Any) -> bool:
    """True when at most one action was ever chosen on the validation rows."""

    counts = validation_entry_action_counts({"entry_action_counts": action_counts})
    distinct_actions = sum(1 for count in counts if count > 0)
    return distinct_actions <= 1 and not CONSTANT_ACTION_CHECKPOINT_ELIGIBLE


def native_checkpoint_monitor(objective: Mapping[str, Any]) -> str:
    """Bind native selection to the validated economic accounting mode."""
    from gx1.contracts.unified_exit_economics_objective_v2 import (
        _require_runtime_contract, MARK_TO_MARKET_OBJECTIVE_SCHEMA_VERSION, LIQUIDATION_ADVANTAGE_SCHEMA_VERSION,
    )
    checked = _require_runtime_contract(objective)
    return (MARKED_NET_CHECKPOINT_MONITOR
            if checked["schema_version"] in {MARK_TO_MARKET_OBJECTIVE_SCHEMA_VERSION, LIQUIDATION_ADVANTAGE_SCHEMA_VERSION}
            else COUPLED_NET_CHECKPOINT_MONITOR)


def require_checkpoint_policy(
    value: Mapping[str, Any], *, context: str,
    checkpoint_monitor: str = CHECKPOINT_MONITOR,
    prefix_training_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject a recipe/session that drifts from its explicit frozen policy."""

    expected = checkpoint_policy_metadata(checkpoint_monitor=checkpoint_monitor,
        prefix_training_binding=prefix_training_binding)
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise RuntimeError(f"[{context}_CHECKPOINT_POLICY_INVALID]")
    if "entry_m5_path_single_epoch" in expected:
        from gx1.contracts.local_random_access_campaign_v2 import canonical_sha256
        if canonical_sha256(dict(value)) != canonical_sha256(expected):
            raise RuntimeError(f"[{context}_CHECKPOINT_POLICY_INVALID]")
    return expected


def checkpoint_metric(
    metrics: Mapping[str, Any], *, checkpoint_monitor: str = CHECKPOINT_MONITOR,
) -> float:
    """Read the selected metric without falling back to a different objective."""

    checkpoint_policy_metadata(checkpoint_monitor=checkpoint_monitor)
    if checkpoint_monitor == MARKED_NET_CHECKPOINT_MONITOR:
        marked = metrics.get("marked_policy_evaluation", {})
        replay = marked.get("single_position_replay", {}) if isinstance(marked, Mapping) else {}
        if not isinstance(replay, Mapping) or replay.get("full_cohort_authoritative") is not True:
            raise RuntimeError("[ENTRY_CANDIDATE_MARKED_CHECKPOINT_INCOMPLETE]")
    value: Any = metrics
    for key in checkpoint_monitor.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise RuntimeError("[ENTRY_CANDIDATE_CHECKPOINT_METRIC_MISSING]")
        value = value[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeError("[ENTRY_CANDIDATE_CHECKPOINT_METRIC_INVALID]")
    return float(value)


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
    "MARKED_NET_SCHEMA_VERSION",
    "MARKED_NET_CHECKPOINT_MONITOR",
    "native_checkpoint_monitor",
    "COUPLED_NET_SCHEMA_VERSION",
    "COUPLED_NET_CHECKPOINT_MONITOR",
    "CHECKPOINT_MODE",
    "CHECKPOINT_MONITOR",
    "EARLY_STOP_MIN_DELTA",
    "EARLY_STOP_PATIENCE",
    "MAX_EPOCHS",
    "MINIMUM_EPOCHS_BEFORE_STOP",
    "SAVE_TOP_K",
    "SCHEMA_VERSION",
    "VALIDATION_FREQUENCY_EPOCHS",
    "checkpoint_metric",
    "checkpoint_policy_metadata",
    "metric_improved",
    "require_checkpoint_policy",
    "retain_top_k",
    "should_early_stop",
]
