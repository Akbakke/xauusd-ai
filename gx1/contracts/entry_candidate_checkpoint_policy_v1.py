"""Frozen checkpoint-selection policy for the external Entry candidate.

This module owns *selection mechanics*, not model behaviour.  In particular it
does not alter features, labels, architecture, the optimizer or any task loss.
Keeping the policy as a small pure contract lets the trainer, resume state and
remote recipe share one fail-closed definition of ``best``, ``last`` and the
three retained checkpoint records.
"""
from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "gx1_entry_candidate_checkpoint_policy_v6"
COUPLED_NET_SCHEMA_VERSION = "gx1_entry_candidate_checkpoint_policy_v7"
MARKED_NET_SCHEMA_VERSION = "gx1_entry_candidate_checkpoint_policy_v8"
M5_SINGLE_EPOCH_SCHEMA_VERSION = "gx1_entry_candidate_checkpoint_policy_m5_single_epoch_v1"
DYNAMIC_M5_FIXED_ONLINE_SCHEMA_VERSION = "gx1_entry_candidate_checkpoint_policy_dynamic_m5_fixed_online_v1"
DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR = "fixed_final_online_single_epoch_no_validation_selection"
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
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    dynamic_prefix = prefix_training_binding.get("entry_training_phase") == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    if "entry_m5_path_single_epoch" not in prefix_training_binding:
        if dynamic_prefix:
            raise RuntimeError("[ENTRY_CANDIDATE_DYNAMIC_M5_EPOCH_BINDING_REQUIRED]")
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
                or not isinstance(observed[role].get("identity"), Mapping)
                or observed[role]["identity"].get("training_phase") != single["witness"]["training_phase"]
                for role in observed)):
        raise RuntimeError("[ENTRY_CANDIDATE_M5_EPOCH_BINDING_INVALID]")
    dynamic_witness = single["witness"]["training_phase"] == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    if dynamic_prefix or dynamic_witness:
        if (not dynamic_prefix or not dynamic_witness
                or prefix_training_binding.get("fixed_final_model_variant") != "ONLINE"
                or prefix_training_binding.get("physical_sources", {}).get("train", {}).get("manifest") != single["train_manifest"]
                or any(observed[role]["identity"].get("target_contract") != single["witness"]["target_contract"]
                    for role in observed)):
            raise RuntimeError("[ENTRY_CANDIDATE_DYNAMIC_M5_EPOCH_BINDING_INVALID]")
        from gx1.contracts.entry_observed_market_v1 import (
            require_entry_observed_market_scope, entry_observed_market_identity,
        )
        identity = observed["train"]["identity"]
        if observed["control"]["identity"] != identity:
            raise RuntimeError("[ENTRY_CANDIDATE_DYNAMIC_M5_OBSERVED_IDENTITY_MISMATCH]")
        observed_binding = require_binding(identity.get("design"),
            label="dynamic checkpoint observed plan", verify_file=True)
        observed_plan = read_bound_json(Path(observed_binding["path"]), observed_binding["sha256"])
        observed_prefix, observed_files = observed_plan.get("chronological_prefix"), observed_plan.get("files")
        if (not isinstance(observed_prefix, Mapping) or not isinstance(observed_files, Mapping)
                or observed_prefix.get("design") != binding
                or observed_prefix != prefix_training_binding.get("artifacts")
                or observed_prefix.get("normalization_result") != prefix_training_binding.get("entry_only_base_normalization")):
            raise RuntimeError("[ENTRY_CANDIDATE_DYNAMIC_M5_DESIGN_JOIN_INVALID]")
        sources = prefix_training_binding["physical_sources"]
        scope = require_entry_observed_market_scope({
            "entry_observed_market": observed_binding, "chronological_prefix": dict(observed_prefix),
            "files": {**{f"entry_{split}_{kind}": sources[split].get(kind)
                          for split in ("train", "val") for kind in ("parquet", "manifest")},
                      "train_cost_authority": observed_files.get("train_cost_authority")},
        })
        if (entry_observed_market_identity(scope) != identity
                or scope.get("physical_sources") != sources):
            raise RuntimeError("[ENTRY_CANDIDATE_DYNAMIC_M5_OBSERVED_SCOPE_INVALID]")
    return {**expected, "optimizer_steps":single["optimizer_steps"]}


_VERIFIED_CHECKPOINT_POLICY_SEAL = object()


class _VerifiedCheckpointPolicy:
    """Session-only completed policy proof; never a serialized resume authority."""

    __slots__ = ("_contract_sha256", "_prefix_sha256", "_monitor", "_policy_json", "_snapshot")

    def __init__(self, *, contract_sha256, prefix_sha256, monitor, policy_json, snapshot, _seal):
        if _seal is not _VERIFIED_CHECKPOINT_POLICY_SEAL:
            raise RuntimeError("[ENTRY_CANDIDATE_VERIFIED_CHECKPOINT_CONTEXT_INVALID]")
        for name, value in (("_contract_sha256", contract_sha256), ("_prefix_sha256", prefix_sha256),
                            ("_monitor", monitor), ("_policy_json", policy_json), ("_snapshot", snapshot)):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("verified checkpoint context is immutable")

    def __reduce_ex__(self, protocol):
        raise TypeError("verified checkpoint context is runtime-only")

    def require_contract(self, contract: Mapping[str, Any]) -> None:
        from gx1.contracts.local_random_access_campaign_v2 import canonical_sha256
        try:
            same = isinstance(contract, Mapping) and canonical_sha256(dict(contract)) == self._contract_sha256
        except (TypeError, ValueError):
            same = False
        if not same:
            raise RuntimeError("[ENTRY_CANDIDATE_VERIFIED_CHECKPOINT_CONTRACT_MISMATCH]")
        self.metadata(checkpoint_monitor=contract["training"]["checkpoint_policy"]["checkpoint_monitor"],
            prefix_training_binding=contract["chronological_prefix"])
        if _policy_json(contract["training"]["checkpoint_policy"]) != self._policy_json:
            raise RuntimeError("[ENTRY_CANDIDATE_VERIFIED_CHECKPOINT_POLICY_MISMATCH]")

    def metadata(self, *, checkpoint_monitor, prefix_training_binding) -> dict[str, Any]:
        from gx1.contracts.local_random_access_campaign_v2 import canonical_sha256
        try:
            same_prefix = (isinstance(prefix_training_binding, Mapping)
                and canonical_sha256(dict(prefix_training_binding)) == self._prefix_sha256)
        except (TypeError, ValueError):
            same_prefix = False
        if checkpoint_monitor != self._monitor or not same_prefix:
            raise RuntimeError("[ENTRY_CANDIDATE_VERIFIED_CHECKPOINT_BINDING_MISMATCH]")
        self._snapshot.require_unchanged()
        return json.loads(self._policy_json)


def _policy_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def prepare_verified_checkpoint_policy(*, contract: Mapping[str, Any]) -> _VerifiedCheckpointPolicy:
    """Physically validate one dynamic Session's exact immutable policy inputs."""
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    from gx1.contracts.local_random_access_campaign_v2 import (
        canonical_sha256, fresh_file_verification_scope, capture_verified_file_snapshot,
    )
    try:
        # Use a detached JSON value during validation; reject caller mutation at
        # the closing contract join rather than sealing a mixture of versions.
        if not isinstance(contract, Mapping):
            raise TypeError
        frozen = json.loads(_policy_json(dict(contract)))
        prefix = frozen["chronological_prefix"]
        declared = frozen["training"]["checkpoint_policy"]
        monitor = declared["checkpoint_monitor"]
        if (frozen.get("schema_version") != "gx1_candidate_training_session_v1"
                or not isinstance(prefix, dict)
                or prefix.get("entry_training_phase") != ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("[ENTRY_CANDIDATE_VERIFIED_CHECKPOINT_CONTRACT_INVALID]") from None
    with fresh_file_verification_scope():
        # Default semantic path remains the sole owner of witness/receipt/cost
        # and complete pointer validation. This call runs exactly once here.
        expected = checkpoint_policy_metadata(checkpoint_monitor=monitor, prefix_training_binding=prefix)
        if declared != expected or _policy_json(declared) != _policy_json(expected):
            raise RuntimeError("[ENTRY_CANDIDATE_VERIFIED_CHECKPOINT_POLICY_MISMATCH]")
        snapshot = capture_verified_file_snapshot()
        snapshot.require_unchanged()
        verified = _VerifiedCheckpointPolicy(contract_sha256=canonical_sha256(frozen),
            prefix_sha256=canonical_sha256(prefix), monitor=monitor, policy_json=_policy_json(expected),
            snapshot=snapshot, _seal=_VERIFIED_CHECKPOINT_POLICY_SEAL)
        verified.require_contract(contract)
    return verified


def checkpoint_policy_metadata(
    *, checkpoint_monitor: str = CHECKPOINT_MONITOR,
    prefix_training_binding: Mapping[str, Any] | None = None,
    verified_context: Any = None,
) -> dict[str, Any]:
    """Return the exact external-candidate policy in JSON-safe form."""

    if verified_context is not None:
        if not isinstance(verified_context, _VerifiedCheckpointPolicy):
            raise RuntimeError("[ENTRY_CANDIDATE_VERIFIED_CHECKPOINT_CONTEXT_INVALID]")
        return verified_context.metadata(
            checkpoint_monitor=checkpoint_monitor, prefix_training_binding=prefix_training_binding,
        )
    from gx1.contracts.local_random_access_campaign_v2 import file_verification_scope
    with file_verification_scope():
        if checkpoint_monitor not in {
            CHECKPOINT_MONITOR, COUPLED_NET_CHECKPOINT_MONITOR, MARKED_NET_CHECKPOINT_MONITOR,
            DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR,
        }:
            raise RuntimeError("[ENTRY_CANDIDATE_CHECKPOINT_MONITOR_INVALID]")
        single = _m5_single_epoch_checkpoint_scope(prefix_training_binding)
        from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
        dynamic = (isinstance(prefix_training_binding, Mapping)
                   and prefix_training_binding.get("entry_training_phase") == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
        if dynamic:
            if single is None or checkpoint_monitor != DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR:
                raise RuntimeError("[ENTRY_CANDIDATE_DYNAMIC_M5_FIXED_MONITOR_REQUIRED]")
            return {
                "schema_version": DYNAMIC_M5_FIXED_ONLINE_SCHEMA_VERSION,
                "checkpoint_monitor": DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR,
                "checkpoint_mode": "fixed_final_online", "max_epochs": 1,
                "entry_training_phase": ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
                "entry_m5_path_single_epoch": single, "selected_model_variant": "ONLINE",
                "selection_boundary": "complete_one_whole_physical_train_epoch",
                "validation_selection": False, "economic_selection": False,
                "early_stopping": False, "save_top_k": 0, "native_training_admitted": False,
            }
        if checkpoint_monitor == DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR:
            raise RuntimeError("[ENTRY_CANDIDATE_DYNAMIC_M5_FIXED_MONITOR_REQUIRES_BOUND_PREFIX]")
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
    verified_context: Any = None,
) -> dict[str, Any]:
    """Reject a recipe/session that drifts from its explicit frozen policy."""

    expected = checkpoint_policy_metadata(checkpoint_monitor=checkpoint_monitor,
        prefix_training_binding=prefix_training_binding, verified_context=verified_context)
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
    """Read a selectable metric; fixed-final ONLINE has no ranking metric."""

    if checkpoint_monitor == DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR:
        raise RuntimeError("[ENTRY_CANDIDATE_DYNAMIC_M5_FIXED_FINAL_NO_METRIC]")
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
    "DYNAMIC_M5_FIXED_ONLINE_SCHEMA_VERSION",
    "DYNAMIC_M5_FIXED_ONLINE_CHECKPOINT_MONITOR",
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
    "prepare_verified_checkpoint_policy",
    "metric_improved",
    "require_checkpoint_policy",
    "retain_top_k",
    "should_early_stop",
]
