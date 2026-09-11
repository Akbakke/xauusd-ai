"""Restartable full-cohort VAL evaluation for random-access Exit v2."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from gx1.contracts.unified_exit_entry_policy_evaluation_v1 import (
    require_entry_policy_decisions, coupled_entry_exit_policy_metrics,
)

from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.unified_exit_random_access_model_v1 import (
    RANDOM_ACCESS_MODEL_SCHEMA_SHA256,
    RANDOM_ACCESS_MODEL_SCHEMA_VERSION,
)
from gx1.contracts.unified_exit_random_access_training_v1 import (
    collate_random_access_states_v1,
)
from gx1.contracts.unified_exit_random_access_val_checkpoint_v1 import (
    require_selected_weight_ema_checkpoint_binding_v1,
)
from gx1.contracts.unified_exit_random_access_val_rollout_v1 import (
    VAL_ENTRY_COHORT_SIZE,
    RandomAccessValRolloutAdapterV1,
    require_random_access_val_rollout_contract,
    unique_active_exit_actions,
)

PROGRESS_SCHEMA_VERSION = "gx1_unified_exit_random_access_val_progress_v1"
RESULT_SCHEMA_VERSION = "gx1_unified_exit_random_access_val_evaluation_v2"
PAUSE_SCHEMA_VERSION = "gx1_unified_exit_random_access_val_pause_v1"
_ROUTE_KEYS = (
    "exit_specialist_gate",
    "exit_tf_gate",
    "exit_family_tf_cooperation_gate",
    "exit_family_tf_feature_gate",
)
_OPEN = "OPEN"
_ZERO_SHA = "0" * 64


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise RuntimeError("UNIFIED_EXIT_VAL_ENTRY_REPRESENTATION_INVALID")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any], *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            os.unlink(temporary)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _new_trade() -> dict[str, Any]:
    return {
        "status": _OPEN,
        "decision_count": 0,
        "hold_count": 0,
        "hold_wall_clock_seconds": 0,
        "undiscounted_net_cash_pnl_bps": 0.0,
        "undiscounted_risk_utility_penalty_bps": 0.0,
        "discounted_risk_adjusted_utility_bps": 0.0,
        "continuation_discount": 1.0,
        "economic_slice_count": 0,
        "economic_slice_stream_sha256": _ZERO_SHA,
        "exit_state_index": None,
        "exit_decision_time_ns": None,
    }


def _new_q_diagnostics() -> dict[str, Any]:
    return {
        "active_side_decision_count": 0,
        "q_cell_count": 0,
        "q_sum": 0.0,
        "q_sumsq": 0.0,
        "q_min": None,
        "q_max": None,
        "margin_sum": 0.0,
        "margin_sumsq": 0.0,
        "hold_action_count": 0,
        "exit_now_action_count": 0,
        "hold_action_count_by_side": [0, 0],
        "exit_now_action_count_by_side": [0, 0],
    }


def _new_progress(
    *,
    contract_sha256: str,
    checkpoint_binding_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "decision": "IN_PROGRESS",
        "contract_sha256": contract_sha256,
        "checkpoint_binding_sha256": checkpoint_binding_sha256,
        "next_state_index": 0,
        "next_entry_scan_position": 0,
        "model_forward_count": 0,
        "materialized_state_view_count": 0,
        "completed_invocation_count": 0,
        "elapsed_compute_seconds": 0.0,
        "active_side_mask": [[True, True] for _ in range(VAL_ENTRY_COHORT_SIZE)],
        "trade_accumulators": [
            [_new_trade(), _new_trade()] for _ in range(VAL_ENTRY_COHORT_SIZE)
        ],
        "compaction_trace": [],
        "route_accumulators": {},
        "q_diagnostics": _new_q_diagnostics(),
        "test_data_used": False,
    }


def _seal_progress(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("progress_sha256", None)
    result["progress_sha256"] = canonical_sha256(result)
    return result


def _require_trade(value: Any) -> dict[str, Any]:
    expected = set(_new_trade())
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RuntimeError("UNIFIED_EXIT_VAL_PROGRESS_TRADE_INVALID")
    row = dict(value)
    numeric = (
        "undiscounted_net_cash_pnl_bps",
        "undiscounted_risk_utility_penalty_bps",
        "discounted_risk_adjusted_utility_bps",
        "continuation_discount",
    )
    if (
        row["status"] != _OPEN
        and row["status"] != "EXITED"
        and not str(row["status"]).startswith("RIGHT_CENSORED_")
        or any(not math.isfinite(float(row[name])) for name in numeric)
        or any(
            isinstance(row[name], bool)
            or not isinstance(row[name], int)
            or row[name] < 0
            for name in (
                "decision_count",
                "hold_count",
                "hold_wall_clock_seconds",
                "economic_slice_count",
            )
        )
        or not isinstance(row["economic_slice_stream_sha256"], str)
        or len(row["economic_slice_stream_sha256"]) != 64
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_PROGRESS_TRADE_INVALID")
    return row


def _require_progress(
    value: Mapping[str, Any],
    *,
    contract_sha256: str,
    checkpoint_binding_sha256: str,
) -> dict[str, Any]:
    observed = dict(value)
    claimed = observed.pop("progress_sha256", None)
    template = _new_progress(
        contract_sha256=contract_sha256,
        checkpoint_binding_sha256=checkpoint_binding_sha256,
    )
    if (
        set(observed) != set(template)
        or claimed != canonical_sha256(observed)
        or observed["schema_version"] != PROGRESS_SCHEMA_VERSION
        or observed["decision"] != "IN_PROGRESS"
        or observed["contract_sha256"] != contract_sha256
        or observed["checkpoint_binding_sha256"] != checkpoint_binding_sha256
        or observed["test_data_used"] is not False
        or not isinstance(observed["active_side_mask"], list)
        or len(observed["active_side_mask"]) != VAL_ENTRY_COHORT_SIZE
        or not isinstance(observed["trade_accumulators"], list)
        or len(observed["trade_accumulators"]) != VAL_ENTRY_COHORT_SIZE
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_PROGRESS_INVALID")
    for mask, pair in zip(observed["active_side_mask"], observed["trade_accumulators"]):
        if (
            not isinstance(mask, list)
            or len(mask) != 2
            or any(type(item) is not bool for item in mask)
            or not isinstance(pair, list)
            or len(pair) != 2
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_PROGRESS_INVALID")
        checked = [_require_trade(row) for row in pair]
        if any(
            is_active != (row["status"] == _OPEN)
            for is_active, row in zip(mask, checked)
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_PROGRESS_ACTIVE_INVALID")
    for name in (
        "next_state_index",
        "next_entry_scan_position",
        "model_forward_count",
        "materialized_state_view_count",
        "completed_invocation_count",
    ):
        raw = observed[name]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise RuntimeError("UNIFIED_EXIT_VAL_PROGRESS_INVALID")
    if observed["next_entry_scan_position"] > VAL_ENTRY_COHORT_SIZE:
        raise RuntimeError("UNIFIED_EXIT_VAL_PROGRESS_INVALID")
    elapsed = observed["elapsed_compute_seconds"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or elapsed < 0
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_PROGRESS_INVALID")
    observed["progress_sha256"] = claimed
    return observed


def _accumulate_slice(
    trade: dict[str, Any], step: Mapping[str, Any], slice_sha256: str
) -> None:
    discount = float(trade["continuation_discount"])
    trade["undiscounted_net_cash_pnl_bps"] += float(
        step["undiscounted_net_cash_pnl_increment_bps"]
    )
    trade["undiscounted_risk_utility_penalty_bps"] += float(
        step["risk_utility_penalty_bps"]
    )
    trade["discounted_risk_adjusted_utility_bps"] += discount * float(
        step["undiscounted_risk_adjusted_utility_increment_bps"]
    )
    trade["continuation_discount"] = discount * float(step["continuation_gamma"])
    previous = bytes.fromhex(trade["economic_slice_stream_sha256"])
    trade["economic_slice_stream_sha256"] = hashlib.sha256(
        previous + bytes.fromhex(slice_sha256)
    ).hexdigest()
    trade["economic_slice_count"] += 1


def _accumulate_routes(
    accumulators: dict[str, Any],
    output: Mapping[str, Any],
    *,
    active_side_mask: np.ndarray | None = None,
) -> None:
    for name in _ROUTE_KEYS:
        tensor = output.get(name)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim < 2
            or tensor.shape[0] < 1
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise RuntimeError(f"UNIFIED_EXIT_VAL_ROUTE_OUTPUT_INVALID:{name}")
        values = tensor.detach().to(dtype=torch.float64, device="cpu")
        if active_side_mask is not None:
            mask = np.asarray(active_side_mask, dtype=np.bool_)
            if values.ndim < 3 or tuple(values.shape[:2]) != tuple(mask.shape):
                raise RuntimeError(f"UNIFIED_EXIT_VAL_ROUTE_SIDE_SHAPE_INVALID:{name}")
            values = values[torch.from_numpy(mask)]
        if name == "exit_family_tf_feature_gate":
            # The shared Entry/Exit model uses 2 * sigmoid feature scaling.
            # Its (0,2) contract is distinct from the simplex route weights.
            invalid_range = bool(
                ((values <= 0.0) | (values >= 2.0)).any().item()
            )
        else:
            invalid_range = bool((values < -1e-8).any().item()) or bool(
                (values > 1.0 + 1e-6).any().item()
            )
        if invalid_range:
            raise RuntimeError(f"UNIFIED_EXIT_VAL_ROUTE_RANGE_INVALID:{name}")
        flat = values.reshape(values.shape[0], -1)
        row_sum = flat.sum(dim=1, keepdim=True)
        if bool((row_sum <= 0).any().item()):
            raise RuntimeError(f"UNIFIED_EXIT_VAL_ROUTE_ZERO_INVALID:{name}")
        probabilities = flat / row_sum
        effective = torch.exp(
            -(probabilities * probabilities.clamp_min(1e-300).log()).sum(dim=1)
        )
        top = torch.argmax(flat, dim=1)
        top_counts = torch.bincount(top, minlength=flat.shape[1]).tolist()
        observed = accumulators.get(name)
        if observed is None:
            observed = {
                "shape_tail": list(values.shape[1:]),
                "batch_row_count": 0,
                "element_count": 0,
                "sum": 0.0,
                "sumsq": 0.0,
                "min": None,
                "max": None,
                "effective_route_count_sum": 0.0,
                "top_flat_index_count": [0] * flat.shape[1],
            }
            accumulators[name] = observed
        if observed["shape_tail"] != list(values.shape[1:]):
            raise RuntimeError(f"UNIFIED_EXIT_VAL_ROUTE_SHAPE_DRIFT:{name}")
        observed["batch_row_count"] += int(values.shape[0])
        observed["element_count"] += int(values.numel())
        observed["sum"] += float(values.sum().item())
        observed["sumsq"] += float((values * values).sum().item())
        minimum = float(values.min().item())
        maximum = float(values.max().item())
        observed["min"] = (
            minimum if observed["min"] is None else min(observed["min"], minimum)
        )
        observed["max"] = (
            maximum if observed["max"] is None else max(observed["max"], maximum)
        )
        observed["effective_route_count_sum"] += float(effective.sum().item())
        observed["top_flat_index_count"] = [
            int(old) + int(new)
            for old, new in zip(observed["top_flat_index_count"], top_counts)
        ]


def _accumulate_q(
    diagnostics: dict[str, Any],
    *,
    q: torch.Tensor,
    actions: np.ndarray,
    active_rows: np.ndarray,
) -> None:
    mask = torch.from_numpy(active_rows).to(q.device)
    selected = q.detach()[mask]
    if selected.numel() == 0:
        return
    values = selected.to(dtype=torch.float64, device="cpu")
    diagnostics["active_side_decision_count"] += int(values.shape[0])
    diagnostics["q_cell_count"] += int(values.numel())
    diagnostics["q_sum"] += float(values.sum().item())
    diagnostics["q_sumsq"] += float((values * values).sum().item())
    minimum, maximum = float(values.min().item()), float(values.max().item())
    diagnostics["q_min"] = (
        minimum if diagnostics["q_min"] is None else min(diagnostics["q_min"], minimum)
    )
    diagnostics["q_max"] = (
        maximum if diagnostics["q_max"] is None else max(diagnostics["q_max"], maximum)
    )
    margins = (values[:, 1] - values[:, 0]).numpy()
    diagnostics["margin_sum"] += float(margins.sum())
    diagnostics["margin_sumsq"] += float(np.square(margins).sum())
    for side in range(2):
        side_active = active_rows[:, side]
        side_actions = actions[:, side][side_active]
        hold = int(np.sum(side_actions == 0))
        exit_now = int(np.sum(side_actions == 1))
        diagnostics["hold_action_count"] += hold
        diagnostics["exit_now_action_count"] += exit_now
        diagnostics["hold_action_count_by_side"][side] += hold
        diagnostics["exit_now_action_count_by_side"][side] += exit_now


def accumulate_route_diagnostics_v1(
    accumulators: dict[str, Any],
    output: Mapping[str, Any],
    *,
    active_side_mask: np.ndarray | None = None,
) -> None:
    """Accumulate bounded observation-only route evidence from one model call."""

    _accumulate_routes(accumulators, output, active_side_mask=active_side_mask)


def finalize_route_diagnostics_v1(
    accumulators: Mapping[str, Any],
) -> dict[str, Any]:
    """Finalize observation-only route evidence without importance claims."""

    return _finalize_routes(accumulators)


def _finalize_routes(accumulators: Mapping[str, Any]) -> dict[str, Any]:
    if set(accumulators) != set(_ROUTE_KEYS):
        raise RuntimeError("UNIFIED_EXIT_VAL_ROUTE_EVIDENCE_INCOMPLETE")
    result: dict[str, Any] = {}
    for name, raw in accumulators.items():
        count = int(raw["element_count"])
        rows = int(raw["batch_row_count"])
        mean = float(raw["sum"]) / count
        variance = max(0.0, float(raw["sumsq"]) / count - mean * mean)
        result[name] = {
            "shape_tail": list(raw["shape_tail"]),
            "batch_row_count": rows,
            "element_count": count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": float(raw["min"]),
            "max": float(raw["max"]),
            "mean_effective_route_count": float(raw["effective_route_count_sum"])
            / rows,
            "top_flat_index_count": list(raw["top_flat_index_count"]),
        }
    return {
        "semantics": "observation_only_gate_and_route_usage_not_causal_feature_importance",
        "causal_feature_importance_claimed": False,
        "routes": result,
    }


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "p75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def _finalize_result(
    progress: Mapping[str, Any],
    *,
    adapter: RandomAccessValRolloutAdapterV1,
    checkpoint_binding: Mapping[str, Any],
    execution_contract: Mapping[str, Any],
    entry_route_diagnostics: Mapping[str, Any],
    entry_policy_decisions: Mapping[str, Any],
    guard_reason: str | None,
) -> dict[str, Any]:
    outcomes: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    for entry_position, entry in enumerate(adapter.entries):
        for side_index in range(2):
            trade = dict(progress["trade_accumulators"][entry_position][side_index])
            row = {
                "entry_row_index": int(entry["entry_row_index"]),
                "side_index": side_index,
                **trade,
                "economic_terminal": False,
                "capacity_or_512_terminal": False,
            }
            row["outcome_sha256"] = canonical_sha256(row)
            outcomes.append(row)
            status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    truncated = sum(
        count
        for status, count in status_counts.items()
        if status.startswith("TRUNCATED_")
    )
    censored = sum(
        count
        for status, count in status_counts.items()
        if status.startswith("RIGHT_CENSORED_")
    )
    exited_rows = [row for row in outcomes if row["status"] == "EXITED"]
    pnl_all = [float(row["undiscounted_net_cash_pnl_bps"]) for row in outcomes]
    pnl_exited = [float(row["undiscounted_net_cash_pnl_bps"]) for row in exited_rows]
    by_side = {}
    for side in range(2):
        rows = [row for row in outcomes if row["side_index"] == side]
        exited_side = [row for row in rows if row["status"] == "EXITED"]
        by_side[str(side)] = {
            "side": ("long", "short")[side],
            "trade_count": len(rows),
            "exited_count": len(exited_side),
            "observed_net_cash_pnl_bps_sum": float(
                sum(row["undiscounted_net_cash_pnl_bps"] for row in rows)
            ),
            "exited_net_cash_pnl_bps_quantiles": _quantiles(
                [row["undiscounted_net_cash_pnl_bps"] for row in exited_side]
            ),
        }
    q_diag = dict(progress["q_diagnostics"])
    decision_count = int(q_diag["active_side_decision_count"])
    q_count = int(q_diag["q_cell_count"])
    q_mean = float(q_diag["q_sum"]) / max(1, q_count)
    q_variance = max(0.0, float(q_diag["q_sumsq"]) / max(1, q_count) - q_mean * q_mean)
    margin_mean = float(q_diag["margin_sum"]) / max(1, decision_count)
    margin_variance = max(
        0.0,
        float(q_diag["margin_sumsq"]) / max(1, decision_count)
        - margin_mean * margin_mean,
    )
    decision = (
        "TRUNCATED_NON_AUTHORITATIVE"
        if truncated
        else "COMPLETE_WITH_RIGHT_CENSORING"
        if censored
        else "PASS_COMPLETE"
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "decision": decision,
        "contract_sha256": adapter.contract["contract_sha256"],
        "checkpoint_binding": dict(checkpoint_binding),
        "checkpoint_binding_sha256": checkpoint_binding["binding_sha256"],
        "execution_contract": dict(execution_contract),
        "execution_contract_sha256": execution_contract["execution_contract_sha256"],
        "entry_pair_cohort_size": VAL_ENTRY_COHORT_SIZE,
        "side_trade_count": VAL_ENTRY_COHORT_SIZE * 2,
        "exited_side_trade_count": len(exited_rows),
        "right_censored_side_trade_count": censored,
        "compute_truncated_side_trade_count": truncated,
        "status_counts": status_counts,
        "economic_terminal_count": 0,
        "capacity_or_512_terminal_count": 0,
        "model_forward_count": int(progress["model_forward_count"]),
        "materialized_state_view_count": int(progress["materialized_state_view_count"]),
        "max_decision_state_index": max(
            (int(row["state_index"]) for row in progress["compaction_trace"]),
            default=None,
        ),
        "compaction_trace": list(progress["compaction_trace"]),
        "trade_outcomes": outcomes,
        "policy_economics": {
            "semantics": "observed_executable_net_cash_pnl_including_exit_or_censor_boundary",
            "all_side_trades_net_cash_pnl_bps_sum": float(sum(pnl_all)),
            "all_side_trades_net_cash_pnl_bps_quantiles": _quantiles(pnl_all),
            "exited_side_trades_net_cash_pnl_bps_sum": float(sum(pnl_exited)),
            "exited_side_trades_net_cash_pnl_bps_quantiles": _quantiles(pnl_exited),
            "by_side": by_side,
            "authoritative_complete_policy_metric": truncated == 0 and censored == 0,
        },
        "exit_policy_diagnostics": {
            "action_order": ["hold", "exit_now"],
            "tie_break": "action_order_first_max_hold",
            "active_side_decision_count": decision_count,
            "hold_action_count": int(q_diag["hold_action_count"]),
            "exit_now_action_count": int(q_diag["exit_now_action_count"]),
            "hold_action_count_by_side": list(q_diag["hold_action_count_by_side"]),
            "exit_now_action_count_by_side": list(
                q_diag["exit_now_action_count_by_side"]
            ),
            "q_mean": q_mean,
            "q_std": math.sqrt(q_variance),
            "q_min": q_diag["q_min"],
            "q_max": q_diag["q_max"],
            "exit_minus_hold_margin_mean": margin_mean,
            "exit_minus_hold_margin_std": math.sqrt(margin_variance),
            "exit_state_index_quantiles": _quantiles(
                [float(row["exit_state_index"]) for row in exited_rows]
            ),
            "hold_count_quantiles": _quantiles(
                [float(row["hold_count"]) for row in outcomes]
            ),
        },
        "exit_gate_and_feature_route_diagnostics": _finalize_routes(
            progress["route_accumulators"]
        ),
        "entry_gate_and_feature_route_diagnostics": dict(entry_route_diagnostics),
        "entry_policy_decisions": dict(entry_policy_decisions),
        "entry_exit_policy_metrics": coupled_entry_exit_policy_metrics(
            entry_policy=entry_policy_decisions, trade_outcomes=outcomes,
            full_cohort_authoritative=truncated == 0 and censored == 0,
        ),
        "compute_guard_triggered": guard_reason,
        "rollout_execution_complete": truncated == 0,
        "full_cohort_policy_metrics_authoritative": truncated == 0 and censored == 0,
        "resume_semantics": {
            "progress_saved_after_complete_state_decision": True,
            "replayed_unsaved_work_can_only_be_read_only": True,
            "semantic_result_independent_of_segment_boundaries": True,
        },
        "test_data_used": False,
    }
    semantic = dict(result)
    semantic.pop("semantic_result_sha256", None)
    result["semantic_result_sha256"] = canonical_sha256(semantic)
    return result


def require_random_access_val_evaluation_result_v1(
    value: Mapping[str, Any],
    *,
    rollout_contract_sha256: str,
    checkpoint_binding_sha256: str,
    execution_contract_sha256: str,
) -> dict[str, Any]:
    """Validate a completed immutable result so outer publication can recover."""

    if not isinstance(value, Mapping):
        raise RuntimeError("UNIFIED_EXIT_VAL_RESULT_INVALID")
    result = dict(value)
    claimed = result.pop("semantic_result_sha256", None)
    checkpoint = result.get("checkpoint_binding")
    execution = result.get("execution_contract")
    outcomes = result.get("trade_outcomes")
    if (
        result.get("schema_version") != RESULT_SCHEMA_VERSION
        or result.get("decision")
        not in {
            "PASS_COMPLETE",
            "COMPLETE_WITH_RIGHT_CENSORING",
            "TRUNCATED_NON_AUTHORITATIVE",
        }
        or result.get("contract_sha256") != rollout_contract_sha256
        or result.get("checkpoint_binding_sha256") != checkpoint_binding_sha256
        or result.get("execution_contract_sha256") != execution_contract_sha256
        or not isinstance(execution, Mapping)
        or execution.get("execution_contract_sha256") != execution_contract_sha256
        or canonical_sha256(
            {
                key: item
                for key, item in execution.items()
                if key != "execution_contract_sha256"
            }
        )
        != execution_contract_sha256
        or not isinstance(checkpoint, Mapping)
        or require_selected_weight_ema_checkpoint_binding_v1(checkpoint)[
            "binding_sha256"
        ]
        != checkpoint_binding_sha256
        or result.get("entry_pair_cohort_size") != VAL_ENTRY_COHORT_SIZE
        or result.get("side_trade_count") != 2 * VAL_ENTRY_COHORT_SIZE
        or not isinstance(outcomes, list)
        or len(outcomes) != 2 * VAL_ENTRY_COHORT_SIZE
        or result.get("test_data_used") is not False
        or not isinstance(claimed, str)
        or claimed != canonical_sha256(result)
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_RESULT_INVALID")
    policy = require_entry_policy_decisions(
        result.get("entry_policy_decisions"),
        entry_row_indices=list(range(VAL_ENTRY_COHORT_SIZE)),
        checkpoint_binding_sha256=checkpoint_binding_sha256,
    )
    authoritative = all(row["status"] == "EXITED" for row in outcomes)
    metrics = coupled_entry_exit_policy_metrics(
        entry_policy=policy, trade_outcomes=outcomes,
        full_cohort_authoritative=authoritative,
    )
    if (
        execution.get("entry_policy_sha256") != policy["policy_sha256"]
        or result.get("entry_exit_policy_metrics") != metrics
        or result.get("full_cohort_policy_metrics_authoritative") is not authoritative
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_RESULT_POLICY_INVALID")
    result["semantic_result_sha256"] = claimed
    return result


def run_resumable_random_access_val_evaluation_v1(
    *,
    model: nn.Module,
    entry_decision_representations: torch.Tensor,
    adapter: RandomAccessValRolloutAdapterV1,
    checkpoint_binding: Mapping[str, Any],
    entry_route_diagnostics: Mapping[str, Any],
    entry_policy_decisions: Mapping[str, Any],
    progress_path: Path,
    result_path: Path,
    max_forwards_this_invocation: int,
    policy_batch_size: int,
    progress_interval_forwards: int = 64,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Evaluate both sides, pausing only at a fully committed state boundary."""

    contract = require_random_access_val_rollout_contract(adapter.contract)
    checked_checkpoint = require_selected_weight_ema_checkpoint_binding_v1(
        checkpoint_binding
    )
    entry_policy_decisions = require_entry_policy_decisions(
        entry_policy_decisions,
        entry_row_indices=[int(row["entry_row_index"]) for row in adapter.entries],
        checkpoint_binding_sha256=checked_checkpoint["binding_sha256"],
    )
    execution_contract = {
        "schema_version": "gx1_unified_exit_random_access_val_execution_v2",
        "entry_policy_sha256": entry_policy_decisions["policy_sha256"],
        "rollout_contract_sha256": contract["contract_sha256"],
        "checkpoint_binding_sha256": checked_checkpoint["binding_sha256"],
        "policy_batch_size": policy_batch_size,
        "cohort_cursor_semantics": "ascending_entry_position_within_common_state_v1",
        "progress_commit_semantics": "complete_policy_subbatch_only_v1",
        "test_data_used": False,
    }
    execution_contract["execution_contract_sha256"] = canonical_sha256(
        execution_contract
    )
    checkpoint_binding_sha = checked_checkpoint["binding_sha256"]
    if (
        checked_checkpoint["model_variant"] != "weight_ema"
        or checked_checkpoint["model_state_sha256"] != contract["model_state_sha256"]
        or checked_checkpoint["checkpoint_file_sha256"]
        != contract["checkpoint_file_sha256"]
        or model.training
        or getattr(model, "unified_exit_random_access_architecture_version", None)
        != RANDOM_ACCESS_MODEL_SCHEMA_VERSION
        or bytes(
            model.state_dict()["unified_exit_random_access_architecture_sha256"]
            .detach()
            .cpu()
            .tolist()
        ).hex()
        != RANDOM_ACCESS_MODEL_SCHEMA_SHA256
        or canonical_model_state_sha256(model.state_dict())
        != contract["model_state_sha256"]
        or _tensor_sha256(entry_decision_representations)
        != contract["entry_decision_representations_sha256"]
        or entry_decision_representations.shape[0] != VAL_ENTRY_COHORT_SIZE
        or isinstance(max_forwards_this_invocation, bool)
        or max_forwards_this_invocation < 1
        or isinstance(policy_batch_size, bool)
        or policy_batch_size not in (4, 8, 16)
        or isinstance(progress_interval_forwards, bool)
        or progress_interval_forwards < 1
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_EXECUTION_BINDING_INVALID")
    if result_path.exists():
        if result_path.is_symlink() or not result_path.is_file():
            raise RuntimeError("UNIFIED_EXIT_VAL_RESULT_INVALID")
        return require_random_access_val_evaluation_result_v1(
            json.loads(result_path.read_text()),
            rollout_contract_sha256=contract["contract_sha256"],
            checkpoint_binding_sha256=checkpoint_binding_sha,
            execution_contract_sha256=execution_contract["execution_contract_sha256"],
        )
    if progress_path.exists():
        progress = _require_progress(
            json.loads(progress_path.read_text()),
            contract_sha256=execution_contract["execution_contract_sha256"],
            checkpoint_binding_sha256=checkpoint_binding_sha,
        )
    else:
        progress = _seal_progress(
            _new_progress(
                contract_sha256=execution_contract["execution_contract_sha256"],
                checkpoint_binding_sha256=checkpoint_binding_sha,
            )
        )
        _atomic_json(progress_path, progress, replace=False)
    active = np.asarray(progress["active_side_mask"], dtype=np.bool_)
    guard = contract["compute_guard"]
    invocation_started = monotonic()
    window_started = invocation_started
    forwards_this_invocation = 0
    guard_reason: str | None = None
    while bool(active.any()):
        elapsed = float(progress["elapsed_compute_seconds"]) + (
            monotonic() - invocation_started
        )
        if int(progress["model_forward_count"]) >= guard["max_model_forwards"]:
            guard_reason = "max_model_forwards"
            break
        if (
            guard["wall_limit_scope"] == "entire_rollout"
            and elapsed >= float(guard["max_wall_seconds"])
        ):
            guard_reason = "max_wall_seconds"
            break
        wall_window_exhausted = (
            guard["wall_limit_scope"] == "invocation"
            and monotonic() - window_started >= float(guard["max_wall_seconds"])
        )
        if forwards_this_invocation >= max_forwards_this_invocation or wall_window_exhausted:
            progress["elapsed_compute_seconds"] = elapsed
            progress["completed_invocation_count"] += 1
            progress["active_side_mask"] = active.tolist()
            progress = _seal_progress(progress)
            _atomic_json(progress_path, progress, replace=True)
            pause = {
                "schema_version": PAUSE_SCHEMA_VERSION,
                "decision": "PAUSED_RESUMABLE",
                "pause_reason": "invocation_wall_limit" if wall_window_exhausted else "invocation_forward_limit",
                "rollout_contract_sha256": contract["contract_sha256"],
                "execution_contract_sha256": execution_contract[
                    "execution_contract_sha256"
                ],
                "checkpoint_binding_sha256": checkpoint_binding_sha,
                "next_state_index": int(progress["next_state_index"]),
                "next_entry_scan_position": int(progress["next_entry_scan_position"]),
                "model_forward_count": int(progress["model_forward_count"]),
                "completed_entry_pair_count": int((~active.any(axis=1)).sum()),
                "completed_side_trade_count": int(active.size - active.sum()),
                "progress_path": str(progress_path),
                "progress_file_sha256": file_sha256(progress_path),
                "test_data_used": False,
            }
            pause["pause_sha256"] = canonical_sha256(pause)
            return pause
        state_index = int(progress["next_state_index"])
        scan_position = int(progress["next_entry_scan_position"])
        remaining_mask = active.any(axis=1)
        remaining_mask[:scan_position] = False
        active_entries = np.flatnonzero(remaining_mask)[:policy_batch_size]
        if len(active_entries) == 0:
            progress["next_state_index"] = state_index + 1
            progress["next_entry_scan_position"] = 0
            continue
        if (
            int(progress["materialized_state_view_count"]) + len(active_entries)
            > guard["max_materialized_state_views"]
        ):
            guard_reason = "max_materialized_state_views"
            break
        if scan_position == 0:
            progress["compaction_trace"].append(
                {
                    "state_index": state_index,
                    "active_entry_count": int(active.any(axis=1).sum()),
                    "active_side_count": int(active.sum()),
                }
            )
        row_indices = [
            adapter.entries[int(position)]["entry_row_index"]
            for position in active_entries
        ]
        envelopes = adapter.materialize_active_batch(row_indices, state_index)
        states = [envelope["state"] for envelope in envelopes]
        model_inputs = collate_random_access_states_v1(
            states,
            normalization_artifact=adapter.normalization,
            device=entry_decision_representations.device,
        )
        selected = torch.as_tensor(
            active_entries,
            dtype=torch.long,
            device=entry_decision_representations.device,
        )
        model_inputs["entry_decision_representation"] = (
            entry_decision_representations.index_select(0, selected)
        )
        model_inputs["action_valid_mask"] = torch.ones(
            (len(envelopes), 2, 2),
            dtype=torch.bool,
            device=entry_decision_representations.device,
        )
        with torch.inference_mode():
            output = model.forward_exit_random_access_batch(**model_inputs)
        q = output.get("exit_action_q_bps")
        valid = output.get("exit_action_valid_mask")
        if (
            not isinstance(q, torch.Tensor)
            or q.shape != (len(envelopes), 2, 2)
            or not bool(torch.isfinite(q).all().item())
            or not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.bool
            or valid.shape != q.shape
            or not bool(valid.all().item())
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_MODEL_OUTPUT_INVALID")
        active_rows_before = active[active_entries].copy()
        _accumulate_routes(
            progress["route_accumulators"],
            output,
            active_side_mask=active_rows_before,
        )
        actions = unique_active_exit_actions(q, active_rows_before)
        _accumulate_q(
            progress["q_diagnostics"],
            q=q,
            actions=actions,
            active_rows=active_rows_before,
        )
        progress["model_forward_count"] += 1
        progress["materialized_state_view_count"] += len(envelopes)
        forwards_this_invocation += 1
        for batch_position, raw_entry_position in enumerate(active_entries):
            entry_position = int(raw_entry_position)
            envelope = envelopes[batch_position]
            for side in range(2):
                if not active[entry_position, side]:
                    continue
                trade = progress["trade_accumulators"][entry_position][side]
                trade["decision_count"] += 1
                if int(actions[batch_position, side]) == 1:
                    step, slice_sha = adapter.compose_selected_action(
                        entry_row_index=adapter.entries[entry_position][
                            "entry_row_index"
                        ],
                        side_index=side,
                        state_index=state_index,
                        action="exit_now",
                    )
                    _accumulate_slice(trade, step, slice_sha)
                    trade["status"] = "EXITED"
                    trade["exit_state_index"] = state_index
                    trade["exit_decision_time_ns"] = envelope["state"][
                        "decision_time_ns"
                    ]
                    active[entry_position, side] = False
                elif envelope["successor_observed"]:
                    step, slice_sha = adapter.compose_selected_action(
                        entry_row_index=adapter.entries[entry_position][
                            "entry_row_index"
                        ],
                        side_index=side,
                        state_index=state_index,
                        action="hold",
                    )
                    _accumulate_slice(trade, step, slice_sha)
                    trade["hold_count"] += 1
                    trade["hold_wall_clock_seconds"] += int(
                        step["elapsed_wall_clock_seconds"]
                    )
                else:
                    reason = envelope["right_censor_reason_if_hold"]
                    if reason not in {"split_end", "unknown_source_gap"}:
                        raise RuntimeError("UNIFIED_EXIT_VAL_CENSOR_REASON_INVALID")
                    trade["status"] = f"RIGHT_CENSORED_{reason.upper()}"
                    active[entry_position, side] = False
        progress["next_entry_scan_position"] = int(active_entries[-1]) + 1
        if not bool(active.any(axis=1)[progress["next_entry_scan_position"] :].any()):
            progress["next_state_index"] = state_index + 1
            progress["next_entry_scan_position"] = 0
        progress["active_side_mask"] = active.tolist()
        if forwards_this_invocation % progress_interval_forwards == 0:
            progress["elapsed_compute_seconds"] = float(
                progress["elapsed_compute_seconds"]
            ) + (monotonic() - invocation_started)
            invocation_started = monotonic()
            progress = _seal_progress(progress)
            _atomic_json(progress_path, progress, replace=True)
    progress["elapsed_compute_seconds"] = float(progress["elapsed_compute_seconds"]) + (
        monotonic() - invocation_started
    )
    if guard_reason is not None:
        for entry_position, side in zip(*np.nonzero(active)):
            progress["trade_accumulators"][int(entry_position)][int(side)]["status"] = (
                f"TRUNCATED_COMPUTE_GUARD_{guard_reason.upper()}"
            )
            active[int(entry_position), int(side)] = False
    progress["active_side_mask"] = active.tolist()
    progress["completed_invocation_count"] += 1
    progress = _seal_progress(progress)
    _atomic_json(progress_path, progress, replace=True)
    result = _finalize_result(
        progress,
        adapter=adapter,
        checkpoint_binding=checked_checkpoint,
        execution_contract=execution_contract,
        entry_route_diagnostics=entry_route_diagnostics,
        entry_policy_decisions=entry_policy_decisions,
        guard_reason=guard_reason,
    )
    _atomic_json(result_path, result, replace=False)
    terminal_progress = dict(progress)
    terminal_progress["decision"] = "COMPLETE"
    terminal_progress["result_path"] = str(result_path)
    terminal_progress["result_file_sha256"] = file_sha256(result_path)
    terminal_progress["semantic_result_sha256"] = result["semantic_result_sha256"]
    terminal_progress = _seal_progress(terminal_progress)
    _atomic_json(progress_path, terminal_progress, replace=True)
    return result


__all__ = (
    "PAUSE_SCHEMA_VERSION",
    "PROGRESS_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "accumulate_route_diagnostics_v1",
    "canonical_sha256",
    "finalize_route_diagnostics_v1",
    "file_sha256",
    "require_random_access_val_evaluation_result_v1",
    "run_resumable_random_access_val_evaluation_v1",
)
