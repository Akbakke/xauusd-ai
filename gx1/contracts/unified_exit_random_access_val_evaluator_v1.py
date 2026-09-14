"""Restartable full-cohort VAL evaluation for random-access Exit v2."""

from __future__ import annotations

import hashlib
import copy
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
    marked_entry_exit_policy_metrics,
)

from gx1.contracts.unified_exit_economics_objective_v2 import MARK_TO_MARKET_REWARD_ACCOUNTING
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.unified_exit_random_access_model_v1 import (
    RANDOM_ACCESS_MODEL_SCHEMA_SHA256,
    RANDOM_ACCESS_MODEL_SCHEMA_VERSION,
)
from gx1.contracts.unified_exit_random_access_training_v1 import (
    collate_random_access_states_v1,
    _collate_states,
    _normalization_surface,
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

PROGRESS_SCHEMA_VERSION = "gx1_unified_exit_random_access_val_progress_v3"
RESULT_SCHEMA_VERSION = "gx1_unified_exit_random_access_val_evaluation_v2"
MARKED_RESULT_SCHEMA_VERSION = "gx1_unified_exit_random_access_val_evaluation_v3"
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
        observation_unit = "entry_row"
        if active_side_mask is not None:
            mask = np.asarray(active_side_mask, dtype=np.bool_)
            # Native Exit routes describe one shared market state. Only the
            # action Q values carry a LONG/SHORT axis. Count that state once
            # while either side is active, never duplicate it for both sides.
            if (
                mask.shape != (values.shape[0], 2)
                or values.ndim < 3
                or values.shape[1] != 1
            ):
                raise RuntimeError(f"UNIFIED_EXIT_VAL_ROUTE_SHARED_STATE_SHAPE_INVALID:{name}")
            active_entries = mask.any(axis=1)
            if not bool(active_entries.any()):
                raise RuntimeError(f"UNIFIED_EXIT_VAL_ROUTE_NO_ACTIVE_STATE:{name}")
            values = values[:, 0][torch.from_numpy(active_entries)]
            observation_unit = "active_entry_state_shared_by_sides"
        if name == "exit_family_tf_feature_gate":
            # FP32 2 * sigmoid can round to exactly 0 or 2. Observe these
            # finite endpoints and report saturation below. Admission/runtime
            # owners still enforce the stricter open-range quality contract.
            invalid_range = bool(
                ((values < 0.0) | (values > 2.0)).any().item()
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
                "observation_unit": observation_unit,
                "batch_row_count": 0,
                "element_count": 0,
                "sum": 0.0,
                "sumsq": 0.0,
                "min": None,
                "max": None,
                "effective_route_count_sum": 0.0,
                "top_flat_index_count": [0] * flat.shape[1],
                "saturated_lower_element_count": 0,
                "saturated_upper_element_count": 0,
                "coordinate_sum": [0.0] * flat.shape[1],
                "coordinate_sumsq": [0.0] * flat.shape[1],
                "coordinate_min": None,
                "coordinate_max": None,
                "raw_entropy_sum": 0.0,
            }
            accumulators[name] = observed
        if observed.get("observation_unit") != observation_unit:
            raise RuntimeError(f"UNIFIED_EXIT_VAL_ROUTE_OBSERVATION_UNIT_DRIFT:{name}")
        if observed["shape_tail"] != list(values.shape[1:]):
            raise RuntimeError(f"UNIFIED_EXIT_VAL_ROUTE_SHAPE_DRIFT:{name}")
        if name == "exit_family_tf_feature_gate":
            observed["saturated_lower_element_count"] += int((values == 0.0).sum().item())
            observed["saturated_upper_element_count"] += int((values == 2.0).sum().item())
        observed["batch_row_count"] += int(values.shape[0])
        observed["element_count"] += int(values.numel())
        observed["sum"] += float(values.sum().item())
        observed["sumsq"] += float((values * values).sum().item())
        # Persist per-coordinate moments for the existing candidate gate-health
        # rules. A global variance cannot show whether an individual feature
        # is constant. Each active shared state contributes once to these observations.
        for key, update in (
            ("coordinate_sum", flat.sum(dim=0).numpy()),
            ("coordinate_sumsq", flat.square().sum(dim=0).numpy()),
        ):
            observed[key] = (np.asarray(observed[key]) + update).tolist()
        for key, update, combine in (
            ("coordinate_min", flat.min(dim=0).values.numpy(), np.minimum),
            ("coordinate_max", flat.max(dim=0).values.numpy(), np.maximum),
        ):
            observed[key] = (
                update if observed[key] is None
                else combine(np.asarray(observed[key]), update)
            ).tolist()
        clipped = flat.clamp(min=1e-12)
        observed["raw_entropy_sum"] += float(
            (-(clipped * clipped.log()).sum(dim=1).sum()).item()
        )
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
        coordinate_mean = np.asarray(raw["coordinate_sum"], dtype=np.float64) / rows
        coordinate_variance = np.maximum(
            np.asarray(raw["coordinate_sumsq"], dtype=np.float64) / rows
            - np.square(coordinate_mean),
            0.0,
        )
        result[name] = {
            "shape_tail": list(raw["shape_tail"]),
            "observation_unit": raw["observation_unit"],
            "batch_row_count": rows,
            "element_count": count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": float(raw["min"]),
            "max": float(raw["max"]),
            "mean_effective_route_count": float(raw["effective_route_count_sum"])
            / rows,
            "top_flat_index_count": list(raw["top_flat_index_count"]),
            "coordinate_mean_weight": coordinate_mean.tolist(),
            "coordinate_std_weight": np.sqrt(coordinate_variance).tolist(),
            "coordinate_min_observed": list(raw["coordinate_min"]),
            "coordinate_max_observed": list(raw["coordinate_max"]),
            "raw_entropy_mean": float(raw["raw_entropy_sum"]) / rows,
        }
        if name == "exit_family_tf_feature_gate":
            lower = int(raw["saturated_lower_element_count"])
            upper = int(raw["saturated_upper_element_count"])
            result[name]["feature_gate_quality"] = {
                "required_open_range": [0.0, 2.0],
                "saturated_lower_element_count": lower,
                "saturated_upper_element_count": upper,
                "saturated_element_fraction": (lower + upper) / count,
                "open_range_quality_pass": lower == 0 and upper == 0,
                "candidate_admission_claimed": False,
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


def _observed_trade_valuation(trade, *, adapter, entry):
    """Mark a censored position without accumulating or recording an EXIT action."""
    status = trade["status"]
    remaining, time_ns, slice_sha = None, None, None
    if status == "EXITED":
        remaining, time_ns = 0.0, int(trade["exit_decision_time_ns"])
    elif status in {"RIGHT_CENSORED_SPLIT_END", "RIGHT_CENSORED_UNKNOWN_SOURCE_GAP"}:
        index = int(trade["decision_count"]) - 1
        if index < 0:
            raise RuntimeError("UNIFIED_EXIT_MARKED_BOUNDARY_STATE_INVALID")
        step, slice_sha = adapter.compose_selected_action(
            entry_row_index=entry["entry_row_index"], side_index=trade["side_index"],
            state_index=index, action="exit_now",
        )
        remaining = float(step["undiscounted_net_cash_pnl_increment_bps"])
        time_ns = int(step["interval_end_time_ns"])
        expected_time = int(adapter.times.asi8[entry["entry_m1_start_row"] + index]) + 60_000_000_000
        if time_ns != expected_time or (status == "RIGHT_CENSORED_SPLIT_END" and time_ns != int(adapter.times.asi8[-1]) + 60_000_000_000):
            raise RuntimeError("UNIFIED_EXIT_MARKED_BOUNDARY_CLOCK_INVALID")
    return {
        "schema_version": "gx1_observed_position_valuation_v1",
        "valuation_time_ns": time_ns,
        "remaining_liquidation_value_bps": remaining,
        "net_cash_plus_open_value_bps": (float(trade["undiscounted_net_cash_pnl_bps"]) + remaining if remaining is not None else None),
        "discounted_utility_plus_open_value_bps": (float(trade["discounted_risk_adjusted_utility_bps"]) + float(trade["continuation_discount"]) * remaining if remaining is not None else None),
        "valuation_slice_sha256": slice_sha,
        "model_exit_executed": status == "EXITED",
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
    marked = adapter.objective.get("reward_accounting") == MARK_TO_MARKET_REWARD_ACCOUNTING
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
            if marked:
                row["entry_fill_time_ns"] = int(adapter.times.asi8[entry["entry_m1_start_row"]])
                row["valuation"] = _observed_trade_valuation(row, adapter=adapter, entry=entry)
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
    # Full coverage is distinct from completion of every counterfactual side.
    # The policy owner still withholds Bps if any actually selected trade is censored.
    rollout_complete = all(
        row["status"] == "EXITED" or row["status"].startswith("RIGHT_CENSORED_")
        for row in outcomes
    )
    policy_metrics = coupled_entry_exit_policy_metrics(
        entry_policy=entry_policy_decisions, trade_outcomes=outcomes,
        full_cohort_authoritative=rollout_complete,
    )
    result = {
        "schema_version": MARKED_RESULT_SCHEMA_VERSION if marked else RESULT_SCHEMA_VERSION,
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
        "entry_exit_policy_metrics": policy_metrics,
        "compute_guard_triggered": guard_reason,
        "rollout_execution_complete": rollout_complete,
        "full_cohort_policy_metrics_authoritative": policy_metrics["full_cohort_authoritative"],
        "resume_semantics": {
            "progress_saved_after_complete_state_decision": True,
            "replayed_unsaved_work_can_only_be_read_only": True,
            "semantic_result_independent_of_segment_boundaries": True,
        },
        "test_data_used": False,
    }
    if marked:
        result["marked_policy_evaluation"] = marked_entry_exit_policy_metrics(
            entry_policy=entry_policy_decisions, trade_outcomes=outcomes,
            full_cohort_authoritative=rollout_complete,
        )
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
        result.get("schema_version") not in {RESULT_SCHEMA_VERSION, MARKED_RESULT_SCHEMA_VERSION}
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
    rollout_complete = all(
        row["status"] == "EXITED" or str(row["status"]).startswith("RIGHT_CENSORED_")
        for row in outcomes
    )
    metrics = coupled_entry_exit_policy_metrics(
        entry_policy=policy, trade_outcomes=outcomes,
        full_cohort_authoritative=rollout_complete,
    )
    if (
        execution.get("entry_policy_sha256") != policy["policy_sha256"]
        or result.get("entry_exit_policy_metrics") != metrics
        or result.get("rollout_execution_complete") is not rollout_complete
        or result.get("full_cohort_policy_metrics_authoritative")
        is not metrics["full_cohort_authoritative"]
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_RESULT_POLICY_INVALID")
    if result["schema_version"] == MARKED_RESULT_SCHEMA_VERSION:
        marked_metrics = marked_entry_exit_policy_metrics(
            entry_policy=policy, trade_outcomes=outcomes,
            full_cohort_authoritative=rollout_complete,
        )
        if result.get("marked_policy_evaluation") != marked_metrics:
            raise RuntimeError("UNIFIED_EXIT_VAL_RESULT_MARKED_POLICY_INVALID")
    elif "marked_policy_evaluation" in result:
        raise RuntimeError("UNIFIED_EXIT_VAL_RESULT_MARKED_SCHEMA_INVALID")
    result["semantic_result_sha256"] = claimed
    return result


def _verify_val_batch_throughput(model, inputs, output, forward_started: float) -> None:
    """Check actual first-batch Q/actions and log measured inference throughput."""
    q = output["exit_action_q_bps"]
    if q.device.type == "cuda":
        torch.cuda.synchronize(q.device)
    large_seconds = time.monotonic() - forward_started
    size = q.shape[0]

    def sliced(value, start, stop):
        if isinstance(value, torch.Tensor):
            return value[start:stop] if value.ndim > 0 and value.shape[0] == size else value
        if isinstance(value, Mapping):
            return {key: sliced(item, start, stop) for key, item in value.items()}
        return value

    started = time.monotonic()
    with torch.inference_mode():
        parts = [model.forward_exit_random_access_batch(**sliced(inputs, start, start + 16))
                 for start in range(0, size, 16)]
        reference = torch.cat([part["exit_action_q_bps"] for part in parts])
    if q.device.type == "cuda":
        torch.cuda.synchronize(q.device)
    reference_seconds = time.monotonic() - started
    active = np.ones((size, 2), dtype=np.bool_)
    actions_equal = np.array_equal(unique_active_exit_actions(q, active), unique_active_exit_actions(reference, active))
    intermediate_differences = {}
    for key in ("exit_random_access_path_state", "exit_random_access_summary_state",
                "exit_random_access_local_state", "exit_random_access_mtf_state"):
        if key in output and all(key in part for part in parts):
            ref = torch.cat([part[key] for part in parts])
            intermediate_differences[key] = float((output[key] - ref).abs().max().item())
    print(json.dumps({"event": "VAL_BATCH_THROUGHPUT_COMPARISON", "rows": size,
        "policy_batch_size": size, "reference_batch_size": 16,
        "large_batch_seconds": large_seconds, "reference_seconds": reference_seconds,
        "inference_speedup": reference_seconds / large_seconds,
        "max_abs_q_difference_bps": float((q - reference).abs().max().item()),
        "actions_equal": bool(actions_equal), "q_absolute_tolerance_bps": 1e-4,
        "intermediate_max_abs_difference": intermediate_differences,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}), flush=True)
    # Different FP32 batch shapes change reduction order. The real 128-row
    # comparison with both TF32 backends disabled measured <=4.92e-5 Bps,
    # <7.2e-7 intermediate-state error, and exactly identical actions.
    # Bound rounding in Bps (no magnitude-dependent relative allowance);
    # the observed 0.03 Bps reduced-precision regression still fails.
    if not actions_equal:
        raise RuntimeError("UNIFIED_EXIT_VAL_BATCH_ACTION_MISMATCH")
    torch.testing.assert_close(q, reference, atol=1e-4, rtol=0.0)
    print(json.dumps({"event": "VAL_BATCH_THROUGHPUT_VERIFIED", "rows": size,
        "inference_speedup": reference_seconds / large_seconds, "actions_equal": True}), flush=True)


def _restore_cpu_pipeline_progress(origin, *, execution_contract, checkpoint_binding_sha,
                                   cpu_pipeline_workers):
    from gx1.contracts.local_random_access_campaign_v2 import read_bound_json
    if cpu_pipeline_workers is None or set(origin) != {"path", "sha256"}:
        raise RuntimeError("UNIFIED_EXIT_VAL_CPU_RESUME_ORIGIN_INVALID")
    prior_execution = {key: value for key, value in execution_contract.items()
                       if key != "execution_contract_sha256"}
    current_sha = canonical_sha256(prior_execution)
    if execution_contract.get("execution_contract_sha256") != current_sha:
        raise RuntimeError("UNIFIED_EXIT_VAL_PROGRESS_INVALID")
    prior = read_bound_json(Path(origin["path"]), origin["sha256"])
    if prior.get("contract_sha256") != current_sha:
        if prior_execution.get("cpu_pipeline") == "immutable_metadata_shared_path_v2":
            prior_execution["cpu_pipeline"] = "compact_market_inputs_batched_economics_v1"
        else:
            prior_execution.pop("cpu_pipeline", None)
            prior_execution.pop("cpu_workers", None)
    progress = _require_progress(prior, contract_sha256=canonical_sha256(prior_execution),
                                 checkpoint_binding_sha256=checkpoint_binding_sha)
    if progress["completed_invocation_count"] < 1 or progress["materialized_state_view_count"] < 1:
        raise RuntimeError("UNIFIED_EXIT_VAL_CPU_RESUME_COMPLETED_WINDOW_REQUIRED")
    prior_sha = progress["progress_sha256"]
    progress["contract_sha256"] = execution_contract["execution_contract_sha256"]
    return _seal_progress(progress), prior_sha


def _verify_cpu_pipeline(*, model, adapter, rows, state_index, representations,
                         action_mask, active, requests, cache, workers):
    device = representations.device
    keys = [adapter._entry_by_index[row]["entry_m1_start_row"] + state_index for row in rows]
    frozen_surface, _ = _normalization_surface(copy.deepcopy(adapter.normalization))
    # Start the bounded CPU pool before timing steady-state work.
    adapter.materialize_cached_active_batch(rows, state_index, cached_market_rows=set(cache), workers=workers)

    def execute(compact):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.monotonic()
        envelopes = (adapter.materialize_cached_active_batch(rows, state_index, cached_market_rows=set(cache), workers=workers)
                     if compact else adapter.materialize_active_batch(rows, state_index))
        states = [envelope["state"] for envelope in envelopes]
        inputs = (_collate_states(states, surface=frozen_surface, device=device,
                                 _market_state_batch_positions=[]) if compact
                  else collate_random_access_states_v1(
                      states, normalization_artifact=adapter.normalization, device=device))
        inputs.update(entry_decision_representation=representations, action_valid_mask=action_mask)
        with torch.inference_mode():
            output = model.forward_exit_random_access_batch(
                **inputs, _market_state_cache=cache, _market_state_keys=keys,
                _market_state_batch_positions=[] if compact else None,
                _share_identical_path=compact,
            )
        actions = unique_active_exit_actions(output["exit_action_q_bps"], active)
        steps = (adapter.compose_selected_actions(requests) if compact
                 else [adapter.compose_selected_action(**request) for request in requests])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.monotonic() - started, states, output, actions, steps

    reference = execute(False)
    optimized = execute(True)
    if not np.array_equal(reference[3], optimized[3]) or reference[4] != optimized[4]:
        raise RuntimeError("UNIFIED_EXIT_VAL_CPU_PIPELINE_ACTION_OR_ECONOMICS_CHANGED")
    for before, after in zip(reference[1], optimized[1]):
        for name in before:
            if name in {"m1_local_history_x", "state_ctx_cat", "state_ctx_cont", "mtf"}:
                continue
            if isinstance(before[name], np.ndarray):
                np.testing.assert_array_equal(before[name], after[name])
            elif before[name] != after[name]:
                raise RuntimeError("UNIFIED_EXIT_VAL_CPU_PIPELINE_STATE_CHANGED")
    for name, value in reference[2].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(optimized[2][name], value, atol=1e-4, rtol=0.0, msg=name)
    print("VAL_CPU_PIPELINE_VERIFIED " + json.dumps({
        "rows": len(rows), "workers": workers, "reference_seconds": reference[0],
        "optimized_seconds": optimized[0], "pipeline_speedup": reference[0] / optimized[0],
        "actions_equal": True, "economic_steps_and_hashes_equal": True,
        "dynamic_states_equal": True,
        "max_abs_q_difference_bps": float((reference[2]["exit_action_q_bps"] - optimized[2]["exit_action_q_bps"]).abs().max().item()),
    }), flush=True)


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
    cache_market_states: bool = False,
    cpu_pipeline_workers: int | None = None,
    resume_progress_origin: Mapping[str, Any] | None = None,
    progress_interval_forwards: int = 64,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Evaluate both sides, pausing only at a fully committed state boundary."""

    if type(cache_market_states) is not bool:
        raise RuntimeError("UNIFIED_EXIT_VAL_MARKET_CACHE_FLAG_INVALID")
    # A fresh cache for this frozen EMA model and adapter only. Never carry
    # cached tensors across an epoch, checkpoint change or resumed invocation.
    market_state_cache = {} if cache_market_states else None
    cache_verified = False
    if cpu_pipeline_workers not in (None, 0, 4, 8) or (cpu_pipeline_workers is not None and not cache_market_states):
        raise RuntimeError("UNIFIED_EXIT_VAL_CPU_PIPELINE_INVALID")
    pipeline_verified = cpu_pipeline_workers is None
    # Own a private copy of the verified, frozen TRAIN normalization. Dynamic
    # position values still pass the same collation and normalization owner.
    frozen_surface = (_normalization_surface(copy.deepcopy(adapter.normalization))[0]
                      if cpu_pipeline_workers is not None else None)
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
    if cpu_pipeline_workers is not None:
        execution_contract["cpu_pipeline"] = "immutable_metadata_shared_path_v2"
        execution_contract["cpu_workers"] = cpu_pipeline_workers
    if cache_market_states:
        execution_contract["market_state_cache"] = "frozen_model_absolute_m1_row_v1"
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
        or policy_batch_size not in (4, 8, 16, 128, 256)
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
    elif resume_progress_origin is not None:
        progress, prior_sha = _restore_cpu_pipeline_progress(
            resume_progress_origin, execution_contract=execution_contract,
            checkpoint_binding_sha=checkpoint_binding_sha, cpu_pipeline_workers=cpu_pipeline_workers,
        )
        _atomic_json(progress_path, progress, replace=False)
        print("VAL_CPU_PIPELINE_PROGRESS_RESTORED " + json.dumps({
            "prior_progress_sha256": prior_sha,
            "materialized_state_view_count": progress["materialized_state_view_count"],
            "model_forward_count": progress["model_forward_count"],
            "next_state_index": progress["next_state_index"],
            "checkpoint_binding_sha256": checkpoint_binding_sha,
        }), flush=True)
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
        if cpu_pipeline_workers is None:
            envelopes = adapter.materialize_active_batch(row_indices, state_index)
        else:
            envelopes = adapter.materialize_cached_active_batch(
                row_indices, state_index, cached_market_rows=set(market_state_cache),
                workers=cpu_pipeline_workers,
            )
        states = [envelope["state"] for envelope in envelopes]
        market_positions = None
        if cpu_pipeline_workers is not None:
            missing = {}
            for position, state in enumerate(states):
                key = int(state["m1_row_index"])
                if key not in market_state_cache:
                    missing.setdefault(key, position)
            market_positions = list(missing.values())
        model_inputs = (_collate_states(
            states, surface=frozen_surface, device=entry_decision_representations.device,
            _market_state_batch_positions=market_positions,
        ) if cpu_pipeline_workers is not None else collate_random_access_states_v1(
            states, normalization_artifact=adapter.normalization,
            device=entry_decision_representations.device,
        ))
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
        verify_batch = policy_batch_size >= 128 and progress["model_forward_count"] == 0
        if verify_batch and entry_decision_representations.device.type == "cuda":
            torch.cuda.synchronize(entry_decision_representations.device)
        forward_started = time.monotonic()
        cache_arguments = ({"_market_state_cache": market_state_cache,
                            "_market_state_keys": [int(state["m1_row_index"]) for state in states]}
                           if cache_market_states else {})
        if cpu_pipeline_workers is not None:
            cache_arguments["_market_state_batch_positions"] = market_positions
            cache_arguments["_share_identical_path"] = True
        with torch.inference_mode():
            output = model.forward_exit_random_access_batch(**model_inputs, **cache_arguments)
        if verify_batch:
            _verify_val_batch_throughput(model, model_inputs, output, forward_started)
        if cache_market_states and not cache_verified:
            # Compare an actual cache hit against an uncached same-shape call
            # before accepting any actions from this invocation.
            if entry_decision_representations.device.type == "cuda":
                torch.cuda.synchronize(entry_decision_representations.device)
            started = time.monotonic()
            with torch.inference_mode():
                cached = model.forward_exit_random_access_batch(**model_inputs, **{name: value for name, value in cache_arguments.items() if name != "_market_state_batch_positions"})
            if entry_decision_representations.device.type == "cuda":
                torch.cuda.synchronize(entry_decision_representations.device)
            cached_seconds = time.monotonic() - started
            started = time.monotonic()
            with torch.inference_mode():
                reference = model.forward_exit_random_access_batch(**model_inputs)
            if entry_decision_representations.device.type == "cuda":
                torch.cuda.synchronize(entry_decision_representations.device)
            reference_seconds = time.monotonic() - started
            active_check = np.ones((len(states), 2), dtype=np.bool_)
            if not np.array_equal(unique_active_exit_actions(cached["exit_action_q_bps"], active_check),
                                  unique_active_exit_actions(reference["exit_action_q_bps"], active_check)):
                raise RuntimeError("UNIFIED_EXIT_VAL_MARKET_CACHE_ACTION_MISMATCH")
            for name, value in reference.items():
                torch.testing.assert_close(cached[name], value, atol=1e-4, rtol=0.0, msg=name)
            print(json.dumps({"event": "VAL_MARKET_CACHE_VERIFIED", "rows": len(states),
                "cached_seconds": cached_seconds, "uncached_seconds": reference_seconds,
                "inference_speedup": reference_seconds / cached_seconds,
                "max_abs_q_difference_bps": float((cached["exit_action_q_bps"] - reference["exit_action_q_bps"]).abs().max().item()),
                "actions_equal": True, "cached_market_rows": len(market_state_cache)}), flush=True)
            cache_verified = True
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
        requests = []
        request_keys = []
        for position, raw_entry_position in enumerate(active_entries):
            entry_position = int(raw_entry_position)
            for side in range(2):
                if not active[entry_position, side]:
                    continue
                action = "exit_now" if int(actions[position, side]) == 1 else "hold"
                if action == "hold" and not envelopes[position]["successor_observed"]:
                    continue
                request_keys.append((entry_position, side))
                requests.append({"entry_row_index": adapter.entries[entry_position]["entry_row_index"],
                                 "side_index": side, "state_index": state_index, "action": action})
        composed_steps = (adapter.compose_selected_actions(requests) if cpu_pipeline_workers is not None
                          else [adapter.compose_selected_action(**request) for request in requests])
        step_by_key = dict(zip(request_keys, composed_steps))
        if not pipeline_verified:
            _verify_cpu_pipeline(
                model=model, adapter=adapter, rows=row_indices, state_index=state_index,
                representations=model_inputs["entry_decision_representation"],
                action_mask=model_inputs["action_valid_mask"], active=active[active_entries],
                requests=requests, cache=market_state_cache, workers=cpu_pipeline_workers,
            )
            pipeline_verified = True
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
                    step, slice_sha = step_by_key[(entry_position, side)]
                    _accumulate_slice(trade, step, slice_sha)
                    trade["status"] = "EXITED"
                    trade["exit_state_index"] = state_index
                    trade["exit_decision_time_ns"] = envelope["state"][
                        "decision_time_ns"
                    ]
                    active[entry_position, side] = False
                elif envelope["successor_observed"]:
                    step, slice_sha = step_by_key[(entry_position, side)]
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
        if forwards_this_invocation % progress_interval_forwards == 0:
            progress["active_side_mask"] = active.tolist()
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
    "MARKED_RESULT_SCHEMA_VERSION",
    "accumulate_route_diagnostics_v1",
    "canonical_sha256",
    "finalize_route_diagnostics_v1",
    "file_sha256",
    "require_random_access_val_evaluation_result_v1",
    "run_resumable_random_access_val_evaluation_v1",
)
