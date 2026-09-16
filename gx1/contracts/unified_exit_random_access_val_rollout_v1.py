"""Deterministic full-cohort VAL rollout for random-access Exit v2."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.unified_exit_economics_objective_v2 import (
    compose_economic_step,
    revalue_marked_hold_step,
    elapsed_wall_clock_gamma,
)
from gx1.contracts.unified_exit_lifetime_summary_v1 import LIFETIME_SUMMARY_DIM
from gx1.contracts.unified_exit_market_closure_authority_v1 import (
    closure_intervals_by_gap_after_row,
    m1_clock_sha256,
    require_market_closure_authority,
)
from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import (
    canonical_sha256 as _economic_json_sha256,
)
from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    require_lifetime_summary_normalization,
)
from gx1.contracts.unified_exit_random_access_model_v1 import (
    RANDOM_ACCESS_MODEL_SCHEMA_SHA256,
    RANDOM_ACCESS_MODEL_SCHEMA_VERSION,
)
from gx1.contracts.unified_exit_random_access_state_view_v1 import (
    EXIT_ACTION_ORDER,
    M1_LOCAL_HISTORY_ROWS,
    TRADE_PATH_TAIL_MAX_ROWS,
)
from gx1.contracts.unified_exit_random_access_training_v1 import (
    collate_random_access_states_v1,
)
from gx1.features.htf_features import MULTI_TF_TIMEFRAMES


VAL_ROLLOUT_CONTRACT_SCHEMA_VERSION = (
    "gx1_unified_exit_random_access_val_rollout_contract_v2"
)
VAL_ROLLOUT_STATE_SCHEMA_VERSION = "gx1_unified_exit_random_access_val_state_v1"
VAL_ROLLOUT_RESULT_SCHEMA_VERSION = (
    "gx1_unified_exit_random_access_val_rollout_result_v1"
)
VAL_ENTRY_COHORT_SIZE = 5_508
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ENTRY_KEYS = {
    "entry_row_index",
    "entry_m1_start_row",
    "available_state_count",
    "entry_episode_binding_sha256",
    "entry_fill_binding_sha256",
}
_STATE_KEYS = {
    "state_index",
    "m1_row_index",
    "bar_start_time_ns",
    "decision_time_ns",
    "m1_local_history_start_row",
    "m1_local_history_x",
    "state_ctx_cont",
    "state_ctx_cat",
    "trade_path_start_state_index",
    "trade_path_length",
    "trade_path_tail_x",
    "lifetime_summary_x",
    "lifetime_summary_sha256_by_side",
    "mtf",
}


def _canonical_sha256(value: Any) -> str:
    def project(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            return {
                "__ndarray__": True,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        if isinstance(item, Mapping):
            return {str(key): project(raw) for key, raw in item.items()}
        if isinstance(item, (list, tuple)):
            return [project(raw) for raw in item]
        return item

    return hashlib.sha256(
        json.dumps(
            project(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"UNIFIED_EXIT_VAL_{label}_SHA_INVALID")
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise RuntimeError("UNIFIED_EXIT_VAL_ENTRY_REPRESENTATION_INVALID")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _require_entries(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if (
        not isinstance(entries, Sequence)
        or isinstance(entries, (str, bytes))
        or len(entries) != VAL_ENTRY_COHORT_SIZE
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_FULL_COHORT_REQUIRED")
    checked: list[dict[str, Any]] = []
    seen: set[int] = set()
    previous = -1
    for raw in entries:
        if not isinstance(raw, Mapping) or set(raw) != _ENTRY_KEYS:
            raise RuntimeError("UNIFIED_EXIT_VAL_ENTRY_SCHEMA_INVALID")
        row = dict(raw)
        entry = row["entry_row_index"]
        start = row["entry_m1_start_row"]
        count = row["available_state_count"]
        if (
            isinstance(entry, bool)
            or not isinstance(entry, int)
            or entry < 0
            or entry <= previous
            or entry in seen
            or isinstance(start, bool)
            or not isinstance(start, int)
            or start < M1_LOCAL_HISTORY_ROWS - 1
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_ENTRY_INVALID")
        _require_sha(row["entry_episode_binding_sha256"], "EPISODE_BINDING")
        _require_sha(row["entry_fill_binding_sha256"], "FILL_BINDING")
        seen.add(entry)
        previous = entry
        checked.append(row)
    return checked


def _entry_stream_sha256(entries: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in entries:
        digest.update(_canonical_sha256(row).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_random_access_val_rollout_contract(
    *,
    entries: Sequence[Mapping[str, Any]],
    entry_decision_representations: torch.Tensor,
    source_lineage_sha256: str,
    m1_source_sha256: str,
    m1_clock_sha256_value: str,
    m1_source_manifest_file_sha256: str,
    parent_m1_source_sha256: str,
    parent_m1_source_manifest_sha256: str,
    parent_m1_row_offset: int,
    market_closure_authority_sha256: str,
    market_closure_authority_file_sha256: str,
    economic_step_manifest_sha256: str,
    economics_objective_contract_sha256: str,
    normalization_artifact: Mapping[str, Any],
    normalization_file_sha256: str,
    model_state_sha256: str,
    checkpoint_file_sha256: str,
    compute_guard_max_model_forwards: int,
    compute_guard_max_materialized_state_views: int,
    compute_guard_max_wall_seconds: float,
    resumable_wall_limit: bool = False,
) -> dict[str, Any]:
    """Bind the exact immutable VAL cohort, model and source artifacts."""

    cohort = _require_entries(entries)
    representations_sha = _tensor_sha256(entry_decision_representations)
    normalization = require_lifetime_summary_normalization(normalization_artifact)
    if (
        int(entry_decision_representations.shape[0]) != VAL_ENTRY_COHORT_SIZE
        or normalization["val_mode"] != "apply_frozen_train_transform_only"
        or normalization["val_fit_rows"] != 0
        or normalization["test_fit_rows"] != 0
        or normalization["test_accessed"] is not False
        or isinstance(parent_m1_row_offset, bool)
        or not isinstance(parent_m1_row_offset, int)
        or parent_m1_row_offset < 0
        or isinstance(compute_guard_max_model_forwards, bool)
        or not isinstance(compute_guard_max_model_forwards, int)
        or compute_guard_max_model_forwards < 1
        or isinstance(compute_guard_max_materialized_state_views, bool)
        or not isinstance(compute_guard_max_materialized_state_views, int)
        or compute_guard_max_materialized_state_views < VAL_ENTRY_COHORT_SIZE
        or isinstance(compute_guard_max_wall_seconds, bool)
        or not isinstance(compute_guard_max_wall_seconds, (int, float))
        or not math.isfinite(float(compute_guard_max_wall_seconds))
        or float(compute_guard_max_wall_seconds) <= 0.0
        or type(resumable_wall_limit) is not bool
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CONTRACT_INPUT_INVALID")
    result = {
        "schema_version": VAL_ROLLOUT_CONTRACT_SCHEMA_VERSION,
        "decision": "PASS",
        "split": "val",
        "entry_pair_cohort_size": VAL_ENTRY_COHORT_SIZE,
        "both_sides_evaluated": True,
        "entry_cohort_stream_sha256": _entry_stream_sha256(cohort),
        "entry_decision_representations_sha256": representations_sha,
        "source_lineage_sha256": _require_sha(source_lineage_sha256, "SOURCE_LINEAGE"),
        "m1_source_sha256": _require_sha(m1_source_sha256, "M1_SOURCE"),
        "m1_clock_sha256": _require_sha(m1_clock_sha256_value, "M1_CLOCK"),
        "m1_source_manifest_file_sha256": _require_sha(
            m1_source_manifest_file_sha256, "M1_SOURCE_MANIFEST_FILE"
        ),
        "parent_m1_source_sha256": _require_sha(
            parent_m1_source_sha256, "PARENT_M1_SOURCE"
        ),
        "parent_m1_source_manifest_sha256": _require_sha(
            parent_m1_source_manifest_sha256, "PARENT_M1_SOURCE_MANIFEST"
        ),
        "parent_m1_row_offset": parent_m1_row_offset,
        "market_closure_authority_sha256": _require_sha(
            market_closure_authority_sha256, "CLOSURE_AUTHORITY"
        ),
        "market_closure_authority_file_sha256": _require_sha(
            market_closure_authority_file_sha256, "CLOSURE_AUTHORITY_FILE"
        ),
        "economic_step_manifest_sha256": _require_sha(
            economic_step_manifest_sha256, "ECONOMIC_STEP_MANIFEST"
        ),
        "economics_objective_contract_sha256": _require_sha(
            economics_objective_contract_sha256, "ECONOMICS_OBJECTIVE"
        ),
        "normalization_sha256": normalization["normalization_sha256"],
        "normalization_file_sha256": _require_sha(
            normalization_file_sha256, "NORMALIZATION_FILE"
        ),
        "normalization_fit_scope": normalization["fit_scope"],
        "normalization_val_mode": normalization["val_mode"],
        "model_architecture_schema_version": RANDOM_ACCESS_MODEL_SCHEMA_VERSION,
        "model_architecture_sha256": RANDOM_ACCESS_MODEL_SCHEMA_SHA256,
        "model_state_sha256": _require_sha(model_state_sha256, "MODEL_STATE"),
        "checkpoint_file_sha256": _require_sha(
            checkpoint_file_sha256, "CHECKPOINT_FILE"
        ),
        "action_order": list(EXIT_ACTION_ORDER),
        "tie_break": "unique_argmax_or_fail_closed",
        "rollout_order": "relative_m1_state_index_with_active_entry_compaction",
        "capacity_or_512_is_terminal": False,
        "economic_terminal_present": False,
        "compute_guard": {
            "max_model_forwards": compute_guard_max_model_forwards,
            "max_materialized_state_views": compute_guard_max_materialized_state_views,
            "max_wall_seconds": float(compute_guard_max_wall_seconds),
            "wall_limit_scope": "invocation" if resumable_wall_limit else "entire_rollout",
            "guard_stop_semantics": "hard_budget_truncated_wall_scope_explicit_never_terminal",
        },
        "test_data_used": False,
    }
    result["contract_sha256"] = _canonical_sha256(result)
    return result


def require_random_access_val_rollout_contract(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or "contract_sha256" not in value:
        raise RuntimeError("UNIFIED_EXIT_VAL_CONTRACT_INVALID")
    observed = dict(value)
    claimed = observed.pop("contract_sha256")
    required = {
        "schema_version",
        "decision",
        "split",
        "entry_pair_cohort_size",
        "both_sides_evaluated",
        "entry_cohort_stream_sha256",
        "entry_decision_representations_sha256",
        "source_lineage_sha256",
        "m1_source_sha256",
        "m1_clock_sha256",
        "m1_source_manifest_file_sha256",
        "parent_m1_source_sha256",
        "parent_m1_source_manifest_sha256",
        "parent_m1_row_offset",
        "market_closure_authority_sha256",
        "market_closure_authority_file_sha256",
        "economic_step_manifest_sha256",
        "economics_objective_contract_sha256",
        "normalization_sha256",
        "normalization_file_sha256",
        "normalization_fit_scope",
        "normalization_val_mode",
        "model_architecture_schema_version",
        "model_architecture_sha256",
        "model_state_sha256",
        "checkpoint_file_sha256",
        "action_order",
        "tie_break",
        "rollout_order",
        "capacity_or_512_is_terminal",
        "economic_terminal_present",
        "compute_guard",
        "test_data_used",
    }
    guard = observed.get("compute_guard")
    if (
        set(observed) != required
        or claimed != _canonical_sha256(observed)
        or observed["schema_version"] != VAL_ROLLOUT_CONTRACT_SCHEMA_VERSION
        or observed["decision"] != "PASS"
        or observed["split"] != "val"
        or observed["entry_pair_cohort_size"] != VAL_ENTRY_COHORT_SIZE
        or observed["both_sides_evaluated"] is not True
        or observed["action_order"] != list(EXIT_ACTION_ORDER)
        or observed["tie_break"] != "unique_argmax_or_fail_closed"
        or observed["capacity_or_512_is_terminal"] is not False
        or observed["economic_terminal_present"] is not False
        or observed["normalization_val_mode"] != "apply_frozen_train_transform_only"
        or observed["model_architecture_schema_version"]
        != RANDOM_ACCESS_MODEL_SCHEMA_VERSION
        or observed["model_architecture_sha256"] != RANDOM_ACCESS_MODEL_SCHEMA_SHA256
        or observed["test_data_used"] is not False
        or isinstance(observed["parent_m1_row_offset"], bool)
        or not isinstance(observed["parent_m1_row_offset"], int)
        or observed["parent_m1_row_offset"] < 0
        or not isinstance(guard, Mapping)
        or set(guard)
        != {
            "max_model_forwards",
            "max_materialized_state_views",
            "max_wall_seconds",
            "wall_limit_scope",
            "guard_stop_semantics",
        }
        or guard["wall_limit_scope"] not in {"invocation", "entire_rollout"}
        or guard["guard_stop_semantics"] != "hard_budget_truncated_wall_scope_explicit_never_terminal"
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_CONTRACT_INVALID")
    for field in (
        "entry_cohort_stream_sha256",
        "entry_decision_representations_sha256",
        "source_lineage_sha256",
        "m1_source_sha256",
        "m1_clock_sha256",
        "m1_source_manifest_file_sha256",
        "parent_m1_source_sha256",
        "parent_m1_source_manifest_sha256",
        "market_closure_authority_sha256",
        "market_closure_authority_file_sha256",
        "economic_step_manifest_sha256",
        "economics_objective_contract_sha256",
        "normalization_sha256",
        "normalization_file_sha256",
        "model_state_sha256",
        "checkpoint_file_sha256",
    ):
        _require_sha(observed[field], field.upper())
    observed["contract_sha256"] = claimed
    return observed


def _require_state(
    value: Mapping[str, Any], *, entry: Mapping[str, Any], state_index: int,
    _cached_market: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _STATE_KEYS:
        raise RuntimeError("UNIFIED_EXIT_VAL_STATE_SCHEMA_INVALID")
    state = dict(value)
    row = int(entry["entry_m1_start_row"]) + state_index
    expected_path_start = max(0, state_index - (TRADE_PATH_TAIL_MAX_ROWS - 1))
    expected_path_length = state_index - expected_path_start + 1
    local = state["m1_local_history_x"]
    path = state["trade_path_tail_x"]
    summary = state["lifetime_summary_x"]
    if (
        state["state_index"] != state_index
        or state["m1_row_index"] != row
        or state["m1_local_history_start_row"] != row - (M1_LOCAL_HISTORY_ROWS - 1)
        or state["trade_path_start_state_index"] != expected_path_start
        or state["trade_path_length"] != expected_path_length
        or (not _cached_market and (
            not isinstance(local, np.ndarray)
        or local.dtype != np.dtype("float32")
        or local.ndim != 2
        or local.shape[0] != M1_LOCAL_HISTORY_ROWS
        or not np.isfinite(local).all()
        or local.flags.writeable
        ))
        or not isinstance(path, np.ndarray)
        or path.dtype != np.dtype("float32")
        or path.ndim != 3
        or path.shape[:2] != (2, expected_path_length)
        or not np.isfinite(path).all()
        or path.flags.writeable
        or not isinstance(summary, np.ndarray)
        or summary.dtype != np.dtype("float64")
        or summary.shape != (2, LIFETIME_SUMMARY_DIM)
        or not np.isfinite(summary).all()
        or summary.flags.writeable
        or not isinstance(state["lifetime_summary_sha256_by_side"], list)
        or len(state["lifetime_summary_sha256_by_side"]) != 2
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_STATE_INVALID")
    for digest in state["lifetime_summary_sha256_by_side"]:
        _require_sha(digest, "LIFETIME_SUMMARY")
    if _cached_market:
        if any(state[name] is not None for name in
               ("m1_local_history_x", "state_ctx_cat", "state_ctx_cont", "mtf")):
            raise RuntimeError("UNIFIED_EXIT_VAL_CACHED_MARKET_INPUT_INVALID")
        return state
    mtf = state["mtf"]
    expected_mtf = {
        f"exit_mtf_{kind}_{tf.lower()}"
        for tf in MULTI_TF_TIMEFRAMES
        for kind in ("history", "history_time_ns", "gather")
    }
    if not isinstance(mtf, Mapping) or set(mtf) != expected_mtf:
        raise RuntimeError("UNIFIED_EXIT_VAL_STATE_MTF_INVALID")
    for tf in MULTI_TF_TIMEFRAMES:
        suffix = tf.lower()
        history = mtf[f"exit_mtf_history_{suffix}"]
        history_time = mtf[f"exit_mtf_history_time_ns_{suffix}"]
        gather = mtf[f"exit_mtf_gather_{suffix}"]
        if (
            not isinstance(history, np.ndarray)
            or history.dtype != np.dtype("float32")
            or history.ndim != 2
            or history.shape[0] < 1
            or not np.isfinite(history).all()
            or history.flags.writeable
            or not isinstance(history_time, np.ndarray)
            or history_time.dtype != np.dtype("int64")
            or history_time.shape != (history.shape[0],)
            or history_time.flags.writeable
            or not isinstance(gather, np.ndarray)
            or gather.dtype != np.dtype("int64")
            or gather.shape != (1,)
            or int(gather[0]) != history.shape[0] - 1
            or gather.flags.writeable
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_STATE_MTF_INVALID")
    return state


class RandomAccessValRolloutAdapterV1:
    """Materialize bound current states while closure gaps stay fail closed."""

    def __init__(
        self,
        *,
        contract: Mapping[str, Any],
        entries: Sequence[Mapping[str, Any]],
        m1_times: Sequence[Any],
        market_closure_authority: Mapping[str, Any],
        economic_step_provider: Any,
        economic_step_manifest: Mapping[str, Any],
        economics_objective_contract: Mapping[str, Any],
        normalization_artifact: Mapping[str, Any],
        state_provider: Callable[[Mapping[str, Any], int], Mapping[str, Any]],
    ) -> None:
        self.contract = require_random_access_val_rollout_contract(contract)
        self.entries = _require_entries(entries)
        self._entry_by_index = {row["entry_row_index"]: row for row in self.entries}
        times = pd.DatetimeIndex(
            pd.to_datetime(m1_times, utc=True, errors="coerce")
        ).as_unit("ns")
        if (
            times.empty
            or times.hasnans
            or not times.is_unique
            or not times.is_monotonic_increasing
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_M1_CLOCK_INVALID")
        if (
            _entry_stream_sha256(self.entries)
            != self.contract["entry_cohort_stream_sha256"]
            or m1_clock_sha256(times) != self.contract["m1_clock_sha256"]
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_SOURCE_BINDING_INVALID")
        closure = require_market_closure_authority(
            market_closure_authority,
            expected_m1_source_sha256=self.contract["m1_source_sha256"],
            expected_m1_clock_sha256=self.contract["m1_clock_sha256"],
        )
        normalization = require_lifetime_summary_normalization(normalization_artifact)
        objective = dict(economics_objective_contract)
        elapsed_wall_clock_gamma(
            contract=objective,
            elapsed_wall_clock_seconds=60,
        )
        if (
            closure["artifact_sha256"]
            != self.contract["market_closure_authority_sha256"]
            or economic_step_manifest.get("manifest_sha256")
            != self.contract["economic_step_manifest_sha256"]
            or objective["contract_sha256"]
            != self.contract["economics_objective_contract_sha256"]
            or normalization["normalization_sha256"]
            != self.contract["normalization_sha256"]
            or normalization["val_mode"] != "apply_frozen_train_transform_only"
            or getattr(economic_step_provider, "market_closure_authority_sha256", None)
            != closure["artifact_sha256"]
            or getattr(economic_step_provider, "state_m1_source_sha256", None)
            != self.contract["m1_source_sha256"]
            or getattr(economic_step_provider, "state_m1_source_manifest_sha256", None)
            != self.contract["m1_source_manifest_file_sha256"]
            or getattr(economic_step_provider, "parent_m1_source_sha256", None)
            != self.contract["parent_m1_source_sha256"]
            or getattr(economic_step_provider, "parent_m1_source_manifest_sha256", None)
            != self.contract["parent_m1_source_manifest_sha256"]
            or getattr(economic_step_provider, "parent_m1_row_offset", None)
            != self.contract["parent_m1_row_offset"]
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_SOURCE_BINDING_INVALID")
        for entry in self.entries:
            stop = entry["entry_m1_start_row"] + entry["available_state_count"]
            if stop > len(times):
                raise RuntimeError("UNIFIED_EXIT_VAL_ENTRY_RANGE_INVALID")
        self.times = times
        self.closure = closure
        self.closure_by_row = closure_intervals_by_gap_after_row(closure)
        self.economic_step_provider = economic_step_provider
        self.economic_step_manifest = dict(economic_step_manifest)
        self.objective = objective
        self.normalization = normalization
        self.state_provider = state_provider
        self._state_hash_pool: ThreadPoolExecutor | None = None

    def _materialize_unsealed_state(
        self, entry_row_index: int, state_index: int, *,
        _prepared_cached_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if entry_row_index not in self._entry_by_index:
            raise RuntimeError("UNIFIED_EXIT_VAL_ENTRY_IDENTITY_INVALID")
        entry = self._entry_by_index[entry_row_index]
        if (
            isinstance(state_index, bool)
            or not isinstance(state_index, int)
            or state_index < 0
            or state_index >= entry["available_state_count"]
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_STATE_INDEX_INVALID")
        state = _require_state(
            self.state_provider(entry, state_index) if _prepared_cached_state is None else _prepared_cached_state,
            entry=entry,
            state_index=state_index,
            _cached_market=_prepared_cached_state is not None,
        )
        row = entry["entry_m1_start_row"] + state_index
        if state["bar_start_time_ns"] != int(self.times.asi8[row]) or state[
            "decision_time_ns"
        ] != int(self.times.asi8[row] + 60_000_000_000):
            raise RuntimeError("UNIFIED_EXIT_VAL_STATE_CLOCK_INVALID")
        successor_available = False
        censor_reason: str | None = None
        transition: dict[str, Any] | None = None
        if state_index + 1 < entry["available_state_count"]:
            delta_ns = int(self.times.asi8[row + 1] - self.times.asi8[row])
            if delta_ns == 60_000_000_000:
                successor_available = True
                transition = {
                    "classification": "continuous_observed_m1",
                    "closure_kind": None,
                    "interval_sha256": None,
                    "wall_clock_delta_seconds": 60,
                }
            else:
                interval = self.closure_by_row.get(row)
                if interval is None:
                    raise RuntimeError("UNIFIED_EXIT_VAL_GAP_AUTHORITY_MISSING")
                if interval["successor_across_gap_allowed"] is True:
                    successor_available = True
                    transition = {
                        "classification": interval["classification"],
                        "closure_kind": interval["closure_kind"],
                        "interval_sha256": interval["interval_sha256"],
                        "wall_clock_delta_seconds": delta_ns // 1_000_000_000,
                    }
                else:
                    raise RuntimeError("UNIFIED_EXIT_VAL_LIFECYCLE_CROSSES_UNKNOWN_GAP")
        else:
            if row + 1 >= len(self.times):
                censor_reason = "split_end"
            else:
                interval = self.closure_by_row.get(row)
                if interval is None or interval["successor_across_gap_allowed"] is True:
                    raise RuntimeError("UNIFIED_EXIT_VAL_LIFECYCLE_PREMATURE_STOP")
                censor_reason = "unknown_source_gap"
        result = {
            "schema_version": VAL_ROLLOUT_STATE_SCHEMA_VERSION,
            "contract_sha256": self.contract["contract_sha256"],
            "entry_row_index": entry_row_index,
            "state_index": state_index,
            "state": state,
            "successor_observed": successor_available,
            "right_censor_reason_if_hold": censor_reason,
            "transition_closure": transition,
            "capacity_or_tail_length_is_terminal": False,
            "economic_terminal": False,
            "test_data_used": False,
        }
        return result

    def materialize_state(
        self, entry_row_index: int, state_index: int
    ) -> dict[str, Any]:
        result = self._materialize_unsealed_state(entry_row_index, state_index)
        result["state_envelope_sha256"] = _canonical_sha256(result)
        return result

    def materialize_active_batch(
        self, entry_row_indices: Sequence[int], state_index: int
    ) -> list[dict[str, Any]]:
        if len(entry_row_indices) <= 1:
            return [
                self.materialize_state(entry, state_index)
                for entry in entry_row_indices
            ]
        # State providers keep their original serial order. Only hashing the
        # completed, immutable envelopes runs concurrently; SHA-256 releases
        # the GIL for the large array buffers observed in the VAL profile.
        envelopes = [
            self._materialize_unsealed_state(entry, state_index)
            for entry in entry_row_indices
        ]
        if self._state_hash_pool is None:
            self._state_hash_pool = ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="gx1-val-hash"
            )
            weakref.finalize(self, self._state_hash_pool.shutdown, wait=False)
        digests = list(self._state_hash_pool.map(_canonical_sha256, envelopes))
        for envelope, digest in zip(envelopes, digests):
            envelope["state_envelope_sha256"] = digest
        return envelopes

    def materialize_cached_active_batch(
        self, entry_row_indices: Sequence[int], state_index: int, *,
        cached_market_rows: set[int], workers: int,
    ) -> list[dict[str, Any]]:
        from gx1.contracts.unified_exit_random_access_val_factory_v1 import RandomAccessValStateFactoryV1
        factory = getattr(self.state_provider, "__self__", None)
        if not isinstance(factory, RandomAccessValStateFactoryV1):
            raise RuntimeError("UNIFIED_EXIT_VAL_CPU_FACTORY_REQUIRED")
        hits = [entry for entry in entry_row_indices
                if self._entry_by_index[entry]["entry_m1_start_row"] + state_index in cached_market_rows]
        cached = factory.materialize_cached_cpu_batch(
            [(self._entry_by_index[entry], state_index) for entry in hits], workers=workers,
        )
        prepared = dict(zip(hits, cached))
        envelopes = [self._materialize_unsealed_state(
            entry, state_index, _prepared_cached_state=prepared.get(entry),
        ) for entry in entry_row_indices]
        if self._state_hash_pool is None:
            self._state_hash_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gx1-val-hash")
            weakref.finalize(self, self._state_hash_pool.shutdown, wait=False)
        for envelope, digest in zip(envelopes, self._state_hash_pool.map(_canonical_sha256, envelopes)):
            envelope["state_envelope_sha256"] = digest
        return envelopes

    def compose_selected_action(
        self,
        *,
        entry_row_index: int,
        side_index: int,
        state_index: int,
        action: str,
    ) -> tuple[dict[str, Any], str]:
        if side_index not in (0, 1) or action not in {"hold", "exit_now"}:
            raise RuntimeError("UNIFIED_EXIT_VAL_ACTION_INVALID")
        envelope = self.economic_step_provider(
            entry_row_index,
            side_index,
            action,
            state_index,
            state_index + 1,
        )
        return self._compose_selected_envelope(
            envelope, entry_row_index=entry_row_index, side_index=side_index,
            state_index=state_index, action=action,
        )

    def compose_selected_actions(self, requests: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
        batch = getattr(self.economic_step_provider, "materialize_selected_actions", None)
        if batch is None:
            return [self.compose_selected_action(**request) for request in requests]
        envelopes = batch(requests)
        if len(envelopes) != len(requests):
            raise RuntimeError("UNIFIED_EXIT_VAL_ECONOMIC_BATCH_INVALID")
        cache = getattr(self, "_val_composed_hold_cache", None)
        if cache is None:
            cache = self._val_composed_hold_cache = {}
        objective_sha = _economic_json_sha256(self.objective)
        return [self._compose_selected_envelope(
            envelope, **request, _hold_cache=cache, _objective_sha=objective_sha,
        ) for request, envelope in zip(requests, envelopes)]

    def _compose_selected_envelope(
        self, envelope: Mapping[str, Any], *, entry_row_index: int, side_index: int,
        state_index: int, action: str, _hold_cache: dict | None = None,
        _objective_sha: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        if not isinstance(envelope, Mapping) or "slice_sha256" not in envelope:
            raise RuntimeError("UNIFIED_EXIT_VAL_ECONOMIC_SLICE_INVALID")
        # Economic envelopes are JSON-only and already sealed by the provider
        # with this owner; array projection belongs only to market-state hashes.
        raw = dict(envelope)
        claimed = raw.pop("slice_sha256")
        if (
            claimed != _economic_json_sha256(raw)
            or raw.get("entry_row_index") != entry_row_index
            or raw.get("side_index") != side_index
            or raw.get("action") != action
            or raw.get("start_state_index") != state_index
            or raw.get("stop_state_index") != state_index + 1
            or len(raw.get("steps", ())) != 1
            or raw.get("economic_step_model_sha256")
            != self.economic_step_manifest.get("economic_step_model_sha256")
            or raw.get("economic_step_source_manifest_sha256")
            != self.economic_step_manifest.get("economic_step_source_manifest_sha256")
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_ECONOMIC_SLICE_INVALID")
        cache_step = raw["steps"][0]
        mark = cache_step.get("successor_liquidation_value")
        if isinstance(mark, Mapping) and action == "hold":
            # Cache interval costs once. Entry-specific marks must not multiply
            # cache size by the number of overlapping counterfactual entries.
            cache_step = {**cache_step, "successor_liquidation_value": {**mark, "value_bps": 0.0}}
        cache_key = (_objective_sha, _economic_json_sha256(cache_step)) if _hold_cache is not None and action == "hold" else None
        if cache_key is not None and cache_key in _hold_cache:
            composed = _hold_cache[cache_key]
            if isinstance(mark, Mapping):
                composed = revalue_marked_hold_step(composed, successor_liquidation_value=mark)
        else:
            composed = compose_economic_step(raw["steps"][0], contract=self.objective)
            if cache_key is not None:
                _hold_cache[cache_key] = composed
        expected_kind = "HOLD" if action == "hold" else "EXIT_NOW"
        if composed["event_kind"] != expected_kind:
            raise RuntimeError("UNIFIED_EXIT_VAL_ECONOMIC_EVENT_INVALID")
        return composed, claimed


def unique_active_exit_actions(q: torch.Tensor, active_side_mask: Any) -> np.ndarray:
    active = torch.as_tensor(active_side_mask, device=q.device)
    if (
        q.ndim != 3 or q.shape[-1] != 2 or active.shape != q.shape[:2]
        or active.dtype != torch.bool or not bool(active.any().item())
        or not bool(torch.isfinite(q).all().item())
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_MODEL_OUTPUT_INVALID")
    active_q = q[active]
    if bool((active_q.eq(active_q.amax(dim=1, keepdim=True)).sum(dim=1) != 1).any().item()):
        raise RuntimeError("UNIFIED_EXIT_VAL_MODEL_TIED_ACTION")
    return torch.argmax(q, dim=2).detach().cpu().numpy()


def run_random_access_val_rollout(
    *,
    model: nn.Module,
    entry_decision_representations: torch.Tensor,
    adapter: RandomAccessValRolloutAdapterV1,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Roll both sides until learned EXIT, censor, or an explicit compute guard."""

    contract = adapter.contract
    if contract["compute_guard"]["wall_limit_scope"] == "invocation":
        raise RuntimeError("UNIFIED_EXIT_VAL_RESUMABLE_EVALUATOR_REQUIRED")
    entries = adapter.entries
    if (
        model.training
        or getattr(model, "unified_exit_random_access_architecture_version", None)
        != RANDOM_ACCESS_MODEL_SCHEMA_VERSION
        or not isinstance(
            dict(model.state_dict()).get(
                "unified_exit_random_access_architecture_sha256"
            ),
            torch.Tensor,
        )
        or bytes(
            dict(model.state_dict())["unified_exit_random_access_architecture_sha256"]
            .detach()
            .cpu()
            .tolist()
        ).hex()
        != RANDOM_ACCESS_MODEL_SCHEMA_SHA256
        or canonical_model_state_sha256(model.state_dict())
        != contract["model_state_sha256"]
        or _tensor_sha256(entry_decision_representations)
        != contract["entry_decision_representations_sha256"]
        or int(entry_decision_representations.shape[0]) != VAL_ENTRY_COHORT_SIZE
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_MODEL_BINDING_INVALID")
    device = entry_decision_representations.device
    guard = contract["compute_guard"]
    accumulators = [
        [
            {
                "status": "OPEN",
                "decision_count": 0,
                "hold_count": 0,
                "hold_wall_clock_seconds": 0,
                "undiscounted_net_cash_pnl_bps": 0.0,
                "undiscounted_risk_utility_penalty_bps": 0.0,
                "discounted_risk_adjusted_utility_bps": 0.0,
                "continuation_discount": 1.0,
                "selected_economic_slice_sha256": [],
                "exit_state_index": None,
                "exit_decision_time_ns": None,
            }
            for _side in range(2)
        ]
        for _entry in entries
    ]
    active = np.ones((VAL_ENTRY_COHORT_SIZE, 2), dtype=np.bool_)
    state_index = 0
    forward_count = 0
    materialized_count = 0
    compaction_trace: list[dict[str, int]] = []
    guard_reason: str | None = None
    started = monotonic()
    while bool(active.any()):
        active_entries = np.flatnonzero(active.any(axis=1))
        if forward_count >= guard["max_model_forwards"]:
            guard_reason = "max_model_forwards"
            break
        if (
            materialized_count + len(active_entries)
            > guard["max_materialized_state_views"]
        ):
            guard_reason = "max_materialized_state_views"
            break
        if monotonic() - started >= guard["max_wall_seconds"]:
            guard_reason = "max_wall_seconds"
            break
        row_indices = [
            entries[int(position)]["entry_row_index"] for position in active_entries
        ]
        envelopes = adapter.materialize_active_batch(row_indices, state_index)
        materialized_count += len(envelopes)
        states = [envelope["state"] for envelope in envelopes]
        model_inputs = collate_random_access_states_v1(
            states,
            normalization_artifact=adapter.normalization,
            device=device,
        )
        selected = torch.as_tensor(active_entries, dtype=torch.long, device=device)
        model_inputs["entry_decision_representation"] = (
            entry_decision_representations.index_select(0, selected)
        )
        model_inputs["action_valid_mask"] = torch.ones(
            (len(envelopes), 2, 2), dtype=torch.bool, device=device
        )
        with torch.inference_mode():
            output = model.forward_exit_random_access_batch(**model_inputs)
        q = output.get("exit_action_q_bps")
        valid = output.get("exit_action_valid_mask")
        if (
            not isinstance(q, torch.Tensor)
            or tuple(q.shape) != (len(envelopes), 2, 2)
            or not bool(torch.isfinite(q).all().item())
            or not isinstance(valid, torch.Tensor)
            or valid.dtype != torch.bool
            or tuple(valid.shape) != tuple(q.shape)
            or not bool(valid.all().item())
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_MODEL_OUTPUT_INVALID")
        actions = unique_active_exit_actions(q, active[active_entries])
        compaction_trace.append(
            {
                "state_index": state_index,
                "active_entry_count": len(envelopes),
                "active_side_count": int(active[active_entries].sum()),
            }
        )
        forward_count += 1
        for batch_position, entry_position_raw in enumerate(active_entries):
            entry_position = int(entry_position_raw)
            envelope = envelopes[batch_position]
            for side in range(2):
                if not active[entry_position, side]:
                    continue
                accumulator = accumulators[entry_position][side]
                accumulator["decision_count"] += 1
                if int(actions[batch_position, side]) == 1:
                    step, slice_sha = adapter.compose_selected_action(
                        entry_row_index=entries[entry_position]["entry_row_index"],
                        side_index=side,
                        state_index=state_index,
                        action="exit_now",
                    )
                    _accumulate_step(accumulator, step, slice_sha)
                    accumulator["status"] = "EXITED"
                    accumulator["exit_state_index"] = state_index
                    accumulator["exit_decision_time_ns"] = envelope["state"][
                        "decision_time_ns"
                    ]
                    active[entry_position, side] = False
                elif envelope["successor_observed"]:
                    step, slice_sha = adapter.compose_selected_action(
                        entry_row_index=entries[entry_position]["entry_row_index"],
                        side_index=side,
                        state_index=state_index,
                        action="hold",
                    )
                    _accumulate_step(accumulator, step, slice_sha)
                    accumulator["hold_count"] += 1
                    accumulator["hold_wall_clock_seconds"] += step[
                        "elapsed_wall_clock_seconds"
                    ]
                else:
                    reason = envelope["right_censor_reason_if_hold"]
                    if reason not in {"split_end", "unknown_source_gap"}:
                        raise RuntimeError("UNIFIED_EXIT_VAL_CENSOR_REASON_INVALID")
                    accumulator["status"] = f"RIGHT_CENSORED_{reason.upper()}"
                    active[entry_position, side] = False
        state_index += 1
    if guard_reason is not None:
        for entry_position, side in zip(*np.nonzero(active)):
            accumulators[int(entry_position)][int(side)]["status"] = (
                f"TRUNCATED_COMPUTE_GUARD_{guard_reason.upper()}"
            )
            active[int(entry_position), int(side)] = False
    outcomes: list[dict[str, Any]] = []
    for entry_position, entry in enumerate(entries):
        for side in range(2):
            row = {
                "entry_row_index": entry["entry_row_index"],
                "side_index": side,
                **accumulators[entry_position][side],
                "economic_terminal": False,
                "capacity_or_512_terminal": False,
            }
            row["outcome_sha256"] = _canonical_sha256(row)
            outcomes.append(row)
    status_counts: dict[str, int] = {}
    for row in outcomes:
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
    exited = status_counts.get("EXITED", 0)
    decision = (
        "TRUNCATED_NON_AUTHORITATIVE"
        if truncated
        else "COMPLETE_WITH_RIGHT_CENSORING"
        if censored
        else "PASS_COMPLETE"
    )
    result = {
        "schema_version": VAL_ROLLOUT_RESULT_SCHEMA_VERSION,
        "decision": decision,
        "contract_sha256": contract["contract_sha256"],
        "entry_pair_cohort_size": VAL_ENTRY_COHORT_SIZE,
        "side_trade_count": VAL_ENTRY_COHORT_SIZE * 2,
        "exited_side_trade_count": exited,
        "right_censored_side_trade_count": censored,
        "compute_truncated_side_trade_count": truncated,
        "economic_terminal_count": 0,
        "capacity_or_512_terminal_count": 0,
        "model_forward_count": forward_count,
        "materialized_state_view_count": materialized_count,
        "max_decision_state_index": max(
            (row["state_index"] for row in compaction_trace), default=None
        ),
        "compaction_trace": compaction_trace,
        "trade_outcomes": outcomes,
        "compute_guard_triggered": guard_reason,
        "rollout_execution_complete": truncated == 0,
        "full_cohort_policy_metrics_authoritative": truncated == 0 and censored == 0,
        "test_data_used": False,
    }
    result["result_sha256"] = _canonical_sha256(result)
    return result


def require_random_access_val_rollout_result(
    value: Mapping[str, Any], *, contract: Mapping[str, Any]
) -> dict[str, Any]:
    checked_contract = require_random_access_val_rollout_contract(contract)
    if not isinstance(value, Mapping) or "result_sha256" not in value:
        raise RuntimeError("UNIFIED_EXIT_VAL_RESULT_INVALID")
    observed = dict(value)
    claimed = observed.pop("result_sha256")
    required = {
        "schema_version",
        "decision",
        "contract_sha256",
        "entry_pair_cohort_size",
        "side_trade_count",
        "exited_side_trade_count",
        "right_censored_side_trade_count",
        "compute_truncated_side_trade_count",
        "economic_terminal_count",
        "capacity_or_512_terminal_count",
        "model_forward_count",
        "materialized_state_view_count",
        "max_decision_state_index",
        "compaction_trace",
        "trade_outcomes",
        "compute_guard_triggered",
        "rollout_execution_complete",
        "full_cohort_policy_metrics_authoritative",
        "test_data_used",
    }
    outcomes = observed.get("trade_outcomes")
    total = VAL_ENTRY_COHORT_SIZE * 2
    if (
        set(observed) != required
        or claimed != _canonical_sha256(observed)
        or observed["schema_version"] != VAL_ROLLOUT_RESULT_SCHEMA_VERSION
        or observed["contract_sha256"] != checked_contract["contract_sha256"]
        or observed["entry_pair_cohort_size"] != VAL_ENTRY_COHORT_SIZE
        or observed["side_trade_count"] != total
        or not isinstance(outcomes, list)
        or len(outcomes) != total
        or observed["economic_terminal_count"] != 0
        or observed["capacity_or_512_terminal_count"] != 0
        or observed["test_data_used"] is not False
        or sum(
            observed[name]
            for name in (
                "exited_side_trade_count",
                "right_censored_side_trade_count",
                "compute_truncated_side_trade_count",
            )
        )
        != total
    ):
        raise RuntimeError("UNIFIED_EXIT_VAL_RESULT_INVALID")
    for outcome in outcomes:
        if not isinstance(outcome, Mapping) or "outcome_sha256" not in outcome:
            raise RuntimeError("UNIFIED_EXIT_VAL_RESULT_INVALID")
        row = dict(outcome)
        row_claim = row.pop("outcome_sha256")
        if (
            row_claim != _canonical_sha256(row)
            or outcome.get("economic_terminal") is not False
            or outcome.get("capacity_or_512_terminal") is not False
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_RESULT_INVALID")
    observed["result_sha256"] = claimed
    return observed


def _accumulate_step(
    accumulator: dict[str, Any], step: Mapping[str, Any], slice_sha: str
) -> None:
    discount = float(accumulator["continuation_discount"])
    accumulator["undiscounted_net_cash_pnl_bps"] += step[
        "undiscounted_net_cash_pnl_increment_bps"
    ]
    accumulator["undiscounted_risk_utility_penalty_bps"] += step[
        "risk_utility_penalty_bps"
    ]
    accumulator["discounted_risk_adjusted_utility_bps"] += (
        discount * step["undiscounted_risk_adjusted_utility_increment_bps"]
    )
    accumulator["continuation_discount"] = discount * step["continuation_gamma"]
    accumulator["selected_economic_slice_sha256"].append(slice_sha)


__all__ = (
    "RandomAccessValRolloutAdapterV1",
    "VAL_ENTRY_COHORT_SIZE",
    "VAL_ROLLOUT_CONTRACT_SCHEMA_VERSION",
    "VAL_ROLLOUT_RESULT_SCHEMA_VERSION",
    "VAL_ROLLOUT_STATE_SCHEMA_VERSION",
    "build_random_access_val_rollout_contract",
    "require_random_access_val_rollout_contract",
    "require_random_access_val_rollout_result",
    "run_random_access_val_rollout",
)
