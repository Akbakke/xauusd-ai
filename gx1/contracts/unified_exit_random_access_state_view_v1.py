"""Causal rolling state views for sampled unbounded Exit transitions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.unified_exit_dataset_adapter_v2 import (
    require_economic_training_projection,
)
from gx1.contracts.unified_exit_economics_objective_v2 import (
    elapsed_wall_clock_gamma,
)
from gx1.contracts.unified_exit_lifetime_summary_v1 import (
    LIFETIME_SUMMARY_DIM,
    lifetime_summary_registry,
    require_lifetime_summary,
)
from gx1.contracts.unified_exit_market_closure_authority_v1 import (
    m1_clock_sha256,
    require_market_closure_authority,
)
from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    require_random_access_sample,
    require_random_access_sampler_contract,
)


RANDOM_ACCESS_STATE_VIEW_SCHEMA_VERSION = (
    "gx1_unified_exit_random_access_state_view_v1"
)
EXIT_ACTION_ORDER = ("HOLD", "EXIT_NOW")
M1_LOCAL_HISTORY_ROWS = 480
TRADE_PATH_TAIL_MAX_ROWS = 512


def _structured_sha256(value: Any) -> str:
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


def _readonly_array(
    value: Any, *, dtype: np.dtype[Any], shape: tuple[int, ...], label: str
) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.shape != shape or (array.dtype.kind == "f" and not np.isfinite(array).all()):
        raise RuntimeError(f"UNIFIED_EXIT_RANDOM_ACCESS_{label}_INVALID")
    array.setflags(write=False)
    return array


def _closure_for_transition(
    *,
    authority: Mapping[str, Any],
    current_row: int,
    current_time_ns: int,
    successor_time_ns: int,
) -> dict[str, Any]:
    delta_ns = successor_time_ns - current_time_ns
    if delta_ns == 60_000_000_000:
        return {
            "classification": "continuous_observed_m1",
            "closure_kind": None,
            "interval_sha256": None,
            "wall_clock_delta_seconds": 60,
        }
    if delta_ns <= 60_000_000_000:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_SUCCESSOR_CLOCK_INVALID")
    matches = [
        item
        for item in authority["intervals"]
        if item["gap_after_m1_row"] == current_row
    ]
    if len(matches) != 1 or matches[0]["successor_across_gap_allowed"] is not True:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_SUCCESSOR_GAP_CENSORED")
    interval = matches[0]
    if (
        pd.Timestamp(interval["previous_bar_start_utc"]).value != current_time_ns
        or pd.Timestamp(interval["next_bar_start_utc"]).value != successor_time_ns
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_SUCCESSOR_GAP_IDENTITY_INVALID")
    return {
        "classification": interval["classification"],
        "closure_kind": interval["closure_kind"],
        "interval_sha256": interval["interval_sha256"],
        "wall_clock_delta_seconds": int(delta_ns // 1_000_000_000),
    }


def materialize_random_access_state_view(
    *,
    sampler_contract: Mapping[str, Any],
    sample: Mapping[str, Any],
    entry_row_index: int,
    entry_m1_start_row: int,
    side_lifecycle_state_counts: Sequence[int],
    m1_times: Sequence[Any],
    m1_signal: np.ndarray,
    m1_ctx_cont: np.ndarray,
    m1_ctx_cat: np.ndarray,
    m1_source_sha256: str,
    market_closure_authority: Mapping[str, Any],
    path_detail_provider: Callable[[int, int, int], np.ndarray],
    lifetime_summary_provider: Callable[[int, int], Mapping[str, Any]],
    mtf_materializer: Callable[[np.ndarray], Mapping[str, np.ndarray]],
    economic_step_provider: Any,
    economic_step_manifest: Mapping[str, Any],
    economics_objective_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize exact t/t+1 views; memory limits never create terminals."""

    contract = require_random_access_sampler_contract(sampler_contract)
    scheduled = require_random_access_sample(sample, sampler_contract=contract)
    if scheduled["entry_row_index"] != entry_row_index:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_ENTRY_IDENTITY_INVALID")
    counts = tuple(side_lifecycle_state_counts)
    if (
        len(counts) != 2
        or any(isinstance(count, bool) or not isinstance(count, int) for count in counts)
        or any(count < 2 for count in counts)
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_SIDE_COUNT_INVALID")
    state_index = scheduled["state_index"]
    successor_index = scheduled["successor_state_index"]
    if any(successor_index >= count for count in counts):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_COMMON_TIMELINE_INVALID")

    times = pd.DatetimeIndex(pd.to_datetime(m1_times, utc=True, errors="coerce")).as_unit(
        "ns"
    )
    signal = np.asarray(m1_signal)
    cont = np.asarray(m1_ctx_cont)
    cat = np.asarray(m1_ctx_cat)
    if (
        times.empty
        or times.hasnans
        or not times.is_unique
        or not times.is_monotonic_increasing
        or signal.ndim != 2
        or cont.ndim != 2
        or cat.ndim != 2
        or not len(times) == len(signal) == len(cont) == len(cat)
        or not np.isfinite(signal).all()
        or not np.isfinite(cont).all()
        or entry_m1_start_row < M1_LOCAL_HISTORY_ROWS - 1
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_M1_SOURCE_INVALID")
    source_clock_sha = m1_clock_sha256(times)
    authority = require_market_closure_authority(
        market_closure_authority,
        expected_m1_source_sha256=m1_source_sha256,
        expected_m1_clock_sha256=source_clock_sha,
    )
    state_row = entry_m1_start_row + state_index
    successor_row = entry_m1_start_row + successor_index
    if successor_row >= len(times):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_SUCCESSOR_MISSING")
    closure = _closure_for_transition(
        authority=authority,
        current_row=state_row,
        current_time_ns=int(times.asi8[state_row]),
        successor_time_ns=int(times.asi8[successor_row]),
    )

    def one_view(index: int, row: int) -> dict[str, Any]:
        local_start = row - (M1_LOCAL_HISTORY_ROWS - 1)
        path_start = max(0, index - (TRADE_PATH_TAIL_MAX_ROWS - 1))
        path_stop = index + 1
        path_length = path_stop - path_start
        raw_paths = [
            np.asarray(path_detail_provider(side, path_start, path_stop))
            for side in range(2)
        ]
        if (
            any(path.ndim != 2 or path.shape[0] != path_length for path in raw_paths)
            or raw_paths[0].shape[1] != raw_paths[1].shape[1]
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_PATH_DETAIL_INVALID")
        paths = np.stack(
            [
                _readonly_array(
                    path,
                    dtype=np.dtype("<f4"),
                    shape=(path_length, raw_paths[0].shape[1]),
                    label="PATH_DETAIL",
                )
                for path in raw_paths
            ],
            axis=0,
        )
        summaries = [
            require_lifetime_summary(lifetime_summary_provider(side, index))
            for side in range(2)
        ]
        summary_values = _readonly_array(
            np.stack([item["values"] for item in summaries], axis=0),
            dtype=np.dtype("<f8"),
            shape=(2, LIFETIME_SUMMARY_DIM),
            label="LIFETIME_SUMMARY",
        )
        state_time = np.asarray([times.asi8[row]], dtype=np.int64)
        mtf_raw = mtf_materializer(state_time)
        if not isinstance(mtf_raw, Mapping) or not mtf_raw:
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_MTF_INVALID")
        mtf: dict[str, np.ndarray] = {}
        for name, raw in mtf_raw.items():
            array = np.ascontiguousarray(raw)
            if array.shape[0] != 1 or (
                array.dtype.kind == "f" and not np.isfinite(array).all()
            ):
                raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_MTF_INVALID")
            array.setflags(write=False)
            mtf[str(name)] = array
        return {
            "state_index": index,
            "m1_row_index": row,
            "bar_start_time_ns": int(times.asi8[row]),
            "decision_time_ns": int(times.asi8[row] + 60_000_000_000),
            "m1_local_history_start_row": local_start,
            "m1_local_history_x": _readonly_array(
                signal[local_start : row + 1],
                dtype=np.dtype("<f4"),
                shape=(M1_LOCAL_HISTORY_ROWS, signal.shape[1]),
                label="M1_LOCAL_HISTORY",
            ),
            "state_ctx_cont": _readonly_array(
                cont[row],
                dtype=np.dtype("<f4"),
                shape=(cont.shape[1],),
                label="STATE_CONTEXT",
            ),
            "state_ctx_cat": _readonly_array(
                cat[row],
                dtype=np.dtype("<i8"),
                shape=(cat.shape[1],),
                label="STATE_CATEGORICAL_CONTEXT",
            ),
            "trade_path_start_state_index": path_start,
            "trade_path_length": path_length,
            "trade_path_tail_x": _readonly_array(
                paths,
                dtype=np.dtype("<f4"),
                shape=paths.shape,
                label="PATH_DETAIL",
            ),
            "lifetime_summary_x": summary_values,
            "lifetime_summary_sha256_by_side": [
                item["summary_sha256"] for item in summaries
            ],
            "mtf": mtf,
        }

    current = one_view(state_index, state_row)
    successor = one_view(successor_index, successor_row)
    rewards = np.empty((2, 2), dtype=np.float32)
    economics_hashes: list[dict[str, str]] = []
    fastpath = getattr(economic_step_provider, "materialize_training_projection", None)
    if not callable(fastpath):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_ECONOMIC_PROVIDER_INVALID")
    for side in range(2):
        projection = require_economic_training_projection(
            fastpath(entry_row_index, side, state_index, state_index + 1, state_index + 1),
            entry_row_index=entry_row_index,
            side_index=side,
            start_state_index=state_index,
            stop_state_index=state_index + 1,
            hold_stop_state_index=state_index + 1,
            economic_manifest=economic_step_manifest,
        )
        if (
            int(projection["exit_event_kind_index"][0]) != 0
            or int(projection["hold_event_kind_index"][0]) != 1
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_ECONOMIC_EVENT_INVALID")
        rewards[side, 0] = projection["hold_reward_bps"][0]
        rewards[side, 1] = projection["exit_reward_bps"][0]
        economics_hashes.append(
            {
                "projection_sha256": projection["projection_sha256"],
                "exit_stream_sha256": projection["exit_stream_sha256"],
                "hold_stream_sha256": projection["hold_stream_sha256"],
            }
        )
    rewards.setflags(write=False)
    valid = np.ones((2, 2), dtype=np.bool_)
    valid.setflags(write=False)
    bellman = valid.copy()
    bellman.setflags(write=False)
    successor_observed = np.ones(2, dtype=np.bool_)
    successor_observed.setflags(write=False)
    gamma = elapsed_wall_clock_gamma(
        contract=economics_objective_contract,
        elapsed_wall_clock_seconds=closure["wall_clock_delta_seconds"],
    )
    view = {
        "schema_version": RANDOM_ACCESS_STATE_VIEW_SCHEMA_VERSION,
        "action_order": list(EXIT_ACTION_ORDER),
        "sampler_contract_sha256": contract["contract_sha256"],
        "sample_sha256": scheduled["sample_sha256"],
        "entry_row_index": entry_row_index,
        "entry_m1_start_row": entry_m1_start_row,
        "m1_source_sha256": m1_source_sha256,
        "m1_clock_sha256": source_clock_sha,
        "market_closure_authority_sha256": authority["artifact_sha256"],
        "lifetime_summary_registry_sha256": lifetime_summary_registry()[
            "registry_sha256"
        ],
        "economic_step_manifest_sha256": economic_step_manifest["manifest_sha256"],
        "economics_objective_contract_sha256": economics_objective_contract[
            "contract_sha256"
        ],
        "current": current,
        "successor": successor,
        "transition_closure": closure,
        "elapsed_wall_clock_gamma": gamma,
        "immediate_reward_bps": rewards,
        "policy_action_valid_mask": valid,
        "bellman_target_valid_mask": bellman,
        "successor_observed_mask": successor_observed,
        "economic_projection_hashes_by_side": economics_hashes,
        "terminal_mask": [False, False],
        "right_censored_mask": [False, False],
        "capacity_or_tail_length_is_terminal": False,
        "test_data_used": False,
    }
    view["state_view_sha256"] = _structured_sha256(view)
    return view


__all__ = (
    "EXIT_ACTION_ORDER",
    "M1_LOCAL_HISTORY_ROWS",
    "RANDOM_ACCESS_STATE_VIEW_SCHEMA_VERSION",
    "TRADE_PATH_TAIL_MAX_ROWS",
    "materialize_random_access_state_view",
)
