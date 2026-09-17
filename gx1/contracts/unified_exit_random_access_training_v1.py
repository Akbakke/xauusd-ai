"""Batched Bellman training for sampled random-access Exit states."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from gx1.contracts.entry_fitted_q_v1 import build_entry_fitted_q_targets
from gx1.contracts.entry_model_native_input_normalization_v1 import (
    apply_surface_normalization,
)
from gx1.contracts.unified_exit_fitted_q_v1 import (
    build_unified_exit_fitted_q_targets,
    unified_exit_first_state_side_values,
)
from gx1.contracts.unified_exit_lifetime_summary_v1 import LIFETIME_SUMMARY_DIM
from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    require_lifetime_summary_normalization,
)
from gx1.contracts.unified_exit_ragged_batch_v1 import (
    flatten_sample_counts,
    one_forward_one_backward,
    right_pad_arrays,
)
from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    require_random_access_entry_anchor,
    require_random_access_sample,
    require_random_access_sampler_contract,
)
from gx1.contracts.unified_exit_reference_policy_v1 import (
    require_reference_policy_contract, build_reference_policy_hold_targets, reference_policy_state_values,
)
from gx1.contracts.unified_exit_random_access_state_view_v1 import (
    TRADE_PATH_TAIL_MAX_ROWS,
    LIQUIDATION_RELATIVE_STATE_VIEW_SCHEMA_VERSION,
    FROZEN_POLICY_TRACE_STATE_VIEW_SCHEMA_VERSION,
    REFERENCE_POLICY_TRACE_STATE_VIEW_SCHEMA_VERSION,
    require_random_access_state_view,
)
from gx1.features.htf_features import MULTI_TF_TIMEFRAMES


RANDOM_ACCESS_TRAIN_BATCH_SCHEMA_VERSION = (
    "gx1_unified_exit_random_access_train_batch_v1"
)
LIQUIDATION_RELATIVE_TRAIN_BATCH_SCHEMA_VERSION = "gx1_unified_exit_random_access_train_batch_v2"
FROZEN_POLICY_TRACE_TRAIN_BATCH_SCHEMA_VERSION = "gx1_unified_exit_random_access_train_batch_v3"
REFERENCE_POLICY_TRACE_TRAIN_BATCH_SCHEMA_VERSION = "gx1_unified_exit_random_access_train_batch_v4"
RANDOM_ACCESS_ENTRY_BRIDGE_BATCH_SCHEMA_VERSION = (
    "gx1_unified_exit_random_access_entry_bridge_batch_v1"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"UNIFIED_EXIT_RANDOM_ACCESS_{label}_SHA_INVALID")
    return value


def _normalization_surface(
    artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Require the normalization owner's immutable TRAIN-only artifact."""

    checked = require_lifetime_summary_normalization(artifact)
    if (
        not isinstance(checked, Mapping)
        or checked.get("decision") != "PASS"
        or not isinstance(checked.get("surface"), Mapping)
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TRAIN_NORMALIZATION_INVALID")
    return dict(checked["surface"]), _require_sha(
        checked.get("normalization_sha256"), "NORMALIZATION"
    )


def _collate_states(
    states: Sequence[Mapping[str, Any]],
    *,
    surface: Mapping[str, Any],
    device: torch.device,
    _market_state_batch_positions: Sequence[int] | None = None,
) -> dict[str, Any]:
    if not states:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_EMPTY_STATE_BATCH")
    path = right_pad_arrays(
        [
            np.ascontiguousarray(state["trade_path_tail_x"].transpose(1, 0, 2))
            for state in states
        ],
        max_rows_cap=TRADE_PATH_TAIL_MAX_ROWS,
    )
    summaries = np.stack(
        [np.asarray(state["lifetime_summary_x"], dtype=np.float32) for state in states]
    )
    if summaries.shape != (len(states), 2, LIFETIME_SUMMARY_DIM):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_SUMMARY_SHAPE_INVALID")
    normalized = apply_surface_normalization(summaries, surface)
    market_states = states if _market_state_batch_positions is None else [states[i] for i in _market_state_batch_positions]
    mtf_histories: dict[str, torch.Tensor] = {}
    mtf_gathers: dict[str, torch.Tensor] = {}
    mtf_lengths: dict[str, torch.Tensor] = {}
    for tf in MULTI_TF_TIMEFRAMES if market_states else ():
        suffix = tf.lower()
        histories = right_pad_arrays(
            [state["mtf"][f"exit_mtf_history_{suffix}"] for state in market_states]
        )
        mtf_histories[suffix] = torch.from_numpy(histories["values"]).to(device)
        mtf_lengths[suffix] = torch.from_numpy(histories["lengths"]).to(device)
        mtf_gathers[suffix] = torch.from_numpy(
            np.stack([state["mtf"][f"exit_mtf_gather_{suffix}"] for state in market_states])
        ).to(device)
    return {
        "m1_local_history_x": torch.from_numpy(
            np.stack([state["m1_local_history_x"] for state in market_states])
        ).to(device) if market_states else None,
        "state_ctx_cat": torch.from_numpy(
            np.stack([state["state_ctx_cat"] for state in market_states])
        ).to(device) if market_states else None,
        "state_ctx_cont": torch.from_numpy(
            np.stack([state["state_ctx_cont"] for state in market_states])
        ).to(device) if market_states else None,
        "trade_path_tail_x": torch.from_numpy(
            np.ascontiguousarray(path["values"].transpose(0, 2, 1, 3))
        ).to(device),
        "trade_path_lengths": torch.from_numpy(path["lengths"]).to(device),
        "normalized_lifetime_summary_x": torch.from_numpy(normalized).to(device),
        "exit_mtf_histories": mtf_histories if market_states else None,
        "exit_mtf_gathers": mtf_gathers if market_states else None,
        "exit_mtf_history_lengths": mtf_lengths if market_states else None,
    }


def collate_random_access_states_v1(
    states: Sequence[Mapping[str, Any]],
    *,
    normalization_artifact: Mapping[str, Any],
    device: torch.device,
    _market_state_batch_positions: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Public exact state-collation owner shared by TRAIN and VAL."""

    surface, _normalization_sha = _normalization_surface(normalization_artifact)
    return _collate_states(states, surface=surface, device=device,
                           _market_state_batch_positions=_market_state_batch_positions)


def collate_random_access_training_items(
    items: Sequence[Mapping[str, Any]],
    *,
    outer_batch_size: int,
    sampler_contract: Mapping[str, Any],
    normalization_artifact: Mapping[str, Any],
    expected_m1_source_sha256: str,
    expected_market_closure_authority_sha256: str,
    expected_economic_step_manifest_sha256: str,
    expected_economics_objective_contract_sha256: str,
    device: torch.device,
    reference_policy: Mapping[str, Any] | None = None,
    reference_cutoff_time_ns: int | None = None,
) -> dict[str, Any]:
    """Validate and flatten variable-K Entry items into two model calls."""

    contract = require_random_access_sampler_contract(sampler_contract)
    reference_contract = None if reference_policy is None else require_reference_policy_contract(reference_policy)
    if reference_cutoff_time_ns is not None and (
            type(reference_cutoff_time_ns) is not int or reference_cutoff_time_ns <= 0 or reference_contract is None):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_REFERENCE_CUTOFF_INVALID")
    entry_reference = reference_cutoff_time_ns is not None
    surface, normalization_sha = _normalization_surface(normalization_artifact)
    if (
        isinstance(outer_batch_size, bool)
        or not isinstance(outer_batch_size, int)
        or outer_batch_size < 1
        or not isinstance(items, Sequence)
        or isinstance(items, (str, bytes))
        or not items
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TRAIN_ITEMS_INVALID")
    counts: list[int] = []
    owner_by_item: list[int] = []
    transitions: list[tuple[dict[str, Any], dict[str, Any]]] = []
    anchors: list[tuple[dict[str, Any], dict[str, Any]]] = []
    episode_hashes: list[str] = []
    fill_hashes: list[str] = []
    witness_hashes: list[str] = []
    seen_owner: set[int] = set()
    seen_entry: set[int] = set()
    for item in items:
        required = {
            "outer_batch_index",
            "entry_row_index",
            "transitions",
            "anchor",
            "entry_episode_binding_sha256",
            "entry_fill_binding_sha256",
            "first_state_bridge_witness_sha256",
        }
        if not isinstance(item, Mapping) or set(item) != required:
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TRAIN_ITEM_SCHEMA_INVALID")
        owner = item["outer_batch_index"]
        entry_row = item["entry_row_index"]
        raw_transitions = item["transitions"]
        if (
            isinstance(owner, bool)
            or not isinstance(owner, int)
            or owner < 0
            or owner >= outer_batch_size
            or owner in seen_owner
            or isinstance(entry_row, bool)
            or not isinstance(entry_row, int)
            or not isinstance(raw_transitions, Sequence)
            or isinstance(raw_transitions, (str, bytes))
            or not raw_transitions
            or len(raw_transitions) != contract["transitions_per_entry"]
            or entry_row in seen_entry
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TRAIN_ITEM_INVALID")
        seen_owner.add(owner)
        seen_entry.add(entry_row)
        owner_by_item.append(owner)
        counts.append(len(raw_transitions))
        transition_samples: list[dict[str, Any]] = []
        for pair in raw_transitions:
            if not isinstance(pair, Mapping) or set(pair) != {"sample", "state_view"}:
                raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TRANSITION_PAIR_INVALID")
            sample = require_random_access_sample(
                pair["sample"], sampler_contract=contract
            )
            view = require_random_access_state_view(
                pair["state_view"],
                sampler_contract=contract,
                sample=sample,
                expected_m1_source_sha256=expected_m1_source_sha256,
                expected_market_closure_authority_sha256=(
                    expected_market_closure_authority_sha256
                ),
                expected_economic_step_manifest_sha256=(
                    expected_economic_step_manifest_sha256
                ),
                expected_economics_objective_contract_sha256=(
                    expected_economics_objective_contract_sha256
                ),
            )
            if (
                sample["entry_row_index"] != entry_row
                or view["entry_row_index"] != entry_row
            ):
                raise RuntimeError(
                    "UNIFIED_EXIT_RANDOM_ACCESS_TRAIN_ENTRY_BINDING_INVALID"
                )
            expected_current = sample["state_index"]
            if (
                view["sample_role"] != "bellman_transition"
                or view["loss_weight"] != sample["importance_weight"]
                or view["current"]["state_index"] != expected_current
                or view["successor"]["state_index"] != sample["successor_state_index"]
                or not np.isfinite(view["elapsed_wall_clock_gamma"])
                or not 0.0 < view["elapsed_wall_clock_gamma"] <= 1.0
            ):
                raise RuntimeError(
                    "UNIFIED_EXIT_RANDOM_ACCESS_TRANSITION_LINKAGE_INVALID"
                )
            policy = np.asarray(view["policy_action_valid_mask"], dtype=np.bool_)
            bellman = np.asarray(view["bellman_target_valid_mask"], dtype=np.bool_)
            observed = np.asarray(view["successor_observed_mask"], dtype=np.bool_)
            successor_terminal = np.asarray(
                view["successor_terminal_mask"], dtype=np.bool_
            )
            successor_policy = np.asarray(
                view["successor_policy_action_valid_mask"], dtype=np.bool_
            )
            current_terminal = np.asarray(view["terminal_mask"], dtype=np.bool_)
            current_censored = np.asarray(view["right_censored_mask"], dtype=np.bool_)
            expected_successor_policy = np.ones((2, 2), dtype=np.bool_)
            expected_successor_policy[:, 0] &= ~successor_terminal
            expected_bellman = policy.copy()
            expected_bellman[:, 0] &= observed
            if (
                current_terminal.any()
                or current_censored.any()
                or not observed.all()
                or not policy[:, 1].all()
                or not np.array_equal(successor_policy, expected_successor_policy)
                or not np.array_equal(bellman, expected_bellman)
                or not np.isfinite(view["immediate_reward_bps"]).all()
            ):
                raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TRANSITION_MASK_INVALID")
            transition_samples.append(sample)
            transitions.append((sample, view))
        anchor_pair = item["anchor"]
        if not isinstance(anchor_pair, Mapping) or set(anchor_pair) != {
            "sample",
            "state_view",
        }:
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_ANCHOR_PAIR_INVALID")
        anchor_sample = require_random_access_entry_anchor(
            anchor_pair["sample"], sampler_contract=contract
        )
        anchor_view = require_random_access_state_view(
            anchor_pair["state_view"],
            sampler_contract=contract,
            sample=anchor_sample,
            expected_m1_source_sha256=expected_m1_source_sha256,
            expected_market_closure_authority_sha256=(
                expected_market_closure_authority_sha256
            ),
            expected_economic_step_manifest_sha256=(
                expected_economic_step_manifest_sha256
            ),
            expected_economics_objective_contract_sha256=(
                expected_economics_objective_contract_sha256
            ),
        )
        if (
            anchor_sample["entry_row_index"] != entry_row
            or anchor_view["entry_row_index"] != entry_row
            or anchor_view["current"]["state_index"] != 0
            or anchor_view["sample_role"] != "entry_anchor_no_loss"
            or anchor_view["loss_weight"] != 0.0
            or anchor_sample["epoch_index"] != transition_samples[0]["epoch_index"]
            or anchor_sample["entry_slot"] != transition_samples[0]["entry_slot"]
            or sorted(sample["sample_slot"] for sample in transition_samples)
            != list(range(contract["transitions_per_entry"]))
            or any(
                sample["epoch_index"] != anchor_sample["epoch_index"]
                or sample["entry_slot"] != anchor_sample["entry_slot"]
                for sample in transition_samples
            )
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_ANCHOR_BINDING_INVALID")
        anchors.append((anchor_sample, anchor_view))
        episode_hashes.append(
            _require_sha(item["entry_episode_binding_sha256"], "EPISODE_BINDING")
        )
        fill_hashes.append(
            _require_sha(item["entry_fill_binding_sha256"], "FILL_BINDING")
        )
        witness_hashes.append(
            _require_sha(item["first_state_bridge_witness_sha256"], "BRIDGE_WITNESS")
        )
    layout = flatten_sample_counts(counts)
    expected_owner = np.asarray(owner_by_item, dtype=np.int64)[
        layout["entry_batch_index"]
    ]
    online_states = [view["current"] for _sample, view in transitions]
    target_states = [view["successor"] for _sample, view in transitions] + [
        view["current"] for _sample, view in anchors
    ]
    transition_views = [view for _sample, view in transitions]
    anchor_views = [view for _sample, view in anchors]
    relative_flags = {view["schema_version"] in {LIQUIDATION_RELATIVE_STATE_VIEW_SCHEMA_VERSION, FROZEN_POLICY_TRACE_STATE_VIEW_SCHEMA_VERSION, REFERENCE_POLICY_TRACE_STATE_VIEW_SCHEMA_VERSION}
                      for view in [*transition_views, *anchor_views]}
    if len(relative_flags) != 1:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_MIXED_VALUE_COORDINATES")
    relative = relative_flags.pop()
    trace_flags = {"frozen_policy_trace" in view for view in transition_views}
    if len(trace_flags) != 1 or any("frozen_policy_trace" in view for view in anchor_views):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_MIXED_BACKUP_POLICY")
    reference_flags = {"reference_policy_trace" in view for view in transition_views}
    if (reference_flags != {reference_contract is not None}
            or {"reference_policy_trace" in view for view in anchor_views} != {entry_reference}
            or (reference_contract is not None and trace_flags != {False})):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_REFERENCE_POLICY_BINDING_INVALID")
    if reference_contract is not None:
        traces = [view["reference_policy_trace"] for view in transition_views]
        anchor_traces = [view["reference_policy_trace"] for view in anchor_views] if entry_reference else []
        if any(trace["policy"] != reference_contract
               or trace.get("supervision_end_time_ns") != reference_cutoff_time_ns
               for trace in [*traces, *anchor_traces]):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_REFERENCE_POLICY_BINDING_INVALID")
        target_states = [trace["boundary"] for trace in traces] + (
            [trace["boundary"] for trace in anchor_traces] if entry_reference else [view["current"] for view in anchor_views])
    extra_states, extra_masks, extra_owners = [], [], []
    if trace_flags == {True}:
        # Keep the legacy successor/anchor layout intact; append only additional
        # frozen-target states. No extra online rows or sampler weights.
        indices = np.zeros((len(transitions), 5), dtype=np.int64)
        rewards = np.zeros((len(transitions), 5, 2), dtype=np.float32)
        gammas = np.ones((len(transitions), 5), dtype=np.float32)
        available = np.zeros((len(transitions), 5), dtype=np.bool_)
        for row, view in enumerate(transition_views):
            indices[row, :] = row
            for offset, step in enumerate([view, *view["frozen_policy_trace"]["steps"]]):
                if offset:
                    indices[row, offset:] = len(target_states) + len(extra_states)
                    extra_states.append(step["successor"])
                    extra_masks.append(step["successor_policy_action_valid_mask"])
                    extra_owners.append(int(expected_owner[row]))
                rewards[row, offset] = step["liquidation_relative_reward_bps"][:, 0]
                gammas[row, offset] = step["elapsed_wall_clock_gamma"]
                available[row, offset] = True
    result = {
        "schema_version": (LIQUIDATION_RELATIVE_TRAIN_BATCH_SCHEMA_VERSION if relative
                           else RANDOM_ACCESS_TRAIN_BATCH_SCHEMA_VERSION),
        "normalization_sha256": normalization_sha,
        "outer_batch_size": outer_batch_size,
        "selected_entry_count": len(items),
        "transition_count": len(transitions),
        "online_entry_batch_index": torch.from_numpy(expected_owner).to(device),
        "target_entry_batch_index": torch.from_numpy(
            np.concatenate((expected_owner, np.asarray(owner_by_item, dtype=np.int64)))
        ).to(device),
        "selected_entry_batch_index": torch.tensor(
            owner_by_item, dtype=torch.long, device=device
        ),
        "online_model_inputs": _collate_states(
            online_states, surface=surface, device=device
        ),
        "target_model_inputs": _collate_states(
            [*target_states, *extra_states], surface=surface, device=device
        ),
        "online_action_valid_mask": torch.from_numpy(
            np.stack([view["policy_action_valid_mask"] for view in transition_views])
        ).to(device),
        "bellman_target_valid_mask": torch.from_numpy(
            np.stack([view["bellman_target_valid_mask"] for view in transition_views])
        ).to(device),
        "successor_action_valid_mask": torch.from_numpy(
            np.stack(
                [
                    view["successor_policy_action_valid_mask"]
                    for view in transition_views
                ]
            )
        ).to(device),
        "successor_observed_mask": torch.from_numpy(
            np.stack([view["successor_observed_mask"] for view in transition_views])
        ).to(device),
        "terminal_mask": torch.from_numpy(
            np.stack([view["terminal_mask"] for view in transition_views])
        ).to(device),
        "right_censored_mask": torch.from_numpy(
            np.stack([view["right_censored_mask"] for view in transition_views])
        ).to(device),
        "immediate_reward_bps": torch.from_numpy(
            np.stack([view["immediate_reward_bps"] for view in transition_views])
        ).to(device),
        "elapsed_wall_clock_gamma": torch.tensor(
            [view["elapsed_wall_clock_gamma"] for view in transition_views],
            dtype=torch.float32,
            device=device,
        ),
        "importance_weight": torch.tensor(
            [
                sample["importance_weight"] * view["loss_weight"]
                for sample, view in transitions
            ],
            dtype=torch.float32,
            device=device,
        ),
        "anchor_action_valid_mask": torch.from_numpy(
            np.stack([view["policy_action_valid_mask"] for view in anchor_views])
        ).to(device),
        "anchor_state_view_sha256": [
            view["state_view_sha256"] for view in anchor_views
        ],
        "entry_episode_binding_sha256": episode_hashes,
        "entry_fill_binding_sha256": fill_hashes,
        "first_state_bridge_witness_sha256": witness_hashes,
    }
    if relative:
        result["liquidation_relative_values"] = True
        result["liquidation_relative_reward_bps"] = torch.from_numpy(
            np.stack([view["liquidation_relative_reward_bps"] for view in transition_views])
        ).to(device)
        result["entry_liquidation_value_bps"] = torch.from_numpy(
            np.stack([view["current_liquidation_value_bps"] for view in anchor_views])
        ).to(device)
    if trace_flags == {True}:
        result["schema_version"] = FROZEN_POLICY_TRACE_TRAIN_BATCH_SCHEMA_VERSION
        if extra_states:
            result["target_entry_batch_index"] = torch.cat((result["target_entry_batch_index"],
                torch.tensor(extra_owners, dtype=torch.long, device=device)))
        result["target_action_valid_mask"] = torch.cat((result["successor_action_valid_mask"],
            result["anchor_action_valid_mask"],
            torch.from_numpy(np.stack(extra_masks) if extra_masks else np.empty((0, 2, 2), dtype=np.bool_)).to(device)))
        result["frozen_policy_trace"] = {
            "successor_q_indices": torch.from_numpy(indices).to(device),
            "hold_reward_bps": torch.from_numpy(rewards).to(device),
            "elapsed_wall_clock_gamma": torch.from_numpy(gammas).to(device),
            "transition_available_mask": torch.from_numpy(available).to(device),
        }
    if reference_contract is not None:
        result["schema_version"] = REFERENCE_POLICY_TRACE_TRAIN_BATCH_SCHEMA_VERSION
        groups = [("reference_policy_trace", transition_views, traces)]
        if entry_reference:
            groups.append(("entry_reference_policy_trace", anchor_views, anchor_traces))
        for name, group_views, traces in groups:
            width = max(len(trace["steps"]) for trace in traces)
            shape = (len(traces), width)
            rewards = np.zeros((*shape, 2), dtype=np.float32)
            gammas = np.ones(shape, dtype=np.float32)
            available = np.zeros(shape, dtype=np.bool_)
            terminals = np.zeros((*shape, 2), dtype=np.bool_)
            censored = np.zeros((len(traces), 2), dtype=np.bool_)
            masks = []
            for row, trace in enumerate(traces):
                steps = trace["steps"]
                for offset, step in enumerate(steps):
                    rewards[row, offset] = step["liquidation_relative_reward_bps"][:, 0]
                    gammas[row, offset] = step["elapsed_wall_clock_gamma"]
                    terminals[row, offset] = step["successor_terminal_mask"]
                    available[row, offset] = True
                masks.append(steps[-1]["successor_policy_action_valid_mask"])
                censored[row] = [trace["boundary"]["state_index"] == count - 1 and not terminal
                                 for count, terminal in zip(trace["side_lifecycle_state_counts"], trace["side_economic_terminal"])]
            result[name] = {
                "policy": reference_contract,
                "state_view_sha256": [view["state_view_sha256"] for view in group_views],
                "hold_reward_bps": torch.from_numpy(rewards).to(device),
                "elapsed_wall_clock_gamma": torch.from_numpy(gammas).to(device),
                "transition_available_mask": torch.from_numpy(available).to(device),
                "successor_terminal_mask": torch.from_numpy(terminals).to(device),
                "boundary_action_valid_mask": torch.from_numpy(np.stack(masks)).to(device),
                "boundary_right_censored_mask": torch.from_numpy(censored).to(device),
            }
        anchor_target_mask = (result["entry_reference_policy_trace"]["boundary_action_valid_mask"]
                              if entry_reference else result["anchor_action_valid_mask"])
        result["target_action_valid_mask"] = torch.cat((
            result["reference_policy_trace"]["boundary_action_valid_mask"], anchor_target_mask))
        if entry_reference:
            result["reference_cutoff_time_ns"] = reference_cutoff_time_ns
            result["reference_boundary_time_ns"] = torch.tensor(
                [view["reference_policy_trace"]["boundary"]["decision_time_ns"]
                 for view in [*transition_views, *anchor_views]], dtype=torch.int64, device=device)
    return result


def frozen_policy_trace_hold_targets(
    *, all_target_q: torch.Tensor, target_action_valid_mask: torch.Tensor,
    trace: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Evaluate observed rewards under the frozen causal policy, not hindsight max.

    An intermediate unique HOLD includes the next observed transition. EXIT or
    a tie retains the teacher value. The last observed/computation boundary
    bootstraps from the teacher; it never becomes an economic terminal.
    """
    if set(trace) != {"successor_q_indices", "hold_reward_bps", "elapsed_wall_clock_gamma", "transition_available_mask"}:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TRACE_BATCH_INVALID")
    indices, rewards, gammas, available = (trace["successor_q_indices"], trace["hold_reward_bps"],
        trace["elapsed_wall_clock_gamma"], trace["transition_available_mask"])
    rows = indices.shape[0] if indices.ndim == 2 else 0
    if (indices.dtype != torch.long or indices.shape != (rows, 5) or rows == 0
            or rewards.shape != (rows, 5, 2) or gammas.shape != (rows, 5)
            or available.shape != (rows, 5) or available.dtype != torch.bool
            or all_target_q.ndim != 3 or all_target_q.shape[1:] != (2, 2)
            or target_action_valid_mask.shape != all_target_q.shape
            or target_action_valid_mask.dtype != torch.bool
            or not bool(available[:, 0].all()) or bool((available[:, 1:] & ~available[:, :-1]).any())
            or bool((indices < 0).any()) or bool((indices >= len(all_target_q)).any())
            or not bool(torch.isfinite(rewards).all()) or not bool(torch.isfinite(gammas).all())
            or bool(((gammas <= 0) | (gammas > 1)).any())):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TRACE_BATCH_INVALID")
    q = all_target_q.detach()[indices]
    mask = target_action_valid_mask[indices]
    if (not bool(mask[..., 1].all()) or not bool(torch.isfinite(q[mask]).all())
            or bool((q[..., 1] != 0).any())):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TRACE_Q_INVALID")
    teacher_values = q.masked_fill(~mask, -torch.inf).max(dim=-1).values
    values = teacher_values[:, -1]
    for offset in range(3, -1, -1):
        follow_hold = (available[:, offset + 1, None] & mask[:, offset, :, 0]
                       & (q[:, offset, :, 0] > q[:, offset, :, 1]))
        continuation = rewards[:, offset + 1] + gammas[:, offset + 1, None] * values
        values = torch.where(follow_hold, continuation, teacher_values[:, offset])
    targets = rewards[:, 0] + gammas[:, 0, None] * values
    if not bool(torch.isfinite(targets).all()):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TRACE_TARGET_INVALID")
    return targets.detach()

def _frozen_exit_q_in_base_batches(
    *, target_model: nn.Module, model_inputs: Mapping[str, Any], rows_per_batch: int,
) -> torch.Tensor:
    """Keep the frozen teacher's working batch at the original one-step size."""
    row_count = int(model_inputs["entry_decision_representation"].shape[0])
    if type(rows_per_batch) is not int or rows_per_batch < 1 or row_count < 1:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TARGET_BATCH_SIZE_INVALID")

    def select(value: Any, start: int, stop: int) -> Any:
        if isinstance(value, Mapping):
            return {name: select(item, start, stop) for name, item in value.items()}
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == row_count:
            return value[start:stop]
        if type(value) is bool:
            return value
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TARGET_BATCH_LAYOUT_INVALID")

    chunks = []
    for start in range(0, row_count, rows_per_batch):
        inputs = select(model_inputs, start, start + rows_per_batch)
        chunks.append(target_model.forward_exit_random_access_batch(**inputs)["exit_action_q_bps"])
    return torch.cat(chunks, dim=0)


def run_random_access_training_step(
    *,
    model: nn.Module,
    target_model: nn.Module,
    entry_decision_representations: torch.Tensor,
    target_entry_decision_representations: torch.Tensor,
    batch: Mapping[str, Any],
    grad_accum_steps: int,
    reference_policy: Mapping[str, Any] | None = None,
    reference_cutoff_time_ns: int | None = None,
) -> dict[str, Any]:
    """One online forward, one frozen-target evaluation and one backward."""

    reference_mode = batch.get("schema_version") == REFERENCE_POLICY_TRACE_TRAIN_BATCH_SCHEMA_VERSION
    policy = None if reference_policy is None else require_reference_policy_contract(reference_policy)
    if (reference_mode != (policy is not None)
            or reference_mode != ("reference_policy_trace" in batch)):
        raise RuntimeError("UNIFIED_EXIT_REFERENCE_TRAINING_NOT_BOUND")
    if reference_mode and batch["reference_policy_trace"].get("policy") != policy:
        raise RuntimeError("UNIFIED_EXIT_REFERENCE_TRAINING_NOT_BOUND")
    entry_reference = reference_cutoff_time_ns is not None
    entry_reference_fields = {"entry_reference_policy_trace", "reference_cutoff_time_ns", "reference_boundary_time_ns"}
    if ((entry_reference and (not reference_mode or type(reference_cutoff_time_ns) is not int
                             or reference_cutoff_time_ns <= 0 or entry_reference_fields.intersection(batch) != entry_reference_fields
                             or batch["reference_cutoff_time_ns"] != reference_cutoff_time_ns))
            or (not entry_reference and entry_reference_fields.intersection(batch))):
        raise RuntimeError("UNIFIED_EXIT_REFERENCE_ENTRY_NOT_BOUND")
    if entry_reference:
        boundary_times = batch["reference_boundary_time_ns"]
        if (batch["entry_reference_policy_trace"].get("policy") != policy
                or not isinstance(boundary_times, torch.Tensor)
                or boundary_times.dtype != torch.int64
                or tuple(boundary_times.shape) != (batch["transition_count"] + batch["selected_entry_count"],)
                or bool(((boundary_times <= 0) | (boundary_times > reference_cutoff_time_ns)).any())):
            raise RuntimeError("UNIFIED_EXIT_REFERENCE_ENTRY_CUTOFF_INVALID")
    trace_mode = batch.get("schema_version") == FROZEN_POLICY_TRACE_TRAIN_BATCH_SCHEMA_VERSION
    relative = reference_mode or trace_mode or batch.get("schema_version") == LIQUIDATION_RELATIVE_TRAIN_BATCH_SCHEMA_VERSION
    if batch.get("schema_version") not in {RANDOM_ACCESS_TRAIN_BATCH_SCHEMA_VERSION, LIQUIDATION_RELATIVE_TRAIN_BATCH_SCHEMA_VERSION, FROZEN_POLICY_TRACE_TRAIN_BATCH_SCHEMA_VERSION, REFERENCE_POLICY_TRACE_TRAIN_BATCH_SCHEMA_VERSION}:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TRAIN_BATCH_INVALID")
    if (trace_mode != ("frozen_policy_trace" in batch)
            or (trace_mode or reference_mode) != ("target_action_valid_mask" in batch)):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_BACKUP_POLICY_INVALID")
    coordinate_fields = {"liquidation_relative_values", "liquidation_relative_reward_bps", "entry_liquidation_value_bps"}
    present = coordinate_fields.intersection(batch)
    if ((relative and (present != coordinate_fields or batch["liquidation_relative_values"] is not True))
            or (not relative and present)):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_VALUE_COORDINATES_INVALID")
    if target_model.training or any(
        parameter.requires_grad for parameter in target_model.parameters()
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TARGET_MODEL_NOT_FROZEN_EVAL")
    outer = batch["outer_batch_size"]
    if (
        entry_decision_representations.ndim != 2
        or tuple(target_entry_decision_representations.shape)
        != tuple(entry_decision_representations.shape)
        or int(entry_decision_representations.shape[0]) != outer
    ):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_ENTRY_TOKEN_SHAPE_INVALID")
    online_owner = batch["online_entry_batch_index"]
    target_owner = batch["target_entry_batch_index"]
    token = (
        entry_decision_representations.index_select(0, online_owner)
        .detach()
        .clone()
        .requires_grad_(True)
    )
    online_inputs = dict(batch["online_model_inputs"])
    online_inputs["entry_decision_representation"] = token
    online_inputs["action_valid_mask"] = batch["online_action_valid_mask"]
    target_inputs = dict(batch["target_model_inputs"])
    target_inputs["entry_decision_representation"] = (
        target_entry_decision_representations.index_select(0, target_owner)
    )
    target_inputs["action_valid_mask"] = batch["target_action_valid_mask"] if trace_mode or reference_mode else torch.cat(
        (batch["successor_action_valid_mask"], batch["anchor_action_valid_mask"]),
        dim=0,
    )
    relative = batch.get("liquidation_relative_values", False)
    if type(relative) is not bool:
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_VALUE_COORDINATES_INVALID")
    if relative:
        online_inputs["liquidation_relative_values"] = True
        target_inputs["liquidation_relative_values"] = True
    rewards = batch["liquidation_relative_reward_bps"] if relative else batch["immediate_reward_bps"]
    transition_count = batch["transition_count"]
    target_cache: dict[str, Any] = {}
    online_cache: dict[str, Any] = {}

    def target_call(**kwargs: Any) -> torch.Tensor:
        if trace_mode:
            q = _frozen_exit_q_in_base_batches(
                target_model=target_model, model_inputs=kwargs,
                rows_per_batch=transition_count + batch["selected_entry_count"],
            )
        else:
            q = target_model.forward_exit_random_access_batch(**kwargs)["exit_action_q_bps"]
        target_cache["all_q"] = q
        return q

    def online_call(**kwargs: Any) -> torch.Tensor:
        output = model.forward_exit_random_access_batch(**kwargs)
        online_cache.update(
            {
                name: value.detach() if isinstance(value, torch.Tensor) else value
                for name, value in output.items()
            }
        )
        return output["exit_action_q_bps"]

    def target_builder(all_target_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        successor_q = all_target_q[:transition_count]
        action = batch["online_action_valid_mask"]
        state_valid = action[..., 1]
        terminal = batch["terminal_mask"]
        if bool(batch["right_censored_mask"].any().item()):
            raise RuntimeError(
                "UNIFIED_EXIT_RANDOM_ACCESS_RIGHT_CENSORED_TRAIN_FORBIDDEN"
            )
        if reference_mode:
            trace = batch["reference_policy_trace"]
            anchor_target_mask = (batch["entry_reference_policy_trace"]["boundary_action_valid_mask"]
                                  if entry_reference else batch["anchor_action_valid_mask"])
            if (not torch.equal(trace["hold_reward_bps"][:, 0], rewards[..., 0])
                    or not torch.equal(trace["elapsed_wall_clock_gamma"][:, 0], batch["elapsed_wall_clock_gamma"])
                    or not torch.equal(batch["target_action_valid_mask"][:transition_count], trace["boundary_action_valid_mask"])
                    or not torch.equal(batch["target_action_valid_mask"][transition_count:], anchor_target_mask)
                    or not bool(action.all()) or bool(terminal.any())
                    or not bool(batch["successor_observed_mask"].all())
                    or not bool(batch["bellman_target_valid_mask"].all())):
                raise RuntimeError("UNIFIED_EXIT_REFERENCE_BATCH_BINDING_INVALID")
            reference = build_reference_policy_hold_targets(
                policy=policy, hold_reward_bps=trace["hold_reward_bps"],
                elapsed_wall_clock_gamma=trace["elapsed_wall_clock_gamma"],
                transition_available_mask=trace["transition_available_mask"],
                successor_terminal_mask=trace["successor_terminal_mask"],
                boundary_action_q_bps=successor_q,
                boundary_action_valid_mask=trace["boundary_action_valid_mask"],
                boundary_right_censored_mask=trace["boundary_right_censored_mask"],
            )
            target_cache["reference_targets"] = reference
            if entry_reference:
                anchor_trace = batch["entry_reference_policy_trace"]
                target_cache["entry_reference_targets"] = build_reference_policy_hold_targets(
                    policy=policy, boundary_action_q_bps=all_target_q[transition_count:],
                    **{name: anchor_trace[name] for name in (
                        "hold_reward_bps", "elapsed_wall_clock_gamma", "transition_available_mask",
                        "successor_terminal_mask", "boundary_action_valid_mask", "boundary_right_censored_mask")},
                )
            return torch.stack((reference["hold_target_bps"], reference["exit_now_target_bps"]), dim=-1), batch["bellman_target_valid_mask"]
        targets, valid = build_unified_exit_fitted_q_targets(
            frozen_target_q_bps=torch.zeros_like(successor_q).unsqueeze(2),
            exit_now_reward_bps=rewards[..., 1].unsqueeze(2),
            action_valid_mask=action.unsqueeze(2),
            state_valid_mask=state_valid.unsqueeze(2),
            terminal_mask=terminal.unsqueeze(2),
            terminal_reason_index=torch.zeros_like(
                terminal, dtype=torch.long
            ).unsqueeze(2),
            chunk_successor_target_q_bps=successor_q,
            chunk_successor_action_valid_mask=batch["successor_action_valid_mask"],
            bellman_target_valid_mask=batch["bellman_target_valid_mask"].unsqueeze(2),
            successor_observed_mask=batch["successor_observed_mask"].unsqueeze(2),
            hold_immediate_reward_bps=rewards[..., 0].unsqueeze(
                2
            ),
            transition_discount=batch["elapsed_wall_clock_gamma"][:, None, None].expand(
                -1, 2, 1
            ),
        )
        if trace_mode:
            trace = batch["frozen_policy_trace"]
            if (not torch.equal(trace["successor_q_indices"][:, 0], torch.arange(transition_count, device=successor_q.device))
                    or not torch.equal(trace["hold_reward_bps"][:, 0], rewards[..., 0])
                    or not torch.equal(trace["elapsed_wall_clock_gamma"][:, 0], batch["elapsed_wall_clock_gamma"])
                    or not torch.equal(batch["target_action_valid_mask"][:transition_count], batch["successor_action_valid_mask"])
                    or not torch.equal(batch["target_action_valid_mask"][transition_count:transition_count + batch["selected_entry_count"]], batch["anchor_action_valid_mask"])):
                raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TRACE_FIRST_STEP_MISMATCH")
            targets[..., 0, 0] = frozen_policy_trace_hold_targets(
                all_target_q=all_target_q, target_action_valid_mask=batch["target_action_valid_mask"],
                trace=batch["frozen_policy_trace"])
        return targets.squeeze(2), valid.squeeze(2)

    loss_scale = torch.exp(-model.task_log_variances["unified_exit_action"])
    outcome = one_forward_one_backward(
        online_model=online_call,
        target_model=target_call,
        online_inputs=online_inputs,
        target_inputs=target_inputs,
        target_builder=target_builder,
        importance_weight=batch["importance_weight"],
        grad_accum_steps=grad_accum_steps,
        loss_scale=loss_scale,
    )
    if token.grad is None or not bool(torch.isfinite(token.grad).all().item()):
        raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_ENTRY_GRADIENT_INVALID")
    entry_gradients = torch.zeros_like(entry_decision_representations)
    entry_gradients.index_add_(0, online_owner, token.grad.detach())
    all_target_q = target_cache["all_q"]
    anchor_q = all_target_q[transition_count:transition_count + batch["selected_entry_count"]]
    if entry_reference:
        anchor_reference = target_cache["entry_reference_targets"]
        anchor_q = torch.stack((anchor_reference["hold_target_bps"], anchor_reference["exit_now_target_bps"]), dim=-1)
    anchor_mask = batch["anchor_action_valid_mask"]
    if entry_reference:
        first_values = reference_policy_state_values(
            policy=policy, action_q_bps=anchor_q, action_valid_mask=anchor_mask)
    else:
        first_values = unified_exit_first_state_side_values(
            frozen_target_q_bps=anchor_q.unsqueeze(2),
            action_valid_mask=anchor_mask.unsqueeze(2),
            state_valid_mask=torch.ones(
                anchor_q.shape[:2] + (1,), dtype=torch.bool, device=anchor_q.device
            ),
        )
    if relative:
        liquidation = batch["entry_liquidation_value_bps"]
        if liquidation.shape != first_values.shape or not bool(torch.isfinite(liquidation).all().item()):
            raise RuntimeError("UNIFIED_EXIT_ENTRY_LIQUIDATION_ANCHOR_INVALID")
        first_values = first_values + liquidation
    selected = batch["selected_entry_batch_index"]
    all_first_values = entry_decision_representations.new_zeros((outer, 2))
    all_side_valid = torch.zeros(
        (outer, 2), dtype=torch.bool, device=all_first_values.device
    )
    all_first_values[selected] = first_values.to(all_first_values.dtype)
    all_side_valid[selected] = True
    episode_hashes: list[str | None] = [None] * outer
    fill_hashes: list[str | None] = [None] * outer
    for position, episode_sha, fill_sha in zip(
        selected.detach().cpu().tolist(),
        batch["entry_episode_binding_sha256"],
        batch["entry_fill_binding_sha256"],
    ):
        episode_hashes[position] = episode_sha
        fill_hashes[position] = fill_sha
    entry_targets, entry_valid, fitted_entry_binding = build_entry_fitted_q_targets(
        frozen_exit_first_state_values_bps=all_first_values,
        exit_side_valid_mask=all_side_valid,
        episode_pack_sha256=episode_hashes,
        fill_binding_sha256=fill_hashes,
    )
    bridge = {
        "schema_version": RANDOM_ACCESS_ENTRY_BRIDGE_BATCH_SCHEMA_VERSION,
        "normalization_sha256": batch["normalization_sha256"],
        "selected_outer_batch_indices": selected.detach().cpu().tolist(),
        "anchor_state_view_sha256": list(batch["anchor_state_view_sha256"]),
        "entry_episode_binding_sha256": list(batch["entry_episode_binding_sha256"]),
        "entry_fill_binding_sha256": list(batch["entry_fill_binding_sha256"]),
        "first_state_bridge_witness_sha256": list(
            batch["first_state_bridge_witness_sha256"]
        ),
        "target_model_values_are_stop_gradient": True,
        "flat_target_bps": 0.0,
        "entry_fitted_q_binding_sha256": fitted_entry_binding["binding_sha256"],
    }
    if entry_reference:
        bridge["entry_bridge_semantics"] = "observed_reference_anchor_V_mu_without_hindsight_action"
        bridge["reference_cutoff_time_ns"] = reference_cutoff_time_ns
    bridge["binding_sha256"] = _canonical_sha256(bridge)
    return {
        **outcome,
        "online_output": online_cache,
        "entry_gradients": entry_gradients,
        "entry_targets": entry_targets.detach(),
        "entry_valid_mask": entry_valid,
        "entry_bridge_binding": bridge,
        **({"reference_target_evidence": target_cache["reference_targets"],
            "entry_bridge_semantics": (bridge["entry_bridge_semantics"] if entry_reference
                                       else "unchanged_initial_teacher_greedy_bridge_no_reference_refresh")}
           if reference_mode else {}),
        **({"entry_reference_target_evidence": target_cache["entry_reference_targets"]} if entry_reference else {}),
    }


__all__ = (
    "RANDOM_ACCESS_ENTRY_BRIDGE_BATCH_SCHEMA_VERSION",
    "RANDOM_ACCESS_TRAIN_BATCH_SCHEMA_VERSION",
    "LIQUIDATION_RELATIVE_TRAIN_BATCH_SCHEMA_VERSION",
    "collate_random_access_states_v1",
    "collate_random_access_training_items",
    "run_random_access_training_step",
)
