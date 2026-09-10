"""Variable-prefix dataset pack for chunked, unbounded Exit training."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from gx1.contracts.entry_exit_feature_base_v1 import (
    EXIT_FEATURE_SEQUENCE_BARS,
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.unified_exit_lifecycle_v2 import (
    UNIFIED_EXIT_CHUNK_ROWS,
    UNIFIED_EXIT_LIFECYCLE_V2_SCHEMA_VERSION,
)
from gx1.contracts.unified_exit_fitted_q_v1 import (
    require_unified_exit_unbounded_training_readiness,
)
from gx1.features.htf_features import MULTI_TF_FEATURE_COUNT_V4
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_PATH_FEATURE_DIM,
)


UNIFIED_EXIT_EPISODE_PACK_V2_SCHEMA_VERSION = (
    "gx1_unified_exit_causal_chunk_pack_v2"
)


def unified_exit_episode_pack_v2_contract() -> dict[str, Any]:
    payload = {
        "schema_version": UNIFIED_EXIT_EPISODE_PACK_V2_SCHEMA_VERSION,
        "lifecycle_schema_version": UNIFIED_EXIT_LIFECYCLE_V2_SCHEMA_VERSION,
        "chunk_state_capacity": UNIFIED_EXIT_CHUNK_ROWS,
        "first_chunk_pre_entry_rows": EXIT_FEATURE_SEQUENCE_BARS - 1,
        "prefix_semantics": (
            "all_post_fill_states_from_entry_through_chunk_successor"
        ),
        "later_chunk_semantics": "full_causal_prefix_equivalent_to_exact_carry",
        "training_unit": "one_entry_side_chunk",
        "model_side_pair_adapter": (
            "selected_side_path_is_inserted_into_its_independent_model_side_branch"
        ),
        "successor_is_training_state": False,
        "successor_role": "frozen_target_network_boundary_value_only",
        "right_censored_hold_target_valid": False,
        "mask_semantics": {
            "exit_policy_action_valid_mask": "runtime executable actions",
            "exit_bellman_target_valid_mask": "observed supervised targets",
            "exit_successor_observed_mask": "observed next state per current state",
        },
        "capacity_forces_exit": False,
        "test_access": False,
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()
    return payload


def _update_array(digest: Any, name: str, value: Any) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())


def seal_unified_exit_episode_pack_v2(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    observed = dict(value)
    digest = hashlib.sha256()
    for name in sorted(set(observed) - {"episode_pack_sha256"}):
        item = observed[name]
        if isinstance(item, np.ndarray):
            _update_array(digest, name, item)
        else:
            digest.update(name.encode("ascii"))
            digest.update(b"\0")
            digest.update(
                json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
    observed["episode_pack_sha256"] = digest.hexdigest()
    return observed


def require_unified_exit_episode_pack_v2(
    value: Mapping[str, Any],
    *,
    per_tf_seq_lens: Mapping[str, int],
    expected_mtf_cache_identity_sha256: str,
    expected_split: str | None = None,
    expected_lifecycle_manifest_sha256: str | None = None,
    expected_chunk_pointer_stream_sha256: str | None = None,
    expected_scheduled_pair_chunk_pointer_sha256: str | None = None,
    context: str,
) -> dict[str, Any]:
    """Validate one full-prefix chunk and its optional causal successor."""

    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context}_UNIFIED_EXIT_CHUNK_PACK_INVALID")
    observed = dict(value)
    tf_names = tuple(tf.lower() for tf in EXIT_MTF_CONTEXT_TIMEFRAMES)
    array_names = {
        "exit_local_history_x",
        "exit_local_history_time_ns",
        "exit_state_ctx_cont",
        "exit_state_ctx_cat",
        "exit_state_row_time_ns",
        "exit_decision_time_ns",
        "exit_path_x",
        "exit_entry_bid_ask",
        "exit_now_reward_bps",
        "hold_immediate_reward_bps",
        "exit_policy_action_valid_mask",
        "exit_bellman_target_valid_mask",
        "exit_successor_observed_mask",
        "exit_state_valid_mask",
        "exit_terminal_mask",
        "exit_terminal_reason_index",
        *(f"exit_mtf_history_{tf}" for tf in tf_names),
        *(f"exit_mtf_history_time_ns_{tf}" for tf in tf_names),
        *(f"exit_mtf_gather_{tf}" for tf in tf_names),
    }
    scalar_names = {
        "schema_version",
        "lifecycle_schema_version",
        "split",
        "entry_row_index",
        "side_index",
        "chunk_index",
        "entry_m1_start_row",
        "chunk_m1_start_row",
        "chunk_start_bars_in_trade",
        "valid_state_count",
        "encoded_prefix_state_count",
        "successor_available",
        "successor_prefix_state_index",
        "right_censored",
        "terminal_reason",
        "lifecycle_manifest_sha256",
        "chunk_pointer_stream_sha256",
        "economic_exit_step_manifest_sha256",
        "economic_exit_step_stream_sha256",
        "economic_hold_step_stream_sha256",
        "scheduled_pair_chunk_pointer_sha256",
        "multi_tf_cache_identity_sha256",
        "unbounded_exit_training_readiness",
        "episode_pack_sha256",
    }
    if set(observed) != array_names | scalar_names:
        raise RuntimeError(f"{context}_UNIFIED_EXIT_CHUNK_PACK_KEYS_INVALID")
    valid_count = observed["valid_state_count"]
    chunk_start = observed["chunk_start_bars_in_trade"]
    successor = observed["successor_available"]
    right_censored = observed["right_censored"]
    encoded_count = observed["encoded_prefix_state_count"]
    if (
        observed["schema_version"]
        != UNIFIED_EXIT_EPISODE_PACK_V2_SCHEMA_VERSION
        or observed["lifecycle_schema_version"]
        != UNIFIED_EXIT_LIFECYCLE_V2_SCHEMA_VERSION
        or observed["split"] not in {"train", "val"}
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in (
                observed["entry_row_index"],
                observed["side_index"],
                observed["chunk_index"],
                observed["entry_m1_start_row"],
                observed["chunk_m1_start_row"],
                chunk_start,
            )
        )
        or observed["side_index"] not in (0, 1)
        or isinstance(valid_count, bool)
        or not isinstance(valid_count, int)
        or not 1 <= valid_count <= UNIFIED_EXIT_CHUNK_ROWS
        or type(successor) is not bool
        or type(right_censored) is not bool
        or (successor and right_censored)
        or encoded_count
        != chunk_start + valid_count + (1 if successor else 0)
        or observed["successor_prefix_state_index"]
        != (chunk_start + valid_count if successor else -1)
        or observed["chunk_m1_start_row"]
        != observed["entry_m1_start_row"] + chunk_start
        or observed["multi_tf_cache_identity_sha256"]
        != expected_mtf_cache_identity_sha256
        or (expected_split is not None and observed["split"] != expected_split)
        or (
            expected_lifecycle_manifest_sha256 is not None
            and observed["lifecycle_manifest_sha256"]
            != expected_lifecycle_manifest_sha256
        )
        or (
            expected_chunk_pointer_stream_sha256 is not None
            and observed["chunk_pointer_stream_sha256"]
            != expected_chunk_pointer_stream_sha256
        )
        or (
            expected_scheduled_pair_chunk_pointer_sha256 is not None
            and observed["scheduled_pair_chunk_pointer_sha256"]
            != expected_scheduled_pair_chunk_pointer_sha256
        )
    ):
        raise RuntimeError(f"{context}_UNIFIED_EXIT_CHUNK_PACK_IDENTITY_INVALID")
    for key in (
        "lifecycle_manifest_sha256",
        "chunk_pointer_stream_sha256",
        "economic_exit_step_manifest_sha256",
        "economic_exit_step_stream_sha256",
        "economic_hold_step_stream_sha256",
        "scheduled_pair_chunk_pointer_sha256",
        "multi_tf_cache_identity_sha256",
    ):
        digest = observed[key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise RuntimeError(f"{context}_UNIFIED_EXIT_CHUNK_PACK_HASH_INVALID")

    warm = EXIT_FEATURE_SEQUENCE_BARS - 1
    expected_shapes = {
        "exit_local_history_x": (
            warm + encoded_count,
            MODEL_NATIVE_SIGNAL_DIM,
        ),
        "exit_local_history_time_ns": (warm + encoded_count,),
        "exit_state_ctx_cont": (encoded_count, MODEL_NATIVE_CTX_CONT_DIM),
        "exit_state_ctx_cat": (encoded_count, MODEL_NATIVE_CTX_CAT_DIM),
        "exit_state_row_time_ns": (encoded_count,),
        "exit_decision_time_ns": (encoded_count,),
        "exit_path_x": (encoded_count, UNIFIED_EXIT_PATH_FEATURE_DIM),
        "exit_entry_bid_ask": (2,),
        "exit_now_reward_bps": (valid_count,),
        "hold_immediate_reward_bps": (valid_count,),
        "exit_policy_action_valid_mask": (valid_count, 2),
        "exit_bellman_target_valid_mask": (valid_count, 2),
        "exit_successor_observed_mask": (valid_count,),
        "exit_state_valid_mask": (valid_count,),
        "exit_terminal_mask": (valid_count,),
        "exit_terminal_reason_index": (valid_count,),
    }
    for name, shape in expected_shapes.items():
        array = np.asarray(observed[name])
        if array.shape != shape or (
            array.dtype.kind in "fc" and not np.isfinite(array).all()
        ):
            raise RuntimeError(f"{context}_UNIFIED_EXIT_CHUNK_PACK_SHAPE_INVALID:{name}")
    local_times = np.asarray(observed["exit_local_history_time_ns"], dtype=np.int64)
    state_times = np.asarray(observed["exit_state_row_time_ns"], dtype=np.int64)
    decisions = np.asarray(observed["exit_decision_time_ns"], dtype=np.int64)
    if (
        np.any(np.diff(local_times) <= 0)
        or not np.array_equal(state_times, local_times[warm:])
        or not np.array_equal(decisions, state_times + 60_000_000_000)
    ):
        raise RuntimeError(f"{context}_UNIFIED_EXIT_CHUNK_PACK_CLOCK_INVALID")
    state_valid = np.asarray(observed["exit_state_valid_mask"], dtype=np.bool_)
    action_valid = np.asarray(
        observed["exit_policy_action_valid_mask"], dtype=np.bool_
    )
    supervision = np.asarray(
        observed["exit_bellman_target_valid_mask"], dtype=np.bool_
    )
    successor_observed = np.asarray(
        observed["exit_successor_observed_mask"], dtype=np.bool_
    )
    terminal = np.asarray(observed["exit_terminal_mask"], dtype=np.bool_)
    reason = np.asarray(observed["exit_terminal_reason_index"], dtype=np.int64)
    economic_terminal = observed["terminal_reason"] == "economic_terminal"
    expected_successor_observed = state_valid.copy()
    expected_successor_observed[-1] = successor
    expected_supervision = action_valid.copy()
    expected_supervision[..., 0] &= successor_observed
    expected_terminal = np.zeros_like(terminal)
    expected_reason = np.zeros_like(reason)
    if economic_terminal:
        expected_terminal[-1] = True
        expected_reason[-1] = 2
    if (
        not state_valid.all()
        or not np.array_equal(action_valid[..., 1], state_valid)
        or not np.array_equal(action_valid[..., 0], state_valid & ~terminal)
        or not np.array_equal(successor_observed, expected_successor_observed)
        or not np.array_equal(supervision, expected_supervision)
        or not np.array_equal(terminal, expected_terminal)
        or not np.array_equal(reason, expected_reason)
        or (economic_terminal and (successor or right_censored))
        or (not economic_terminal and not successor and not right_censored)
    ):
        raise RuntimeError(f"{context}_UNIFIED_EXIT_CHUNK_PACK_MASK_INVALID")
    require_unified_exit_unbounded_training_readiness(
        observed["unbounded_exit_training_readiness"], context=context
    )
    for tf, canonical_name in zip(tf_names, EXIT_MTF_CONTEXT_TIMEFRAMES):
        history = np.asarray(observed[f"exit_mtf_history_{tf}"])
        times = np.asarray(observed[f"exit_mtf_history_time_ns_{tf}"], dtype=np.int64)
        gather = np.asarray(observed[f"exit_mtf_gather_{tf}"], dtype=np.int64)
        if (
            history.ndim != 2
            or history.shape[1] != MULTI_TF_FEATURE_COUNT_V4
            or history.shape[0] < int(per_tf_seq_lens[canonical_name])
            or times.shape != (history.shape[0],)
            or gather.shape != (encoded_count,)
            or np.any(np.diff(times) <= 0)
            or np.any(np.diff(gather) < 0)
            or np.any(gather < 0)
            or np.any(gather >= history.shape[0])
            or not np.isfinite(history).all()
        ):
            raise RuntimeError(f"{context}_UNIFIED_EXIT_CHUNK_PACK_MTF_INVALID:{tf}")
    sealed = seal_unified_exit_episode_pack_v2(
        {key: item for key, item in observed.items() if key != "episode_pack_sha256"}
    )
    if observed["episode_pack_sha256"] != sealed["episode_pack_sha256"]:
        raise RuntimeError(f"{context}_UNIFIED_EXIT_CHUNK_PACK_CONTENT_HASH_INVALID")
    return observed


__all__ = (
    "UNIFIED_EXIT_EPISODE_PACK_V2_SCHEMA_VERSION",
    "require_unified_exit_episode_pack_v2",
    "seal_unified_exit_episode_pack_v2",
    "unified_exit_episode_pack_v2_contract",
)
