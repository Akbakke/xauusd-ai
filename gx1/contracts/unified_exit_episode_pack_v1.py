"""Canonical one-pass episode pack for learned unified Exit.

One pack owns one Entry snapshot, both counterfactual sides, 479 exact
pre-entry M1 feature rows, 512 post-fill closed rows, one full side-specific
path per side, and one unique closed-bar history per MTF clock.  Per-state MTF
inputs are integer gathers into those histories; rolling feature windows and
padded path prefixes are forbidden.
"""

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
from gx1.features.htf_features import MULTI_TF_FEATURE_COUNT_V4
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_MAX_PATH_BARS,
    UNIFIED_EXIT_PATH_FEATURE_DIM,
)


UNIFIED_EXIT_EPISODE_PACK_SCHEMA_VERSION = (
    "gx1_unified_exit_causal_episode_pack_v1"
)
UNIFIED_EXIT_EPISODE_LOCAL_HISTORY_ROWS = (
    EXIT_FEATURE_SEQUENCE_BARS - 1 + UNIFIED_EXIT_MAX_PATH_BARS
)
UNIFIED_EXIT_EPISODE_STATE_COUNT = UNIFIED_EXIT_MAX_PATH_BARS
UNIFIED_EXIT_EPISODE_SIDE_COUNT = 2
UNIFIED_EXIT_EPISODE_ACTION_COUNT = 2

_FIXED_ARRAY_SHAPES = {
    "exit_local_history_x": (
        UNIFIED_EXIT_EPISODE_LOCAL_HISTORY_ROWS,
        MODEL_NATIVE_SIGNAL_DIM,
    ),
    "exit_local_history_time_ns": (UNIFIED_EXIT_EPISODE_LOCAL_HISTORY_ROWS,),
    "exit_state_ctx_cont": (
        UNIFIED_EXIT_EPISODE_STATE_COUNT,
        MODEL_NATIVE_CTX_CONT_DIM,
    ),
    "exit_state_ctx_cat": (
        UNIFIED_EXIT_EPISODE_STATE_COUNT,
        MODEL_NATIVE_CTX_CAT_DIM,
    ),
    "exit_state_row_time_ns": (UNIFIED_EXIT_EPISODE_STATE_COUNT,),
    "exit_decision_time_ns": (UNIFIED_EXIT_EPISODE_STATE_COUNT,),
    "exit_path_x": (
        UNIFIED_EXIT_EPISODE_SIDE_COUNT,
        UNIFIED_EXIT_EPISODE_STATE_COUNT,
        UNIFIED_EXIT_PATH_FEATURE_DIM,
    ),
    "exit_entry_bid_ask": (UNIFIED_EXIT_EPISODE_SIDE_COUNT, 2),
    "exit_now_reward_bps": (
        UNIFIED_EXIT_EPISODE_SIDE_COUNT,
        UNIFIED_EXIT_EPISODE_STATE_COUNT,
    ),
    "exit_action_valid_mask": (
        UNIFIED_EXIT_EPISODE_SIDE_COUNT,
        UNIFIED_EXIT_EPISODE_STATE_COUNT,
        UNIFIED_EXIT_EPISODE_ACTION_COUNT,
    ),
    "exit_state_valid_mask": (
        UNIFIED_EXIT_EPISODE_SIDE_COUNT,
        UNIFIED_EXIT_EPISODE_STATE_COUNT,
    ),
    "exit_terminal_mask": (
        UNIFIED_EXIT_EPISODE_SIDE_COUNT,
        UNIFIED_EXIT_EPISODE_STATE_COUNT,
    ),
    "exit_terminal_reason_index": (
        UNIFIED_EXIT_EPISODE_SIDE_COUNT,
        UNIFIED_EXIT_EPISODE_STATE_COUNT,
    ),
    "exit_episode_lengths": (UNIFIED_EXIT_EPISODE_SIDE_COUNT,),
}


def unified_exit_episode_pack_contract() -> dict[str, Any]:
    payload = {
        "schema_version": UNIFIED_EXIT_EPISODE_PACK_SCHEMA_VERSION,
        "local_history_rows": UNIFIED_EXIT_EPISODE_LOCAL_HISTORY_ROWS,
        "pre_entry_warm_rows": EXIT_FEATURE_SEQUENCE_BARS - 1,
        "post_fill_closed_rows": UNIFIED_EXIT_EPISODE_STATE_COUNT,
        "side_order": ["long", "short"],
        "action_order": ["HOLD", "EXIT_NOW"],
        "local_owner_roles": {
            "global_full_signal": "cross_family_residual_sequence",
            "eight_family_routes": "disjoint_one_owner_feature_partitions",
        },
        "mtf_timeframes": list(EXIT_MTF_CONTEXT_TIMEFRAMES),
        "mtf_storage": "unique_native_closed_histories_plus_last_closed_gathers",
        "path_storage": "one_complete_unpadded_path_per_side",
        "entry_token_storage": "one_frozen_token_per_entry_broadcast_in_model",
        "state_capacity": UNIFIED_EXIT_EPISODE_STATE_COUNT,
        "state_lengths": "explicit_per_side_not_encoder_weight_shape",
        "current_pack_supports_variable_length": False,
        "current_terminal_semantics": "capacity_forced_at_512",
        "open_next_wave": (
            "data_or_economic_terminal_plus_financing_and_slippage"
        ),
        "terminal_reason_index": {"0": "not_terminal", "1": "capacity_terminal"},
        "supervision": {
            "known_label": "current_executable_exit_now_reward_bps",
            "hold_label": "frozen_train_fitted_q_target_at_next_causal_state",
            "pathwise_hindsight_q": "diagnostic_upper_bound_only",
        },
        "legacy_repeated_480_windows": False,
        "legacy_padded_path_per_state": False,
        "future_rows_in_state_features": False,
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _update_array(digest: Any, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(name.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())


def require_unified_exit_episode_pack(
    value: Mapping[str, Any],
    *,
    per_tf_seq_lens: Mapping[str, int],
    expected_mtf_cache_identity_sha256: str,
    context: str,
) -> dict[str, Any]:
    """Validate and content-hash one fully assembled episode pack."""

    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context}_UNIFIED_EXIT_EPISODE_PACK_INVALID")
    observed = dict(value)
    tf_names = tuple(tf.lower() for tf in EXIT_MTF_CONTEXT_TIMEFRAMES)
    expected_keys = {
        "schema_version",
        "entry_row_index",
        "episode_index_by_side",
        "m1_start_row",
        "lifecycle_state_population_sha256",
        "multi_tf_cache_identity_sha256",
        "episode_pack_sha256",
        *_FIXED_ARRAY_SHAPES,
        *(f"exit_mtf_history_{tf}" for tf in tf_names),
        *(f"exit_mtf_history_time_ns_{tf}" for tf in tf_names),
        *(f"exit_mtf_gather_{tf}" for tf in tf_names),
    }
    if set(observed) != expected_keys:
        raise RuntimeError(
            f"{context}_UNIFIED_EXIT_EPISODE_PACK_KEYS_INVALID"
        )
    if (
        observed["schema_version"] != UNIFIED_EXIT_EPISODE_PACK_SCHEMA_VERSION
        or not isinstance(observed["entry_row_index"], int)
        or isinstance(observed["entry_row_index"], bool)
        or int(observed["entry_row_index"]) < 0
        or not isinstance(observed["m1_start_row"], int)
        or isinstance(observed["m1_start_row"], bool)
        or int(observed["m1_start_row"]) < 0
        or list(observed["episode_index_by_side"]) != sorted(
            list(observed["episode_index_by_side"])
        )
        or len(observed["episode_index_by_side"]) != 2
        or observed["multi_tf_cache_identity_sha256"]
        != expected_mtf_cache_identity_sha256
    ):
        raise RuntimeError(
            f"{context}_UNIFIED_EXIT_EPISODE_PACK_IDENTITY_INVALID"
        )
    for name, shape in _FIXED_ARRAY_SHAPES.items():
        array = np.asarray(observed[name])
        if array.shape != shape:
            raise RuntimeError(
                f"{context}_UNIFIED_EXIT_EPISODE_PACK_SHAPE_INVALID:{name}"
            )
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise RuntimeError(
                f"{context}_UNIFIED_EXIT_EPISODE_PACK_NONFINITE:{name}"
            )
    local_times = np.asarray(
        observed["exit_local_history_time_ns"], dtype=np.int64
    )
    state_times = np.asarray(observed["exit_state_row_time_ns"], dtype=np.int64)
    decision_times = np.asarray(
        observed["exit_decision_time_ns"], dtype=np.int64
    )
    warm = EXIT_FEATURE_SEQUENCE_BARS - 1
    if (
        np.any(np.diff(local_times) <= 0)
        or not np.array_equal(state_times, local_times[warm:])
        or not np.array_equal(decision_times, state_times + 60_000_000_000)
    ):
        raise RuntimeError(
            f"{context}_UNIFIED_EXIT_EPISODE_PACK_LOCAL_CLOCK_INVALID"
        )
    valid = np.asarray(observed["exit_action_valid_mask"], dtype=np.bool_)
    state_valid = np.asarray(observed["exit_state_valid_mask"], dtype=np.bool_)
    terminal = np.asarray(observed["exit_terminal_mask"], dtype=np.bool_)
    terminal_reason = np.asarray(
        observed["exit_terminal_reason_index"], dtype=np.int64
    )
    lengths = np.asarray(observed["exit_episode_lengths"], dtype=np.int64)
    if (
        not state_valid.all()
        or not np.array_equal(lengths, np.full(2, UNIFIED_EXIT_EPISODE_STATE_COUNT))
        or not terminal[:, -1].all()
        or terminal[:, :-1].any()
        or np.any(terminal_reason[:, :-1] != 0)
        or np.any(terminal_reason[:, -1] != 1)
        or not np.array_equal(valid[..., 1], state_valid)
        or not np.array_equal(valid[..., 0], state_valid & ~terminal)
        or valid[:, -1].tolist() != [[False, True], [False, True]]
    ):
        raise RuntimeError(
            f"{context}_UNIFIED_EXIT_EPISODE_PACK_TARGET_MASK_INVALID"
        )
    for tf, canonical_name in zip(tf_names, EXIT_MTF_CONTEXT_TIMEFRAMES):
        history = np.asarray(observed[f"exit_mtf_history_{tf}"])
        times = np.asarray(
            observed[f"exit_mtf_history_time_ns_{tf}"], dtype=np.int64
        )
        gather = np.asarray(observed[f"exit_mtf_gather_{tf}"], dtype=np.int64)
        warm_rows = int(per_tf_seq_lens[canonical_name])
        if (
            history.ndim != 2
            or history.shape[0] < warm_rows
            or history.shape[1] != MULTI_TF_FEATURE_COUNT_V4
            or times.shape != (history.shape[0],)
            or gather.shape != (UNIFIED_EXIT_EPISODE_STATE_COUNT,)
            or not np.isfinite(history).all()
            or np.any(np.diff(times) <= 0)
            or np.any(np.diff(gather) < 0)
            or int(gather[0]) != warm_rows - 1
            or int(gather[-1]) != history.shape[0] - 1
            or np.any(gather < 0)
            or np.any(gather >= history.shape[0])
        ):
            raise RuntimeError(
                f"{context}_UNIFIED_EXIT_EPISODE_PACK_MTF_INVALID:{tf}"
            )
    digest = hashlib.sha256()
    identity = {
        "schema_version": observed["schema_version"],
        "entry_row_index": observed["entry_row_index"],
        "episode_index_by_side": list(observed["episode_index_by_side"]),
        "m1_start_row": observed["m1_start_row"],
        "lifecycle_state_population_sha256": observed[
            "lifecycle_state_population_sha256"
        ],
        "multi_tf_cache_identity_sha256": observed[
            "multi_tf_cache_identity_sha256"
        ],
    }
    digest.update(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    for name in sorted(expected_keys - {"episode_pack_sha256"} - set(identity)):
        _update_array(digest, name, np.asarray(observed[name]))
    if observed["episode_pack_sha256"] != digest.hexdigest():
        raise RuntimeError(
            f"{context}_UNIFIED_EXIT_EPISODE_PACK_HASH_INVALID"
        )
    return observed


def seal_unified_exit_episode_pack(value: Mapping[str, Any]) -> dict[str, Any]:
    """Seal array bytes before the strict validator checks source identities."""

    observed = dict(value)
    digest = hashlib.sha256()
    identity_keys = (
        "schema_version",
        "entry_row_index",
        "episode_index_by_side",
        "m1_start_row",
        "lifecycle_state_population_sha256",
        "multi_tf_cache_identity_sha256",
    )
    identity = {name: observed[name] for name in identity_keys}
    digest.update(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    for name in sorted(set(observed) - set(identity_keys)):
        _update_array(digest, name, np.asarray(observed[name]))
    observed["episode_pack_sha256"] = digest.hexdigest()
    return observed


__all__ = (
    "UNIFIED_EXIT_EPISODE_LOCAL_HISTORY_ROWS",
    "UNIFIED_EXIT_EPISODE_PACK_SCHEMA_VERSION",
    "UNIFIED_EXIT_EPISODE_STATE_COUNT",
    "require_unified_exit_episode_pack",
    "seal_unified_exit_episode_pack",
    "unified_exit_episode_pack_contract",
)
