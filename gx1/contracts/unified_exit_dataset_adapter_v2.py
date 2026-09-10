"""Lazy compact-lifecycle adapter for canonical Exit chunk training."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_exit_feature_base_v1 import (
    EXIT_DECISION_BAR_SECONDS,
    EXIT_FEATURE_SEQUENCE_BARS,
)
from gx1.contracts.unified_exit_economics_objective_v2 import compose_economic_step
from gx1.contracts.unified_exit_episode_pack_v2 import (
    UNIFIED_EXIT_EPISODE_PACK_V2_SCHEMA_VERSION,
    require_unified_exit_episode_pack_v2,
    seal_unified_exit_episode_pack_v2,
)
from gx1.contracts.unified_exit_fitted_q_v1 import (
    require_unified_exit_unbounded_training_readiness,
)
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_PATH_PRICE_FIELDS,
    unified_exit_causal_prefix_path_tensor_from_values,
)
from gx1.scripts.materialize_unified_exit_lifecycle_v2 import (
    COMPACT_LIFECYCLE_SCHEMA_VERSION,
    _compact_pointer_stream_sha256,
    require_compact_split,
    scheduled_pair_chunk_pointer,
)

ECONOMIC_EXIT_STEP_MANIFEST_SCHEMA_VERSION = (
    "gx1_unified_exit_economic_exit_now_step_manifest_v1"
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def seal_economic_exit_step_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    observed = dict(value)
    if "manifest_sha256" in observed:
        raise RuntimeError("UNIFIED_EXIT_ECONOMIC_STEP_MANIFEST_ALREADY_SEALED")
    observed["manifest_sha256"] = _canonical_sha256(observed)
    return observed


class UnifiedExitDatasetAdapterV2:
    """Materialize one scheduled pair slot into one pack per active side."""

    def __init__(
        self,
        *,
        compact_rows: pd.DataFrame,
        compact_manifest: Mapping[str, Any],
        source_owner: Any,
        epoch_index: int,
        expected_m1_source_sha256: str,
        expected_entry_binding_sha256: str,
        expected_gap_classification_source_sha256: str,
        economics_readiness: Mapping[str, Any],
        economic_exit_step_manifest: Mapping[str, Any],
        economic_exit_step_provider: Callable[
            [int, int, str, int, int], Mapping[str, Any]
        ],
        mtf_materializer: Callable[[np.ndarray], Mapping[str, np.ndarray]],
        per_tf_seq_lens: Mapping[str, int],
        mtf_cache_identity_sha256: str,
    ) -> None:
        manifest = dict(compact_manifest)
        if (
            manifest.get("compact_schema_version") != COMPACT_LIFECYCLE_SCHEMA_VERSION
            or manifest.get("split") not in {"train", "val"}
            or manifest.get("test_accessed") is not False
            or manifest.get("target_q_stored") is not False
            or manifest.get("manifest_sha256")
            != _canonical_sha256(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "manifest_sha256"
                }
            )
            or manifest.get("compact_pointer_stream_sha256")
            != _compact_pointer_stream_sha256(compact_rows)
        ):
            raise RuntimeError("UNIFIED_EXIT_DATASET_V2_COMPACT_MANIFEST_INVALID")
        require_compact_split(
            compact_rows,
            split_end=manifest["split_end_utc"],
            m1_times=source_owner._m1_times,
            expected_m1_source_sha256=expected_m1_source_sha256,
            expected_entry_binding_sha256=expected_entry_binding_sha256,
            expected_gap_classification_source_sha256=(
                expected_gap_classification_source_sha256
            ),
        )
        readiness = require_unified_exit_unbounded_training_readiness(
            economics_readiness, context="UNIFIED_EXIT_DATASET_V2"
        )
        economic_manifest = dict(economic_exit_step_manifest)
        provider_manifest = getattr(
            economic_exit_step_provider, "economic_exit_step_manifest", None
        )
        expected_economic_keys = {
            "schema_version",
            "split",
            "lifecycle_manifest_sha256",
            "economics_objective_contract_sha256",
            "economic_step_model_sha256",
            "economic_step_source_manifest_sha256",
            "test_data_used",
            "manifest_sha256",
        }
        if (
            set(economic_manifest) != expected_economic_keys
            or economic_manifest["schema_version"]
            != ECONOMIC_EXIT_STEP_MANIFEST_SCHEMA_VERSION
            or economic_manifest["split"] != manifest["split"]
            or economic_manifest["lifecycle_manifest_sha256"]
            != manifest["manifest_sha256"]
            or economic_manifest["economics_objective_contract_sha256"]
            != readiness["economics_objective_contract"]["contract_sha256"]
            or economic_manifest["test_data_used"] is not False
            or economic_manifest["manifest_sha256"]
            != _canonical_sha256(
                {
                    key: value
                    for key, value in economic_manifest.items()
                    if key != "manifest_sha256"
                }
            )
            or not callable(economic_exit_step_provider)
            or (
                provider_manifest is not None
                and dict(provider_manifest) != economic_manifest
            )
        ):
            raise RuntimeError("UNIFIED_EXIT_DATASET_V2_ECONOMIC_STEPS_INVALID")
        if (
            isinstance(epoch_index, bool)
            or not isinstance(epoch_index, int)
            or epoch_index < 0
        ):
            raise RuntimeError("UNIFIED_EXIT_DATASET_V2_EPOCH_INVALID")
        self._rows = compact_rows.set_index("entry_row_index", drop=False)
        self._manifest = manifest
        self._source = source_owner
        self._epoch_index = epoch_index
        self._readiness = dict(economics_readiness)
        self._economic_manifest = economic_manifest
        self._economic_provider = economic_exit_step_provider
        self._mtf_materializer = mtf_materializer
        self.per_tf_seq_lens = dict(per_tf_seq_lens)
        self.mtf_cache_identity_sha256 = mtf_cache_identity_sha256

    def set_epoch_index(self, epoch_index: int) -> None:
        """Select the outcome-blind TRAIN chunk schedule for one epoch."""

        if (
            self._manifest["split"] != "train"
            or isinstance(epoch_index, bool)
            or not isinstance(epoch_index, int)
            or epoch_index < 0
        ):
            raise RuntimeError("UNIFIED_EXIT_DATASET_V2_EPOCH_INVALID")
        self._epoch_index = epoch_index

    def require_pack(self, value: Mapping[str, Any]) -> dict[str, Any]:
        entry_row = int(value["entry_row_index"])
        if (
            value.get("economic_exit_step_manifest_sha256")
            != self._economic_manifest["manifest_sha256"]
        ):
            raise RuntimeError(
                "UNIFIED_EXIT_DATASET_V2_PACK_ECONOMICS_IDENTITY_INVALID"
            )
        row = self._rows.loc[entry_row]
        side_index = int(value["side_index"])
        side_name = ("long", "short")[side_index]
        matches = [
            pointer
            for pointer, candidate_side in self._scheduled_active_items(row)
            if candidate_side == side_index
            and pointer["schedule_sha256"]
            == value.get("scheduled_pair_chunk_pointer_sha256")
        ]
        if len(matches) != 1:
            raise RuntimeError("UNIFIED_EXIT_DATASET_V2_PACK_SCHEDULE_INVALID")
        pointer = matches[0]
        side_pointer = pointer["sides"][side_name]
        if (
            int(value["chunk_index"]) != int(pointer["pair_chunk_slot"])
            or int(value["chunk_start_bars_in_trade"])
            != int(pointer["chunk_start_bars_in_trade"])
            or int(value["valid_state_count"]) != int(side_pointer["valid_state_count"])
        ):
            raise RuntimeError("UNIFIED_EXIT_DATASET_V2_PACK_SCHEDULE_INVALID")
        return require_unified_exit_episode_pack_v2(
            value,
            per_tf_seq_lens=self.per_tf_seq_lens,
            expected_mtf_cache_identity_sha256=self.mtf_cache_identity_sha256,
            expected_split=self._manifest["split"],
            expected_lifecycle_manifest_sha256=self._manifest["manifest_sha256"],
            expected_chunk_pointer_stream_sha256=side_pointer["pointer_stream_sha256"],
            expected_scheduled_pair_chunk_pointer_sha256=pointer["schedule_sha256"],
            context="UNIFIED_EXIT_DATASET_V2_PACK",
        )

    def _scheduled_active_items(self, row) -> list[tuple[dict[str, Any], int]]:
        pair_count = int(row["pair_chunk_count"])
        pointers = [
            scheduled_pair_chunk_pointer(
                compact_row=row.to_dict(),
                epoch_index=position,
                lineage_sha256=self._manifest["schedule_lineage_sha256"],
                split=self._manifest["split"],
            )
            for position in range(pair_count)
        ]
        return [
            (pointer, side_index)
            for pointer in pointers
            for side_index, side_name in enumerate(("long", "short"))
            if pointer["sides"][side_name]["active"]
        ]

    def materialize(self, entry_row_index: int) -> dict[str, Any] | None:
        if entry_row_index not in self._rows.index:
            return None
        row = self._rows.loc[entry_row_index]
        if isinstance(row, pd.DataFrame):
            raise RuntimeError("UNIFIED_EXIT_DATASET_V2_DUPLICATE_ENTRY_ROW")
        active_items = self._scheduled_active_items(row)
        if not active_items:
            return None
        pointer, side_index = active_items[self._epoch_index % len(active_items)]
        side_name = ("long", "short")[side_index]
        return self._materialize_side(
            row=row,
            pointer=pointer,
            side_pointer=pointer["sides"][side_name],
            side_index=side_index,
        )

    def materialize_validation(
        self, entry_row_index: int
    ) -> tuple[dict[str, Any], ...]:
        """Materialize deterministic full chunk coverage for both VAL sides."""

        if entry_row_index not in self._rows.index:
            return ()
        row = self._rows.loc[entry_row_index]
        return tuple(
            self._materialize_side(
                row=row,
                pointer=pointer,
                side_pointer=pointer["sides"][("long", "short")[side_index]],
                side_index=side_index,
            )
            for pointer, side_index in self._scheduled_active_items(row)
        )

    def _materialize_side(self, *, row, pointer, side_pointer, side_index):
        entry_row = int(row["entry_row_index"])
        start = int(row["entry_m1_start_row"])
        chunk_start = int(pointer["chunk_start_bars_in_trade"])
        valid_count = int(side_pointer["valid_state_count"])
        successor = bool(side_pointer["successor_available"])
        encoded_count = chunk_start + valid_count + int(successor)
        warm = EXIT_FEATURE_SEQUENCE_BARS - 1
        feature_offset = int(self._source._feature_row_offset)
        local_start = start - warm - feature_offset
        local_stop = start + encoded_count - feature_offset
        current_start = start - feature_offset
        current_stop = start + encoded_count - feature_offset
        if (
            local_start < 0
            or local_stop > len(self._source._m1_feature_times)
            or current_start < 0
            or current_stop > len(self._source._m1_feature_times)
        ):
            raise RuntimeError("UNIFIED_EXIT_DATASET_V2_FEATURE_HISTORY_INSUFFICIENT")
        source_slice = slice(start, start + encoded_count)
        entry_bid = float(self._source._m1["bid_open"][start])
        entry_ask = float(self._source._m1["ask_open"][start])
        price_values = np.column_stack(
            [
                self._source._m1[name][source_slice]
                for name in UNIFIED_EXIT_PATH_PRICE_FIELDS
            ]
        )
        path = unified_exit_causal_prefix_path_tensor_from_values(
            price_values=price_values,
            volumes=self._source._m1["volume"][source_slice],
            entry_bid=entry_bid,
            entry_ask=entry_ask,
        )

        def load_slice(action: str, slice_start: int, slice_stop: int):
            try:
                envelope = self._economic_provider(
                    entry_row, side_index, action, slice_start, slice_stop
                )
            except (KeyError, FileNotFoundError, OSError) as exc:
                raise RuntimeError(
                    "UNIFIED_EXIT_DATASET_V2_ECONOMIC_STEPS_MISSING"
                ) from exc
            if not isinstance(envelope, Mapping):
                raise RuntimeError("UNIFIED_EXIT_DATASET_V2_ECONOMIC_STEPS_MISSING")
            observed = dict(envelope)
            expected = {
                "schema_version",
                "entry_row_index",
                "side_index",
                "action",
                "start_state_index",
                "stop_state_index",
                "steps",
                "economic_step_model_sha256",
                "economic_step_source_manifest_sha256",
                "slice_sha256",
            }
            if (
                set(observed) != expected
                or observed["schema_version"]
                != "gx1_unified_exit_economic_step_slice_v1"
                or observed["entry_row_index"] != entry_row
                or observed["side_index"] != side_index
                or observed["action"] != action
                or observed["start_state_index"] != slice_start
                or observed["stop_state_index"] != slice_stop
                or observed["economic_step_model_sha256"]
                != self._economic_manifest["economic_step_model_sha256"]
                or observed["economic_step_source_manifest_sha256"]
                != self._economic_manifest["economic_step_source_manifest_sha256"]
                or not isinstance(observed["steps"], list)
                or len(observed["steps"]) != slice_stop - slice_start
                or observed["slice_sha256"]
                != _canonical_sha256(
                    {
                        key: value
                        for key, value in observed.items()
                        if key != "slice_sha256"
                    }
                )
            ):
                raise RuntimeError(
                    "UNIFIED_EXIT_DATASET_V2_ECONOMIC_STEP_SLICE_INVALID"
                )
            return observed

        exit_slice = load_slice("exit_now", chunk_start, chunk_start + valid_count)
        hold_stop = min(
            chunk_start + valid_count,
            int(row[f"{('long', 'short')[side_index]}_lifecycle_state_count"]) - 1,
        )
        hold_slice = load_slice("hold", chunk_start, hold_stop)
        raw_steps = exit_slice["steps"]
        raw_hold_steps = hold_slice["steps"]
        if (
            not isinstance(raw_steps, Sequence)
            or isinstance(raw_steps, (str, bytes))
            or not isinstance(raw_hold_steps, Sequence)
            or isinstance(raw_hold_steps, (str, bytes))
        ):
            raise RuntimeError("UNIFIED_EXIT_DATASET_V2_ECONOMIC_STEPS_MISSING")
        contract = self._readiness["economics_objective_contract"]
        composed = [
            compose_economic_step(step, contract=contract) for step in raw_steps
        ]
        composed_hold = [
            compose_economic_step(step, contract=contract) for step in raw_hold_steps
        ]
        economic_terminal = side_pointer["terminal_reason"] == "economic_terminal"
        for position, step in enumerate(composed):
            expected_event = (
                "ECONOMIC_TERMINAL"
                if economic_terminal and position == valid_count - 1
                else "EXIT_NOW"
            )
            if step["event_kind"] != expected_event:
                raise RuntimeError(
                    "UNIFIED_EXIT_DATASET_V2_ECONOMIC_STEP_EVENT_INVALID"
                )
        if any(step["event_kind"] != "HOLD" for step in composed_hold):
            raise RuntimeError("UNIFIED_EXIT_DATASET_V2_ECONOMIC_STEP_EVENT_INVALID")
        rewards = np.asarray(
            [
                step["undiscounted_risk_adjusted_utility_increment_bps"]
                for step in composed
            ],
            dtype=np.float32,
        )
        hold_rewards = np.zeros(valid_count, dtype=np.float32)
        hold_rewards[: len(composed_hold)] = np.asarray(
            [
                step["undiscounted_risk_adjusted_utility_increment_bps"]
                for step in composed_hold
            ],
            dtype=np.float32,
        )
        state_valid = np.ones(valid_count, dtype=np.bool_)
        terminal = np.zeros(valid_count, dtype=np.bool_)
        reason = np.zeros(valid_count, dtype=np.int64)
        if economic_terminal:
            terminal[-1] = True
            reason[-1] = 2
        policy = np.repeat(state_valid[:, None], 2, axis=1)
        policy[:, 0] &= ~terminal
        successor_observed = state_valid.copy()
        successor_observed[-1] = successor
        bellman = policy.copy()
        bellman[:, 0] &= successor_observed
        state_times = np.asarray(
            self._source._m1_times.asi8[start : start + encoded_count], dtype=np.int64
        )
        core = {
            "schema_version": UNIFIED_EXIT_EPISODE_PACK_V2_SCHEMA_VERSION,
            "lifecycle_schema_version": self._manifest["schema_version"],
            "split": self._manifest["split"],
            "entry_row_index": entry_row,
            "side_index": side_index,
            "chunk_index": int(pointer["pair_chunk_slot"]),
            "entry_m1_start_row": start,
            "chunk_m1_start_row": int(pointer["chunk_m1_start_row"]),
            "chunk_start_bars_in_trade": chunk_start,
            "valid_state_count": valid_count,
            "encoded_prefix_state_count": encoded_count,
            "successor_available": successor,
            "successor_prefix_state_index": encoded_count - 1 if successor else -1,
            "right_censored": bool(side_pointer["right_censored"]),
            "terminal_reason": side_pointer["terminal_reason"],
            "lifecycle_manifest_sha256": self._manifest["manifest_sha256"],
            "chunk_pointer_stream_sha256": side_pointer["pointer_stream_sha256"],
            "economic_exit_step_manifest_sha256": self._economic_manifest[
                "manifest_sha256"
            ],
            "economic_exit_step_stream_sha256": exit_slice["slice_sha256"],
            "economic_hold_step_stream_sha256": hold_slice["slice_sha256"],
            "scheduled_pair_chunk_pointer_sha256": pointer["schedule_sha256"],
            "multi_tf_cache_identity_sha256": self.mtf_cache_identity_sha256,
            "unbounded_exit_training_readiness": self._readiness,
            "exit_local_history_x": np.ascontiguousarray(
                self._source._m1_features["signal"][local_start:local_stop],
                dtype=np.float32,
            ),
            "exit_local_history_time_ns": np.asarray(
                self._source._m1_feature_times.asi8[local_start:local_stop],
                dtype=np.int64,
            ),
            "exit_state_ctx_cont": np.ascontiguousarray(
                self._source._m1_features["ctx_cont"][current_start:current_stop],
                dtype=np.float32,
            ),
            "exit_state_ctx_cat": np.ascontiguousarray(
                self._source._m1_features["ctx_cat"][current_start:current_stop],
                dtype=np.int64,
            ),
            "exit_state_row_time_ns": state_times,
            "exit_decision_time_ns": state_times
            + int(pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS).value),
            "exit_path_x": np.ascontiguousarray(path, dtype=np.float32),
            "exit_entry_bid_ask": np.asarray([entry_bid, entry_ask], dtype=np.float64),
            "exit_now_reward_bps": rewards,
            "hold_immediate_reward_bps": hold_rewards,
            "exit_policy_action_valid_mask": policy,
            "exit_bellman_target_valid_mask": bellman,
            "exit_successor_observed_mask": successor_observed,
            "exit_state_valid_mask": state_valid,
            "exit_terminal_mask": terminal,
            "exit_terminal_reason_index": reason,
        }
        core.update(self._mtf_materializer(state_times))
        pack = seal_unified_exit_episode_pack_v2(core)
        return self.require_pack(pack)


__all__ = (
    "ECONOMIC_EXIT_STEP_MANIFEST_SCHEMA_VERSION",
    "UnifiedExitDatasetAdapterV2",
    "seal_economic_exit_step_manifest",
)
