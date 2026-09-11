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
    unified_exit_path_tensor_from_values,
)
from gx1.contracts.unified_exit_lifetime_summary_v1 import build_lifetime_summary
from gx1.contracts.unified_exit_market_closure_authority_v1 import (
    m1_clock_sha256,
    require_market_closure_authority,
)
from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    FIRST_STATE_BRIDGE_SCHEMA_VERSION,
    require_lifetime_summary_normalization,
)
from gx1.contracts.unified_exit_random_access_index_v1 import (
    require_random_access_index,
    require_random_access_index_manifest,
)
from gx1.contracts.unified_exit_random_access_sampler_v1 import (
    require_random_access_sampler_contract,
    schedule_random_access_entry_anchors,
    schedule_random_access_epoch,
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
ECONOMIC_TRAINING_PROJECTION_SCHEMA_VERSION = (
    "gx1_unified_exit_economic_training_projection_v1"
)
_EXIT_NOW_EVENT_INDEX = 0
_HOLD_EVENT_INDEX = 1
_ECONOMIC_TERMINAL_EVENT_INDEX = 2


class _RangeExtrema:
    """O(n)-memory/O(log n)-query extrema with stable first-index ties."""

    def __init__(self, values: Any, *, maximum: bool) -> None:
        base = np.ascontiguousarray(values, dtype="<f8")
        if base.ndim != 1 or base.size < 1 or not np.isfinite(base).all():
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_PRICE_SOURCE_INVALID")
        self._maximum = maximum
        self._count = base.size
        size = 1 << (base.size - 1).bit_length()
        neutral = -np.inf if maximum else np.inf
        self._values = np.full(size * 2, neutral, dtype="<f8")
        self._indices = np.full(size * 2, np.iinfo(np.int64).max, dtype="<i8")
        self._values[size : size + base.size] = base
        self._indices[size : size + base.size] = np.arange(base.size, dtype=np.int64)
        self._size = size
        for node in range(size - 1, 0, -1):
            left, right = node * 2, node * 2 + 1
            if self._better(
                self._values[left],
                self._indices[left],
                self._values[right],
                self._indices[right],
            ):
                chosen = left
            else:
                chosen = right
            self._values[node] = self._values[chosen]
            self._indices[node] = self._indices[chosen]

    def _better(self, a: float, ai: int, b: float, bi: int) -> bool:
        return (a > b if self._maximum else a < b) or (a == b and ai <= bi)

    def query(self, left: int, stop: int) -> tuple[float, int]:
        length = stop - left
        if left < 0 or stop > self._count or length < 1:
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_SUMMARY_RANGE_INVALID")
        lo, hi = left + self._size, stop + self._size
        value = -np.inf if self._maximum else np.inf
        index = np.iinfo(np.int64).max
        while lo < hi:
            if lo & 1:
                candidate, candidate_index = self._values[lo], self._indices[lo]
                if self._better(candidate, candidate_index, value, index):
                    value, index = candidate, candidate_index
                lo += 1
            if hi & 1:
                hi -= 1
                candidate, candidate_index = self._values[hi], self._indices[hi]
                if self._better(candidate, candidate_index, value, index):
                    value, index = candidate, candidate_index
            lo //= 2
            hi //= 2
        return float(value), int(index)


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


def _array_mapping_sha256(value: Mapping[str, Any]) -> str:
    """Hash exact scalar identities and C-contiguous array bytes."""

    digest = hashlib.sha256()
    for name in sorted(value):
        item = value[name]
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        if isinstance(item, np.ndarray):
            digest.update(item.dtype.str.encode("ascii"))
            digest.update(b"\0")
            digest.update(np.asarray(item.shape, dtype="<i8").tobytes())
            digest.update(item.tobytes(order="C"))
        else:
            digest.update(
                json.dumps(
                    item, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
            )
    return digest.hexdigest()


def seal_economic_training_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Seal the immutable, model-consumed projection of one economic slice."""

    observed = dict(value)
    forbidden = {"exit_stream_sha256", "hold_stream_sha256", "projection_sha256"}
    if forbidden & set(observed):
        raise RuntimeError("UNIFIED_EXIT_ECONOMIC_PROJECTION_ALREADY_SEALED")
    for name in (
        "exit_event_kind_index",
        "exit_reward_bps",
        "hold_event_kind_index",
        "hold_reward_bps",
    ):
        array = observed.get(name)
        if not isinstance(array, np.ndarray):
            raise RuntimeError("UNIFIED_EXIT_ECONOMIC_PROJECTION_ARRAY_INVALID")
        array.setflags(write=False)
    common = {
        key: observed[key]
        for key in (
            "schema_version",
            "entry_row_index",
            "side_index",
            "economic_step_model_sha256",
            "economic_step_source_manifest_sha256",
        )
    }
    observed["exit_stream_sha256"] = _array_mapping_sha256(
        {
            **common,
            "action": "exit_now",
            "start_state_index": observed["start_state_index"],
            "stop_state_index": observed["stop_state_index"],
            "event_kind_index": observed["exit_event_kind_index"],
            "reward_bps": observed["exit_reward_bps"],
        }
    )
    observed["hold_stream_sha256"] = _array_mapping_sha256(
        {
            **common,
            "action": "hold",
            "start_state_index": observed["start_state_index"],
            "stop_state_index": observed["hold_stop_state_index"],
            "event_kind_index": observed["hold_event_kind_index"],
            "reward_bps": observed["hold_reward_bps"],
        }
    )
    observed["projection_sha256"] = _array_mapping_sha256(observed)
    return observed


def require_economic_training_projection(
    value: Mapping[str, Any],
    *,
    entry_row_index: int,
    side_index: int,
    start_state_index: int,
    stop_state_index: int,
    hold_stop_state_index: int,
    economic_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact immutable arrays without rebuilding scalar step dictionaries."""

    expected = {
        "schema_version",
        "entry_row_index",
        "side_index",
        "start_state_index",
        "stop_state_index",
        "hold_stop_state_index",
        "exit_event_kind_index",
        "exit_reward_bps",
        "hold_event_kind_index",
        "hold_reward_bps",
        "economic_step_model_sha256",
        "economic_step_source_manifest_sha256",
        "exit_stream_sha256",
        "hold_stream_sha256",
        "projection_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RuntimeError("UNIFIED_EXIT_ECONOMIC_PROJECTION_KEYS_INVALID")
    observed = dict(value)
    forbidden = {"exit_stream_sha256", "hold_stream_sha256", "projection_sha256"}
    if (
        observed["schema_version"] != ECONOMIC_TRAINING_PROJECTION_SCHEMA_VERSION
        or observed["entry_row_index"] != entry_row_index
        or observed["side_index"] != side_index
        or observed["start_state_index"] != start_state_index
        or observed["stop_state_index"] != stop_state_index
        or observed["hold_stop_state_index"] != hold_stop_state_index
        or observed["economic_step_model_sha256"]
        != economic_manifest["economic_step_model_sha256"]
        or observed["economic_step_source_manifest_sha256"]
        != economic_manifest["economic_step_source_manifest_sha256"]
    ):
        raise RuntimeError("UNIFIED_EXIT_ECONOMIC_PROJECTION_IDENTITY_INVALID")
    shapes_and_dtypes = {
        "exit_event_kind_index": (
            (stop_state_index - start_state_index,),
            np.dtype("u1"),
        ),
        "exit_reward_bps": ((stop_state_index - start_state_index,), np.dtype("<f8")),
        "hold_event_kind_index": (
            (hold_stop_state_index - start_state_index,),
            np.dtype("u1"),
        ),
        "hold_reward_bps": (
            (hold_stop_state_index - start_state_index,),
            np.dtype("<f8"),
        ),
    }
    for name, (shape, dtype) in shapes_and_dtypes.items():
        array = observed[name]
        if (
            not isinstance(array, np.ndarray)
            or array.shape != shape
            or array.dtype != dtype
            or not array.flags.c_contiguous
            or array.flags.writeable
            or (array.dtype.kind == "f" and not np.isfinite(array).all())
        ):
            raise RuntimeError(f"UNIFIED_EXIT_ECONOMIC_PROJECTION_ARRAY_INVALID:{name}")
    if np.any(
        ~np.isin(
            observed["exit_event_kind_index"],
            (_EXIT_NOW_EVENT_INDEX, _ECONOMIC_TERMINAL_EVENT_INDEX),
        )
    ) or np.any(observed["hold_event_kind_index"] != _HOLD_EVENT_INDEX):
        raise RuntimeError("UNIFIED_EXIT_ECONOMIC_PROJECTION_EVENT_INVALID")
    sealed = seal_economic_training_projection(
        {key: item for key, item in observed.items() if key not in forbidden}
    )
    if any(
        observed[name] != sealed[name]
        for name in ("exit_stream_sha256", "hold_stream_sha256", "projection_sha256")
    ):
        raise RuntimeError("UNIFIED_EXIT_ECONOMIC_PROJECTION_HASH_INVALID")
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
        self._random_access_train: dict[str, Any] | None = None

    @classmethod
    def from_random_access_index_v1(
        cls,
        *,
        index_rows: pd.DataFrame,
        index_manifest: Mapping[str, Any],
        source_owner: Any,
        epoch_index: int,
        economics_readiness: Mapping[str, Any],
        economic_exit_step_manifest: Mapping[str, Any],
        economic_exit_step_provider: Callable[
            [int, int, str, int, int], Mapping[str, Any]
        ],
        mtf_materializer: Callable[[np.ndarray], Mapping[str, np.ndarray]],
        per_tf_seq_lens: Mapping[str, int],
        mtf_cache_identity_sha256: str,
    ) -> "UnifiedExitDatasetAdapterV2":
        """Construct the random-access path without any compact/chunk artifact."""

        manifest = require_random_access_index_manifest(
            index_manifest,
            expected_split=str(index_manifest.get("split")),
            index_frame=index_rows,
        )
        readiness = require_unified_exit_unbounded_training_readiness(
            economics_readiness, context="UNIFIED_EXIT_RANDOM_ACCESS_INDEX_ADAPTER"
        )
        economic_manifest = dict(economic_exit_step_manifest)
        provider_manifest = getattr(
            economic_exit_step_provider, "economic_exit_step_manifest", None
        )
        if (
            manifest["split"] != "train"
            or manifest["manifest_sha256"]
            != economic_manifest.get("lifecycle_manifest_sha256")
            or economic_manifest.get("schema_version")
            != ECONOMIC_EXIT_STEP_MANIFEST_SCHEMA_VERSION
            or economic_manifest.get("split") != "train"
            or economic_manifest.get("economics_objective_contract_sha256")
            != readiness["economics_objective_contract"]["contract_sha256"]
            or economic_manifest.get("test_data_used") is not False
            or economic_manifest.get("manifest_sha256")
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
            or isinstance(epoch_index, bool)
            or not isinstance(epoch_index, int)
            or epoch_index < 0
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_INDEX_ADAPTER_INVALID")
        native = require_random_access_index(index_rows, expected_split="train")
        rows = pd.DataFrame(
            {
                "entry_row_index": native["entry_row_index"].to_numpy(dtype="<i8"),
                "entry_m1_start_row": native["parent_m1_start_row"].to_numpy(
                    dtype="<i8"
                ),
                "first_state_row_time": pd.to_datetime(
                    native["first_state_time_ns"].to_numpy(dtype="<i8"), utc=True
                ),
                "long_lifecycle_state_count": native["lifecycle_state_count"].to_numpy(
                    dtype="<i8"
                ),
                "short_lifecycle_state_count": native["lifecycle_state_count"].to_numpy(
                    dtype="<i8"
                ),
                "long_economic_terminal": native["economic_terminal"].to_numpy(
                    dtype=np.bool_
                ),
                "short_economic_terminal": native["economic_terminal"].to_numpy(
                    dtype=np.bool_
                ),
            }
        )
        instance = cls.__new__(cls)
        instance._rows = rows.set_index("entry_row_index", drop=False)
        instance._manifest = manifest
        instance._source = source_owner
        instance._epoch_index = epoch_index
        instance._readiness = dict(economics_readiness)
        instance._economic_manifest = economic_manifest
        instance._economic_provider = economic_exit_step_provider
        instance._mtf_materializer = mtf_materializer
        instance.per_tf_seq_lens = dict(per_tf_seq_lens)
        instance.mtf_cache_identity_sha256 = mtf_cache_identity_sha256
        instance._random_access_train = None
        instance._native_random_access_index = native.copy()
        return instance

    def configure_random_access_training_v1(
        self,
        *,
        sampler_contract: Mapping[str, Any],
        successor_transition_counts: Sequence[int],
        summary_fit_manifest: Mapping[str, Any],
        market_closure_authority: Mapping[str, Any],
        normalization_artifact: Mapping[str, Any],
        first_state_bridge_witness: Mapping[str, Any],
        random_access_m1_times: Sequence[Any],
        parent_m1_row_offset: int,
        expected_child_parquet_sha256: str,
        expected_state_view_source_sha256: str,
        expected_composite_normalization_sha256: str | None = None,
    ) -> None:
        """Bind immutable TRAIN artifacts before the first DataLoader read."""

        if self._manifest["split"] != "train" or self._random_access_train is not None:
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TRAIN_BINDING_INVALID")
        contract = require_random_access_sampler_contract(sampler_contract)
        counts = np.ascontiguousarray(successor_transition_counts, dtype="<i8")
        summary = dict(summary_fit_manifest)
        normalization = require_lifetime_summary_normalization(
            normalization_artifact,
            expected_sample_authority_sha256=summary.get(
                "summary_sample_authority", {}
            ).get("authority_sha256"),
        )
        population = len(self._rows)
        if (
            contract["split"] != "train"
            or contract["entry_pair_population"] != population
            or contract["source_lineage_sha256"] != summary.get("source_lineage_sha256")
            or counts.shape != (population,)
            or np.any(counts < 1)
            or hashlib.sha256(counts.tobytes()).hexdigest()
            != summary.get("successor_counts_sha256")
            or summary.get("schema_version")
            != "gx1_unified_exit_pilot_summary_fit_inputs_v1"
            or summary.get("decision") != "PASS"
            or summary.get("split") != "train"
            or summary.get("entry_pair_population") != population
            or summary.get("lifetime_summary_normalization") != normalization
            or summary.get("val_fit_rows") != 0
            or summary.get("test_fit_rows") != 0
            or summary.get("test_accessed") is not False
            or summary.get("manifest_sha256")
            != _canonical_sha256(
                {
                    key: value
                    for key, value in summary.items()
                    if key != "manifest_sha256"
                }
            )
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_SUMMARY_BINDING_INVALID")
        child_times = pd.DatetimeIndex(
            pd.to_datetime(random_access_m1_times, utc=True, errors="coerce")
        ).as_unit("ns")
        authority = require_market_closure_authority(
            market_closure_authority,
            expected_m1_source_sha256=summary["m1_source_sha256"],
            expected_m1_clock_sha256=m1_clock_sha256(child_times),
        )
        if (
            authority["artifact_sha256"] != summary["closure_authority_sha256"]
            or isinstance(parent_m1_row_offset, bool)
            or not isinstance(parent_m1_row_offset, int)
            or parent_m1_row_offset < 0
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_CLOSURE_BINDING_INVALID")
        parent_times = pd.DatetimeIndex(self._source._m1_times).as_unit("ns")
        parent_stop = parent_m1_row_offset + len(child_times)
        if (
            child_times.empty
            or parent_stop > len(parent_times)
            or not parent_times[parent_m1_row_offset:parent_stop].equals(child_times)
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_PARENT_CLOCK_INVALID")
        feature_times = pd.DatetimeIndex(self._source._m1_feature_times).as_unit("ns")
        feature_start = int(np.searchsorted(feature_times.asi8, child_times.asi8[0]))
        feature_stop = feature_start + len(child_times)
        if (
            feature_start < 0
            or feature_stop > len(feature_times)
            or not feature_times[feature_start:feature_stop].equals(child_times)
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_FEATURE_CLOCK_INVALID")
        source_features = self._source._m1_features
        signal = np.ascontiguousarray(
            source_features["signal"][feature_start:feature_stop], dtype="<f4"
        )
        ctx_cont = np.ascontiguousarray(
            source_features["ctx_cont"][feature_start:feature_stop], dtype="<f4"
        )
        ctx_cat = np.ascontiguousarray(
            source_features["ctx_cat"][feature_start:feature_stop], dtype="<i8"
        )
        if not len(signal) == len(ctx_cont) == len(ctx_cat) == len(child_times):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_FEATURE_SOURCE_INVALID")
        witness = dict(first_state_bridge_witness)
        claimed_witness = witness.get("witness_sha256")
        bindings = witness.get("bindings")
        if (
            witness.get("schema_version") != FIRST_STATE_BRIDGE_SCHEMA_VERSION
            or witness.get("decision") != "PASS"
            or witness.get("split") != "train"
            or witness.get("entry_row_count") != population
            or witness.get("test_accessed") is not False
            or claimed_witness
            != _canonical_sha256(
                {
                    key: value
                    for key, value in witness.items()
                    if key != "witness_sha256"
                }
            )
            or not isinstance(bindings, Mapping)
            or bindings.get("child_parquet") != expected_child_parquet_sha256
            or bindings.get("m1_source") != summary["m1_source_sha256"]
            or bindings.get("closure_authority") != authority["artifact_sha256"]
            or bindings.get("state_view_source") != expected_state_view_source_sha256
            or bindings.get("train_normalization")
            != (
                expected_composite_normalization_sha256
                if expected_composite_normalization_sha256 is not None
                else normalization["normalization_sha256"]
            )
            or len(witness.get("first_state_episode_binding_sha256_by_entry", ()))
            != population
            or len(witness.get("entry_fill_binding_sha256_by_entry", ())) != population
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_BRIDGE_BINDING_INVALID")
        starts = self._rows["entry_m1_start_row"].to_numpy(dtype=np.int64)
        child_starts = starts - parent_m1_row_offset
        if (
            np.any(child_starts < 479)
            or np.any(child_starts + counts >= len(child_times))
            or not np.array_equal(
                child_times.asi8[child_starts],
                pd.DatetimeIndex(self._rows["first_state_row_time"]).as_unit("ns").asi8,
            )
        ):
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_ENTRY_CLOCK_INVALID")
        price_fields = tuple(
            dict.fromkeys(
                (
                    *UNIFIED_EXIT_PATH_PRICE_FIELDS,
                    "bid_open",
                    "bid_high",
                    "bid_low",
                    "bid_close",
                    "ask_open",
                    "ask_high",
                    "ask_low",
                    "ask_close",
                    "volume",
                )
            )
        )
        prices = {
            name: np.ascontiguousarray(
                self._source._m1[name][parent_m1_row_offset:parent_stop], dtype="<f8"
            )
            for name in price_fields
        }
        self._random_access_train = {
            "sampler_contract": contract,
            "successor_counts": counts,
            "summary_fit_manifest_sha256": summary["manifest_sha256"],
            "normalization_artifact": normalization,
            "m1_source_sha256": summary["m1_source_sha256"],
            "market_closure_authority": authority,
            "m1_times": child_times,
            "m1_signal": signal,
            "m1_ctx_cont": ctx_cont,
            "m1_ctx_cat": ctx_cat,
            "parent_row_offset": parent_m1_row_offset,
            "child_entry_starts": child_starts,
            "prices": prices,
            "ranges": {
                "bid_high": _RangeExtrema(prices["bid_high"], maximum=True),
                "bid_low": _RangeExtrema(prices["bid_low"], maximum=False),
                "ask_low": _RangeExtrema(prices["ask_low"], maximum=False),
                "ask_high": _RangeExtrema(prices["ask_high"], maximum=True),
            },
            "first_state_bridge_witness": witness,
            "first_state_bridge_witness_sha256": claimed_witness,
            "state_view_source_sha256": expected_state_view_source_sha256,
            "child_parquet_sha256": expected_child_parquet_sha256,
        }
        self._prepare_random_access_epoch()

    def _prepare_random_access_epoch(self) -> None:
        binding = self._random_access_train
        if binding is None:
            return
        samples = schedule_random_access_epoch(
            sampler_contract=binding["sampler_contract"],
            epoch_index=self._epoch_index,
            successor_transition_count_by_entry=[
                int(value) for value in binding["successor_counts"]
            ],
        )
        anchors = schedule_random_access_entry_anchors(
            sampler_contract=binding["sampler_contract"], epoch_index=self._epoch_index
        )
        grouped: dict[int, list[dict[str, Any]]] = {}
        for sample in samples:
            grouped.setdefault(int(sample["entry_row_index"]), []).append(sample)
        binding["samples_by_entry"] = {
            entry: tuple(sorted(values, key=lambda item: item["sample_slot"]))
            for entry, values in grouped.items()
        }
        binding["anchors_by_entry"] = {
            int(anchor["entry_row_index"]): anchor for anchor in anchors
        }

    def random_access_training_bindings_v1(self) -> dict[str, Any]:
        binding = self._random_access_train
        if binding is None:
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TRAIN_NOT_CONFIGURED")
        return {
            "sampler_contract": binding["sampler_contract"],
            "normalization_artifact": binding["normalization_artifact"],
            "m1_source_sha256": binding["m1_source_sha256"],
            "market_closure_authority_sha256": binding["market_closure_authority"][
                "artifact_sha256"
            ],
            "economic_step_manifest_sha256": self._economic_manifest["manifest_sha256"],
            "economics_objective_contract_sha256": self._readiness[
                "economics_objective_contract"
            ]["contract_sha256"],
        }

    def random_access_selected_entry_rows_v1(self) -> tuple[int, ...]:
        binding = self._random_access_train
        if binding is None:
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TRAIN_NOT_CONFIGURED")
        anchors = binding["anchors_by_entry"]
        return tuple(
            entry
            for entry, _anchor in sorted(
                anchors.items(), key=lambda item: int(item[1]["entry_slot"])
            )
        )

    def _random_access_lifetime_summary(
        self, *, entry_start: int, side: int, state_index: int
    ) -> dict[str, Any]:
        binding = self._random_access_train
        if binding is None:
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TRAIN_NOT_CONFIGURED")
        stop = entry_start + state_index + 1
        prices, ranges = binding["prices"], binding["ranges"]
        entry_bid = float(prices["bid_open"][entry_start])
        entry_ask = float(prices["ask_open"][entry_start])
        current_row = stop - 1
        if side == 0:
            current = (
                (float(prices["bid_close"][current_row]) - entry_ask)
                / entry_ask
                * 10_000.0
            )
            peak_price, peak_row = ranges["bid_high"].query(entry_start, stop)
            trough_price, _ = ranges["bid_low"].query(entry_start, stop)
            mfe = max(0.0, (peak_price - entry_ask) / entry_ask * 10_000.0)
            mae = min(0.0, (trough_price - entry_ask) / entry_ask * 10_000.0)
        else:
            current = (
                (entry_bid - float(prices["ask_close"][current_row]))
                / entry_bid
                * 10_000.0
            )
            peak_price, peak_row = ranges["ask_low"].query(entry_start, stop)
            trough_price, _ = ranges["ask_high"].query(entry_start, stop)
            mfe = max(0.0, (entry_bid - peak_price) / entry_bid * 10_000.0)
            mae = min(0.0, (entry_bid - trough_price) / entry_bid * 10_000.0)
        bars = state_index + 1
        elapsed = int(
            (
                int(binding["m1_times"].asi8[current_row])
                + 60_000_000_000
                - int(binding["m1_times"].asi8[entry_start])
            )
            // 1_000_000_000
        )
        return build_lifetime_summary(
            side=("long", "short")[side],
            bars_in_trade=bars,
            elapsed_wall_clock_seconds=elapsed,
            current_executable_pnl_bps=current,
            cum_mfe_bps=mfe,
            cum_mae_bps=mae,
            bars_since_mfe_peak=(current_row - peak_row if mfe > 0.0 else bars),
        )

    def materialize_random_access_training_item_v1(
        self, entry_row_index: int, *, outer_batch_index: int
    ) -> dict[str, Any] | None:
        binding = self._random_access_train
        if binding is None:
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_TRAIN_NOT_CONFIGURED")
        samples = binding["samples_by_entry"].get(entry_row_index)
        anchor = binding["anchors_by_entry"].get(entry_row_index)
        if samples is None and anchor is None:
            return None
        if samples is None or anchor is None or entry_row_index not in self._rows.index:
            raise RuntimeError("UNIFIED_EXIT_RANDOM_ACCESS_SCHEDULE_PAIR_INVALID")
        from gx1.contracts.unified_exit_random_access_state_view_v1 import (
            materialize_random_access_state_view,
        )

        entry_start = int(binding["child_entry_starts"][entry_row_index])
        count = int(binding["successor_counts"][entry_row_index]) + 1
        row = self._rows.loc[entry_row_index]
        if isinstance(row, pd.DataFrame):
            raise RuntimeError("UNIFIED_EXIT_DATASET_V2_DUPLICATE_ENTRY_ROW")

        def path_detail(_side: int, start: int, stop: int) -> np.ndarray:
            absolute = slice(entry_start + start, entry_start + stop)
            price_values = np.column_stack(
                [
                    binding["prices"][name][absolute]
                    for name in UNIFIED_EXIT_PATH_PRICE_FIELDS
                ]
            )
            return unified_exit_path_tensor_from_values(
                price_values=price_values,
                volumes=binding["prices"]["volume"][absolute],
                bars_in_trade=stop,
                entry_bid=float(binding["prices"]["bid_open"][entry_start]),
                entry_ask=float(binding["prices"]["ask_open"][entry_start]),
            )

        def one_view(sample: Mapping[str, Any]) -> dict[str, Any]:
            return materialize_random_access_state_view(
                sampler_contract=binding["sampler_contract"],
                sample=sample,
                entry_row_index=entry_row_index,
                entry_m1_start_row=entry_start,
                side_lifecycle_state_counts=(count, count),
                side_economic_terminal=(
                    bool(row["long_economic_terminal"]),
                    bool(row["short_economic_terminal"]),
                ),
                m1_times=binding["m1_times"],
                m1_signal=binding["m1_signal"],
                m1_ctx_cont=binding["m1_ctx_cont"],
                m1_ctx_cat=binding["m1_ctx_cat"],
                m1_source_sha256=binding["m1_source_sha256"],
                market_closure_authority=binding["market_closure_authority"],
                path_detail_provider=path_detail,
                lifetime_summary_provider=lambda side, index: (
                    self._random_access_lifetime_summary(
                        entry_start=entry_start, side=side, state_index=index
                    )
                ),
                mtf_materializer=self._mtf_materializer,
                economic_step_provider=self._economic_provider,
                economic_step_manifest=self._economic_manifest,
                economics_objective_contract=self._readiness[
                    "economics_objective_contract"
                ],
            )

        witness = binding["first_state_bridge_witness"]
        return {
            "outer_batch_index": outer_batch_index,
            "entry_row_index": entry_row_index,
            "transitions": [
                {"sample": sample, "state_view": one_view(sample)} for sample in samples
            ],
            "anchor": {"sample": anchor, "state_view": one_view(anchor)},
            "entry_episode_binding_sha256": witness[
                "first_state_episode_binding_sha256_by_entry"
            ][entry_row_index],
            "entry_fill_binding_sha256": witness["entry_fill_binding_sha256_by_entry"][
                entry_row_index
            ],
            "first_state_bridge_witness_sha256": binding[
                "first_state_bridge_witness_sha256"
            ],
        }

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
        self._prepare_random_access_epoch()

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

        hold_stop = min(
            chunk_start + valid_count,
            int(row[f"{('long', 'short')[side_index]}_lifecycle_state_count"]) - 1,
        )
        economic_terminal = side_pointer["terminal_reason"] == "economic_terminal"
        fastpath = getattr(
            self._economic_provider, "materialize_training_projection", None
        )
        if fastpath is None:
            exit_slice = load_slice("exit_now", chunk_start, chunk_start + valid_count)
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
                compose_economic_step(step, contract=contract)
                for step in raw_hold_steps
            ]
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
                raise RuntimeError(
                    "UNIFIED_EXIT_DATASET_V2_ECONOMIC_STEP_EVENT_INVALID"
                )
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
            exit_stream_sha256 = exit_slice["slice_sha256"]
            hold_stream_sha256 = hold_slice["slice_sha256"]
        else:
            if not callable(fastpath):
                raise RuntimeError("UNIFIED_EXIT_ECONOMIC_PROJECTION_PROVIDER_INVALID")
            try:
                projection = require_economic_training_projection(
                    fastpath(
                        entry_row,
                        side_index,
                        chunk_start,
                        chunk_start + valid_count,
                        hold_stop,
                    ),
                    entry_row_index=entry_row,
                    side_index=side_index,
                    start_state_index=chunk_start,
                    stop_state_index=chunk_start + valid_count,
                    hold_stop_state_index=hold_stop,
                    economic_manifest=self._economic_manifest,
                )
            except (KeyError, FileNotFoundError, OSError) as exc:
                raise RuntimeError(
                    "UNIFIED_EXIT_DATASET_V2_ECONOMIC_STEPS_MISSING"
                ) from exc
            expected_exit_events = np.full(
                valid_count, _EXIT_NOW_EVENT_INDEX, dtype=np.uint8
            )
            if economic_terminal:
                expected_exit_events[-1] = _ECONOMIC_TERMINAL_EVENT_INDEX
            if not np.array_equal(
                projection["exit_event_kind_index"], expected_exit_events
            ):
                raise RuntimeError(
                    "UNIFIED_EXIT_DATASET_V2_ECONOMIC_STEP_EVENT_INVALID"
                )
            rewards = np.ascontiguousarray(
                projection["exit_reward_bps"], dtype=np.float32
            )
            hold_rewards = np.zeros(valid_count, dtype=np.float32)
            hold_rewards[: hold_stop - chunk_start] = np.asarray(
                projection["hold_reward_bps"], dtype=np.float32
            )
            exit_stream_sha256 = projection["exit_stream_sha256"]
            hold_stream_sha256 = projection["hold_stream_sha256"]
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
            "economic_exit_step_stream_sha256": exit_stream_sha256,
            "economic_hold_step_stream_sha256": hold_stream_sha256,
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
    "ECONOMIC_TRAINING_PROJECTION_SCHEMA_VERSION",
    "UnifiedExitDatasetAdapterV2",
    "require_economic_training_projection",
    "seal_economic_exit_step_manifest",
    "seal_economic_training_projection",
)
