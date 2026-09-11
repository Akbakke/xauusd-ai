"""Production artifact factory for deterministic random-access Exit VAL rollout."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from gx1.contracts.unified_exit_dataset_adapter_v2 import _RangeExtrema
from gx1.contracts.unified_exit_lifetime_summary_v1 import build_lifetime_summary
from gx1.contracts.unified_exit_market_closure_authority_v1 import (
    closure_intervals_by_gap_after_row,
    m1_clock_sha256,
    require_market_closure_authority,
)
from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import file_sha256
from gx1.contracts.unified_exit_pilot_final_bindings_v1 import (
    build_split_sequence_binding,
    require_composite_normalization_binding,
    require_split_sequence_binding,
)
from gx1.contracts.unified_exit_pilot_normalization_v1 import (
    build_first_state_entry_bridge_witness,
    canonical_sha256,
)
from gx1.contracts.unified_exit_random_access_val_rollout_v1 import (
    RandomAccessValRolloutAdapterV1,
    VAL_ENTRY_COHORT_SIZE,
    build_random_access_val_rollout_contract,
)
from gx1.features.htf_features import MULTI_TF_TIMEFRAMES
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_PATH_PRICE_FIELDS,
    unified_exit_path_tensor_from_values,
)


VAL_FACTORY_SCHEMA_VERSION = "gx1_unified_exit_random_access_val_factory_v1"
_NS_PER_MINUTE = 60_000_000_000
_M1_HISTORY = 480
_PATH_TAIL = 512


def _read_json(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise RuntimeError(f"UNIFIED_EXIT_VAL_FACTORY_{label}_PATH_INVALID")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"UNIFIED_EXIT_VAL_FACTORY_{label}_INVALID") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"UNIFIED_EXIT_VAL_FACTORY_{label}_INVALID")
    return value


def _readonly(value: Any, dtype: str) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise RuntimeError("UNIFIED_EXIT_VAL_FACTORY_NONFINITE")
    array.setflags(write=False)
    return array


class RandomAccessValStateFactoryV1:
    """Own the exact VAL cohort and lazily materialize causal rolling states."""

    def __init__(
        self,
        *,
        entry_rows: pd.DataFrame,
        child_m1: pd.DataFrame,
        successor_transition_counts: Sequence[int],
        first_state_bridge: Mapping[str, Any],
        sequence_binding: Mapping[str, Any],
        composite_normalization: Mapping[str, Any],
        closure_authority: Mapping[str, Any],
        source_owner: Any,
        mtf_materializer: Callable[[np.ndarray], Mapping[str, np.ndarray]],
        economic_step_provider: Any,
        economic_step_manifest: Mapping[str, Any],
        economics_objective_contract: Mapping[str, Any],
        artifact_file_sha256: Mapping[str, str],
    ) -> None:
        if len(entry_rows) != VAL_ENTRY_COHORT_SIZE or len(child_m1) < _M1_HISTORY:
            raise RuntimeError("UNIFIED_EXIT_VAL_FACTORY_COHORT_INVALID")
        self.entry_rows = entry_rows.reset_index(drop=True)
        self.child_m1 = child_m1.reset_index(drop=True)
        self.times = pd.DatetimeIndex(
            pd.to_datetime(self.child_m1["time"], utc=True, errors="coerce")
        ).as_unit("ns")
        counts = np.ascontiguousarray(successor_transition_counts, dtype="<i8")
        if (
            self.times.hasnans
            or not self.times.is_unique
            or not self.times.is_monotonic_increasing
            or counts.shape != (VAL_ENTRY_COHORT_SIZE,)
            or np.any(counts < 1)
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_FACTORY_SOURCE_INVALID")
        sequence = require_split_sequence_binding(
            sequence_binding,
            expected_split="val",
            expected_entry_rows=VAL_ENTRY_COHORT_SIZE,
        )
        composite = require_composite_normalization_binding(composite_normalization)
        bridge = dict(first_state_bridge)
        bridge_core = {
            key: value for key, value in bridge.items() if key != "witness_sha256"
        }
        if (
            bridge.get("schema_version")
            != "gx1_unified_exit_first_state_entry_bridge_v1"
            or bridge.get("decision") != "PASS"
            or bridge.get("split") != "val"
            or bridge.get("entry_row_count") != VAL_ENTRY_COHORT_SIZE
            or bridge.get("test_accessed") is not False
            or bridge.get("witness_sha256") != canonical_sha256(bridge_core)
            or len(bridge.get("first_state_episode_binding_sha256_by_entry", ()))
            != VAL_ENTRY_COHORT_SIZE
            or len(bridge.get("entry_fill_binding_sha256_by_entry", ()))
            != VAL_ENTRY_COHORT_SIZE
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_FACTORY_BRIDGE_INVALID")
        closure = require_market_closure_authority(
            closure_authority,
            expected_m1_source_sha256=bridge["bindings"]["m1_source"],
            expected_m1_clock_sha256=m1_clock_sha256(self.times),
        )
        hashes = dict(artifact_file_sha256)
        required_hashes = {
            "entry_parquet",
            "entry_manifest",
            "child_m1",
            "child_m1_manifest",
            "successor_counts",
            "summary_manifest",
            "first_state_bridge",
            "split_sequence_binding",
            "composite_normalization",
            "closure_authority",
        }
        if (
            set(hashes) != required_hashes
            or sequence["bindings"]["child_parquet"] != hashes["entry_parquet"]
            or sequence["bindings"]["child_manifest_file"] != hashes["entry_manifest"]
            or sequence["bindings"]["m1_source"] != hashes["child_m1"]
            or sequence["bindings"]["m1_manifest_file"] != hashes["child_m1_manifest"]
            or sequence["bindings"]["closure_authority_file"]
            != hashes["closure_authority"]
            or sequence["bindings"]["closure_authority"] != closure["artifact_sha256"]
            or bridge["bindings"]["child_parquet"] != hashes["entry_parquet"]
            or bridge["bindings"]["m1_source"] != hashes["child_m1"]
            or bridge["bindings"]["closure_authority"] != closure["artifact_sha256"]
            or bridge["bindings"]["train_normalization"]
            != composite["composite_normalization_sha256"]
            or getattr(economic_step_provider, "economic_exit_step_manifest", None)
            != dict(economic_step_manifest)
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_FACTORY_EXACT_ARTIFACT_DRIFT")
        entry_times = pd.DatetimeIndex(
            pd.to_datetime(self.entry_rows["time"], utc=True, errors="coerce")
        ).as_unit("ns")
        first_state_times = entry_times.asi8 + 5 * _NS_PER_MINUTE
        starts = np.searchsorted(self.times.asi8, first_state_times).astype("<i8")
        if (
            entry_times.hasnans
            or not entry_times.is_unique
            or not entry_times.is_monotonic_increasing
            or np.any(starts < _M1_HISTORY - 1)
            or np.any(starts >= len(self.times))
            or not np.array_equal(self.times.asi8[starts], first_state_times)
            or np.any(starts + counts >= len(self.times))
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_FACTORY_ENTRY_CLOCK_INVALID")
        rebuilt_sequence = build_split_sequence_binding(
            split="val",
            entry_times=entry_times,
            m1_times=self.times,
            successor_transition_counts=counts,
            child_admission_file_sha256=sequence["bindings"]["child_admission_file"],
            child_admission_witness_sha256=sequence["bindings"][
                "child_admission_witness"
            ],
            child_parquet_sha256=sequence["bindings"]["child_parquet"],
            child_manifest_file_sha256=sequence["bindings"]["child_manifest_file"],
            child_manifest_contract_sha256=sequence["bindings"][
                "child_manifest_contract"
            ],
            m1_source_sha256=sequence["bindings"]["m1_source"],
            m1_manifest_file_sha256=sequence["bindings"]["m1_manifest_file"],
            closure_authority_file_sha256=sequence["bindings"][
                "closure_authority_file"
            ],
            closure_authority_sha256=sequence["bindings"]["closure_authority"],
        )
        rebuilt_bridge = build_first_state_entry_bridge_witness(
            split="val",
            entry_times=entry_times,
            m1_times=self.times,
            child_admission_sha256=bridge["bindings"]["child_admission"],
            child_parquet_sha256=bridge["bindings"]["child_parquet"],
            entry_sequence_audit_sha256=bridge["bindings"]["entry_sequence_audit"],
            m1_source_sha256=bridge["bindings"]["m1_source"],
            closure_authority_sha256=bridge["bindings"]["closure_authority"],
            state_view_source_sha256=bridge["bindings"]["state_view_source"],
            lifetime_summary_registry_sha256=bridge["bindings"][
                "lifetime_summary_registry"
            ],
            train_normalization_sha256=bridge["bindings"]["train_normalization"],
            m1_bid_open=self.child_m1["bid_open"],
            m1_ask_open=self.child_m1["ask_open"],
        )
        if rebuilt_sequence != sequence or rebuilt_bridge != bridge:
            raise RuntimeError("UNIFIED_EXIT_VAL_FACTORY_FINAL_BINDING_DRIFT")
        parent_times = pd.DatetimeIndex(source_owner._m1_times).as_unit("ns")
        parent_offset = int(np.searchsorted(parent_times.asi8, self.times.asi8[0]))
        parent_stop = parent_offset + len(self.times)
        feature_times = pd.DatetimeIndex(source_owner._m1_feature_times).as_unit("ns")
        feature_offset = int(np.searchsorted(feature_times.asi8, self.times.asi8[0]))
        feature_stop = feature_offset + len(self.times)
        if (
            parent_stop > len(parent_times)
            or not parent_times[parent_offset:parent_stop].equals(self.times)
            or feature_stop > len(feature_times)
            or not feature_times[feature_offset:feature_stop].equals(self.times)
            or getattr(economic_step_provider, "parent_m1_row_offset", None)
            != parent_offset
            or getattr(economic_step_provider, "state_m1_source_sha256", None)
            != bridge["bindings"]["m1_source"]
            or getattr(economic_step_provider, "market_closure_authority_sha256", None)
            != closure["artifact_sha256"]
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_FACTORY_PARENT_BINDING_INVALID")
        features = source_owner._m1_features
        self.signal = _readonly(features["signal"][feature_offset:feature_stop], "<f4")
        self.ctx_cont = _readonly(
            features["ctx_cont"][feature_offset:feature_stop], "<f4"
        )
        self.ctx_cat = _readonly(
            features["ctx_cat"][feature_offset:feature_stop], "<i8"
        )
        required_prices = tuple(
            dict.fromkeys(
                (
                    *UNIFIED_EXIT_PATH_PRICE_FIELDS,
                    "bid_open",
                    "ask_open",
                    "bid_high",
                    "bid_low",
                    "ask_high",
                    "ask_low",
                    "bid_close",
                    "ask_close",
                    "volume",
                )
            )
        )
        self.prices = {
            name: _readonly(source_owner._m1[name][parent_offset:parent_stop], "<f8")
            for name in required_prices
        }
        for name in (
            "bid_open",
            "ask_open",
            "bid_high",
            "bid_low",
            "ask_high",
            "ask_low",
            "bid_close",
            "ask_close",
        ):
            observed = np.asarray(self.child_m1[name], dtype="<f8")
            if not np.array_equal(observed, self.prices[name]):
                raise RuntimeError("UNIFIED_EXIT_VAL_FACTORY_CHILD_PRICE_DRIFT")
        self.ranges = {
            "bid_high": _RangeExtrema(self.prices["bid_high"], maximum=True),
            "bid_low": _RangeExtrema(self.prices["bid_low"], maximum=False),
            "ask_low": _RangeExtrema(self.prices["ask_low"], maximum=False),
            "ask_high": _RangeExtrema(self.prices["ask_high"], maximum=True),
        }
        self.starts = starts
        self.counts = counts
        self.bridge = bridge
        self.sequence = sequence
        self.composite_normalization = composite
        self.normalization = composite["lifetime_summary_normalization"]
        self.closure = closure
        self.closure_by_row = closure_intervals_by_gap_after_row(closure)
        self.mtf_materializer = mtf_materializer
        self.economic_step_provider = economic_step_provider
        self.economic_step_manifest = dict(economic_step_manifest)
        self.economics_objective_contract = dict(economics_objective_contract)
        self.parent_m1_row_offset = parent_offset
        self.artifact_file_sha256 = dict(artifact_file_sha256)
        self.entries = [
            {
                "entry_row_index": index,
                "entry_m1_start_row": int(starts[index]),
                "available_state_count": int(counts[index]) + 1,
                "entry_episode_binding_sha256": bridge[
                    "first_state_episode_binding_sha256_by_entry"
                ][index],
                "entry_fill_binding_sha256": bridge[
                    "entry_fill_binding_sha256_by_entry"
                ][index],
            }
            for index in range(VAL_ENTRY_COHORT_SIZE)
        ]
        self.factory_receipt = {
            "schema_version": VAL_FACTORY_SCHEMA_VERSION,
            "decision": "PASS",
            "split": "val",
            "entry_pair_count": VAL_ENTRY_COHORT_SIZE,
            "both_sides": True,
            "state_zero_has_prior_m1_rows": _M1_HISTORY - 1,
            "trade_path_tail_capacity": _PATH_TAIL,
            "capacity_is_terminal": False,
            "source_lineage_sha256": sequence["binding_sha256"],
            "first_state_bridge_sha256": bridge["witness_sha256"],
            "closure_authority_sha256": closure["artifact_sha256"],
            "composite_normalization_sha256": composite[
                "composite_normalization_sha256"
            ],
            "parent_m1_row_offset": parent_offset,
            "artifact_file_sha256": self.artifact_file_sha256,
            "test_accessed": False,
        }
        self.factory_receipt["factory_sha256"] = canonical_sha256(self.factory_receipt)

    @classmethod
    def from_artifacts(
        cls,
        *,
        entry_parquet_path: Path,
        entry_manifest_path: Path,
        child_m1_path: Path,
        child_m1_manifest_path: Path,
        successor_counts_path: Path,
        summary_manifest_path: Path,
        first_state_bridge_path: Path,
        split_sequence_binding_path: Path,
        composite_normalization_path: Path,
        closure_authority_path: Path,
        source_owner: Any,
        mtf_materializer: Callable[[np.ndarray], Mapping[str, np.ndarray]],
        economic_step_provider: Any,
        economic_step_manifest: Mapping[str, Any],
        economics_objective_contract: Mapping[str, Any],
    ) -> "RandomAccessValStateFactoryV1":
        paths = {
            "entry_parquet": entry_parquet_path.expanduser().resolve(),
            "entry_manifest": entry_manifest_path.expanduser().resolve(),
            "child_m1": child_m1_path.expanduser().resolve(),
            "child_m1_manifest": child_m1_manifest_path.expanduser().resolve(),
            "successor_counts": successor_counts_path.expanduser().resolve(),
            "summary_manifest": summary_manifest_path.expanduser().resolve(),
            "first_state_bridge": first_state_bridge_path.expanduser().resolve(),
            "split_sequence_binding": split_sequence_binding_path.expanduser().resolve(),
            "composite_normalization": composite_normalization_path.expanduser().resolve(),
            "closure_authority": closure_authority_path.expanduser().resolve(),
        }
        if any(not path.is_file() or path.is_symlink() for path in paths.values()):
            raise RuntimeError("UNIFIED_EXIT_VAL_FACTORY_ARTIFACT_PATH_INVALID")
        hashes = {name: file_sha256(path) for name, path in paths.items()}
        entry_manifest = _read_json(paths["entry_manifest"], "ENTRY_MANIFEST")
        child_manifest = _read_json(paths["child_m1_manifest"], "M1_MANIFEST")
        summary = _read_json(paths["summary_manifest"], "SUMMARY_MANIFEST")
        if (
            entry_manifest.get("split") != "val"
            or entry_manifest.get("decision") != "PASS"
            or entry_manifest.get("output_parquet_sha256") != hashes["entry_parquet"]
            or entry_manifest.get("test_accessed") is not False
            or child_manifest.get("split") != "val"
            or child_manifest.get("decision") != "PASS"
            or child_manifest.get("output_parquet_sha256") != hashes["child_m1"]
            or child_manifest.get("test_accessed") is not False
            or summary.get("split") != "val"
            or summary.get("decision") != "PASS"
            or summary.get("entry_pair_population") != VAL_ENTRY_COHORT_SIZE
            or summary.get("m1_source_sha256") != hashes["child_m1"]
            or summary.get("m1_manifest_sha256") != hashes["child_m1_manifest"]
            or summary.get("test_accessed") is not False
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_FACTORY_ARTIFACT_BINDING_INVALID")
        counts = np.load(paths["successor_counts"], allow_pickle=False)
        if hashlib.sha256(
            np.ascontiguousarray(counts, dtype="<i8").tobytes()
        ).hexdigest() != summary.get("successor_counts_sha256"):
            raise RuntimeError("UNIFIED_EXIT_VAL_FACTORY_COUNTS_INVALID")
        return cls(
            entry_rows=pd.read_parquet(paths["entry_parquet"]),
            child_m1=pd.read_parquet(paths["child_m1"]),
            successor_transition_counts=counts,
            first_state_bridge=_read_json(paths["first_state_bridge"], "BRIDGE"),
            sequence_binding=_read_json(paths["split_sequence_binding"], "SEQUENCE"),
            composite_normalization=_read_json(
                paths["composite_normalization"], "NORMALIZATION"
            ),
            closure_authority=_read_json(paths["closure_authority"], "CLOSURE"),
            source_owner=source_owner,
            mtf_materializer=mtf_materializer,
            economic_step_provider=economic_step_provider,
            economic_step_manifest=economic_step_manifest,
            economics_objective_contract=economics_objective_contract,
            artifact_file_sha256=hashes,
        )

    def _summary(self, entry_start: int, side: int, state_index: int) -> dict[str, Any]:
        stop = entry_start + state_index + 1
        current_row = stop - 1
        entry_bid = float(self.prices["bid_open"][entry_start])
        entry_ask = float(self.prices["ask_open"][entry_start])
        if side == 0:
            current = (
                (float(self.prices["bid_close"][current_row]) - entry_ask)
                / entry_ask
                * 10_000.0
            )
            peak, peak_row = self.ranges["bid_high"].query(entry_start, stop)
            trough, _ = self.ranges["bid_low"].query(entry_start, stop)
            mfe = max(0.0, (peak - entry_ask) / entry_ask * 10_000.0)
            mae = min(0.0, (trough - entry_ask) / entry_ask * 10_000.0)
        else:
            current = (
                (entry_bid - float(self.prices["ask_close"][current_row]))
                / entry_bid
                * 10_000.0
            )
            peak, peak_row = self.ranges["ask_low"].query(entry_start, stop)
            trough, _ = self.ranges["ask_high"].query(entry_start, stop)
            mfe = max(0.0, (entry_bid - peak) / entry_bid * 10_000.0)
            mae = min(0.0, (entry_bid - trough) / entry_bid * 10_000.0)
        elapsed = int(
            (
                self.times.asi8[current_row]
                + _NS_PER_MINUTE
                - self.times.asi8[entry_start]
            )
            // 1_000_000_000
        )
        return build_lifetime_summary(
            side=("long", "short")[side],
            bars_in_trade=state_index + 1,
            elapsed_wall_clock_seconds=elapsed,
            current_executable_pnl_bps=current,
            cum_mfe_bps=mfe,
            cum_mae_bps=mae,
            bars_since_mfe_peak=(
                current_row - peak_row if mfe > 0.0 else state_index + 1
            ),
        )

    def materialize_state(
        self, entry: Mapping[str, Any], state_index: int
    ) -> dict[str, Any]:
        entry_id = int(entry["entry_row_index"])
        if (
            entry_id < 0
            or entry_id >= VAL_ENTRY_COHORT_SIZE
            or dict(entry) != self.entries[entry_id]
            or state_index < 0
            or state_index >= entry["available_state_count"]
        ):
            raise RuntimeError("UNIFIED_EXIT_VAL_FACTORY_STATE_REQUEST_INVALID")
        start = int(self.starts[entry_id])
        row = start + state_index
        local_start = row - (_M1_HISTORY - 1)
        path_start = max(0, state_index - (_PATH_TAIL - 1))
        path_stop = state_index + 1
        absolute = slice(start + path_start, start + path_stop)
        price_values = np.column_stack(
            [self.prices[name][absolute] for name in UNIFIED_EXIT_PATH_PRICE_FIELDS]
        )
        path_one = unified_exit_path_tensor_from_values(
            price_values=price_values,
            volumes=self.prices["volume"][absolute],
            bars_in_trade=path_stop,
            entry_bid=float(self.prices["bid_open"][start]),
            entry_ask=float(self.prices["ask_open"][start]),
        )
        path = _readonly(np.stack([path_one, path_one]), "<f4")
        summaries = [self._summary(start, side, state_index) for side in range(2)]
        mtf_raw = self.mtf_materializer(np.asarray([self.times.asi8[row]], dtype="<i8"))
        mtf: dict[str, np.ndarray] = {}
        for tf in MULTI_TF_TIMEFRAMES:
            suffix = tf.lower()
            history = _readonly(mtf_raw[f"exit_mtf_history_{suffix}"], "<f4")
            history_time = _readonly(
                mtf_raw[f"exit_mtf_history_time_ns_{suffix}"], "<i8"
            )
            gather = _readonly(mtf_raw[f"exit_mtf_gather_{suffix}"], "<i8")
            if (
                history.ndim != 2
                or history_time.shape != (history.shape[0],)
                or gather.shape != (1,)
                or int(gather[0]) != history.shape[0] - 1
            ):
                raise RuntimeError("UNIFIED_EXIT_VAL_FACTORY_MTF_INVALID")
            mtf[f"exit_mtf_history_{suffix}"] = history
            mtf[f"exit_mtf_history_time_ns_{suffix}"] = history_time
            mtf[f"exit_mtf_gather_{suffix}"] = gather
        return {
            "state_index": state_index,
            "m1_row_index": row,
            "bar_start_time_ns": int(self.times.asi8[row]),
            "decision_time_ns": int(self.times.asi8[row] + _NS_PER_MINUTE),
            "m1_local_history_start_row": local_start,
            "m1_local_history_x": _readonly(self.signal[local_start : row + 1], "<f4"),
            "state_ctx_cont": _readonly(self.ctx_cont[row], "<f4"),
            "state_ctx_cat": _readonly(self.ctx_cat[row], "<i8"),
            "trade_path_start_state_index": path_start,
            "trade_path_length": path_stop - path_start,
            "trade_path_tail_x": path,
            "lifetime_summary_x": _readonly(
                np.stack([item["values"] for item in summaries]), "<f8"
            ),
            "lifetime_summary_sha256_by_side": [
                item["summary_sha256"] for item in summaries
            ],
            "mtf": mtf,
        }

    def materialize_transition(
        self, entry_row_index: int, state_index: int
    ) -> dict[str, Any]:
        entry = self.entries[entry_row_index]
        if state_index + 1 >= entry["available_state_count"]:
            raise RuntimeError("UNIFIED_EXIT_VAL_FACTORY_SUCCESSOR_UNAVAILABLE")
        current = self.materialize_state(entry, state_index)
        successor = self.materialize_state(entry, state_index + 1)
        delta = (
            successor["bar_start_time_ns"] - current["bar_start_time_ns"]
        ) // 1_000_000_000
        closure = None
        if delta != 60:
            closure = self.closure_by_row.get(current["m1_row_index"])
            if closure is None or closure["successor_across_gap_allowed"] is not True:
                raise RuntimeError("UNIFIED_EXIT_VAL_FACTORY_GAP_CENSORED")
        return {
            "current": current,
            "successor": successor,
            "transition_closure": closure,
        }

    def bind_rollout(
        self,
        *,
        entry_decision_representations: torch.Tensor,
        model_state_sha256: str,
        checkpoint_file_sha256: str,
        compute_guard_max_model_forwards: int,
        compute_guard_max_materialized_state_views: int,
        compute_guard_max_wall_seconds: float,
    ) -> tuple[dict[str, Any], RandomAccessValRolloutAdapterV1]:
        contract = build_random_access_val_rollout_contract(
            entries=self.entries,
            entry_decision_representations=entry_decision_representations,
            source_lineage_sha256=self.sequence["binding_sha256"],
            m1_source_sha256=self.bridge["bindings"]["m1_source"],
            m1_clock_sha256_value=m1_clock_sha256(self.times),
            m1_source_manifest_file_sha256=self.artifact_file_sha256[
                "child_m1_manifest"
            ],
            parent_m1_source_sha256=self.economic_step_provider.parent_m1_source_sha256,
            parent_m1_source_manifest_sha256=self.economic_step_provider.parent_m1_source_manifest_sha256,
            parent_m1_row_offset=self.parent_m1_row_offset,
            market_closure_authority_sha256=self.closure["artifact_sha256"],
            market_closure_authority_file_sha256=self.artifact_file_sha256[
                "closure_authority"
            ],
            economic_step_manifest_sha256=self.economic_step_manifest[
                "manifest_sha256"
            ],
            economics_objective_contract_sha256=self.economics_objective_contract[
                "contract_sha256"
            ],
            normalization_artifact=self.normalization,
            normalization_file_sha256=self.artifact_file_sha256[
                "composite_normalization"
            ],
            model_state_sha256=model_state_sha256,
            checkpoint_file_sha256=checkpoint_file_sha256,
            compute_guard_max_model_forwards=compute_guard_max_model_forwards,
            compute_guard_max_materialized_state_views=compute_guard_max_materialized_state_views,
            compute_guard_max_wall_seconds=compute_guard_max_wall_seconds,
        )
        adapter = RandomAccessValRolloutAdapterV1(
            contract=contract,
            entries=self.entries,
            m1_times=self.times,
            market_closure_authority=self.closure,
            economic_step_provider=self.economic_step_provider,
            economic_step_manifest=self.economic_step_manifest,
            economics_objective_contract=self.economics_objective_contract,
            normalization_artifact=self.normalization,
            state_provider=self.materialize_state,
        )
        return contract, adapter


__all__ = ["RandomAccessValStateFactoryV1", "VAL_FACTORY_SCHEMA_VERSION"]
