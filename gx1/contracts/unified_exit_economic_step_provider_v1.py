"""Lazy economic-step slices from immutable PRETEST quotes and cost authority."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.unified_exit_dataset_adapter_v2 import (
    ECONOMIC_EXIT_STEP_MANIFEST_SCHEMA_VERSION,
    ECONOMIC_TRAINING_PROJECTION_SCHEMA_VERSION,
    seal_economic_exit_step_manifest,
    seal_economic_training_projection,
)
from gx1.contracts.unified_exit_economics_objective_v2 import (
    ECONOMIC_STEP_SCHEMA_VERSION,
    MARK_TO_MARKET_REWARD_ACCOUNTING,
    LIQUIDATION_ADVANTAGE_REWARD_ACCOUNTING,
    MARK_TO_MARKET_STEP_SCHEMA_VERSION,
    SECONDS_PER_YEAR,
)
from gx1.contracts.unified_exit_fitted_q_v1 import (
    require_unified_exit_unbounded_training_readiness,
)
from gx1.contracts.unified_exit_market_closure_authority_v1 import (
    closure_intervals_by_gap_after_row,
    m1_clock_sha256,
    require_market_closure_authority,
)
from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import (
    REQUIRED_COMPONENTS,
    canonical_sha256,
    file_sha256,
)
from gx1.contracts.unified_exit_prospective_cost_policy_v1 import (
    require_cost_parameter_authority,
    require_prospective_cost_policy,
)

ECONOMIC_STEP_SLICE_SCHEMA_VERSION = "gx1_unified_exit_economic_step_slice_v1"
_SIDES = ("long", "short")
_TAPE_CACHE: dict[
    tuple[str, str], tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]
] = {}


def _component(value: float, source_sha256: str) -> dict[str, Any]:
    return {
        "status": "COMPLETE",
        "value_bps": float(value),
        "source_artifact_sha256": source_sha256,
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RuntimeError(f"UNIFIED_EXIT_STEP_PROVIDER_{label}_PATH_INVALID")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"UNIFIED_EXIT_STEP_PROVIDER_{label}_INVALID") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"UNIFIED_EXIT_STEP_PROVIDER_{label}_INVALID")
    return value


def _load_tape(
    *,
    tape_path: Path,
    tape_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    coverage_start: pd.Timestamp,
    coverage_end: pd.Timestamp,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    key = (str(tape_path), tape_sha256)
    cached = _TAPE_CACHE.get(key)
    if cached is not None:
        return cached
    if (
        file_sha256(tape_path) != tape_sha256
        or file_sha256(manifest_path) != manifest_sha256
    ):
        raise RuntimeError("UNIFIED_EXIT_STEP_PROVIDER_TAPE_HASH_INVALID")
    manifest = _read_json(manifest_path, "TAPE_MANIFEST")
    if (
        manifest.get("schema_version") != "gx1_direct_native_pretest_source_v2"
        or manifest.get("instrument") != "XAU_USD"
        or manifest.get("timeframe") != "M1"
        or manifest.get("timestamp_semantics") != "bar_start_utc"
        or manifest.get("quote_complete_m1") is not True
        or manifest.get("test_accessed") is not False
        or manifest.get("output_parquet") != str(tape_path)
        or manifest.get("output_parquet_sha256") != tape_sha256
        or pd.Timestamp(manifest.get("test_boundary_utc")) < coverage_end
    ):
        raise RuntimeError("UNIFIED_EXIT_STEP_PROVIDER_TAPE_MANIFEST_INVALID")
    tape = pd.read_parquet(
        tape_path,
        columns=["time", "bid_open", "ask_open", "bid_close", "ask_close"],
    )
    times = pd.DatetimeIndex(pd.to_datetime(tape["time"], utc=True)).as_unit("ns")
    prices = {
        name: np.asarray(tape[name], dtype=np.float64)
        for name in ("bid_open", "ask_open", "bid_close", "ask_close")
    }
    if (
        len(tape) != manifest.get("row_count")
        or times.empty
        or times.hasnans
        or not times.is_unique
        or not times.is_monotonic_increasing
        or times[0] > coverage_start
        or times[-1] < coverage_end - pd.Timedelta(minutes=1)
        or any(not np.isfinite(values).all() for values in prices.values())
        or any(np.any(values <= 0.0) for values in prices.values())
        or np.any(prices["bid_open"] > prices["ask_open"])
        or np.any(prices["bid_close"] > prices["ask_close"])
    ):
        raise RuntimeError("UNIFIED_EXIT_STEP_PROVIDER_TAPE_INVALID")
    result = np.asarray(times.asi8, dtype=np.int64), prices, manifest
    _TAPE_CACHE[key] = result
    return result


class LazyUnifiedExitEconomicStepProviderV1:
    """Return O(requested states) rewards from one process-shared O(M1) tape."""

    def __init__(
        self,
        *,
        compact_rows: pd.DataFrame,
        compact_manifest: Mapping[str, Any],
        economics_readiness: Mapping[str, Any],
        cost_parameter_authority_path: Path,
        market_closure_authority_path: Path | None = None,
        market_closure_authority_file_sha256: str | None = None,
        state_m1_source_path: Path | None = None,
        state_m1_source_manifest_path: Path | None = None,
        state_m1_source_file_sha256: str | None = None,
        state_m1_source_manifest_file_sha256: str | None = None,
        parent_m1_row_offset: int | None = None,
        common_successor_transition_counts: np.ndarray | None = None,
        expected_successor_counts_sha256: str | None = None,
    ) -> None:
        manifest = dict(compact_manifest)
        readiness = require_unified_exit_unbounded_training_readiness(
            economics_readiness, context="UNIFIED_EXIT_STEP_PROVIDER"
        )
        authority_path = cost_parameter_authority_path.expanduser().resolve()
        raw_authority = _read_json(authority_path, "COST_AUTHORITY")
        coverage_start = pd.Timestamp(raw_authority.get("coverage_start_utc"))
        coverage_end = pd.Timestamp(raw_authority.get("coverage_end_utc_exclusive"))
        authority = require_cost_parameter_authority(
            raw_authority,
            expected_coverage_start_utc=coverage_start,
            expected_coverage_end_utc=coverage_end,
            verify_local_sources=True,
        )
        policy_path = Path(authority["policy"]["path"])
        policy = require_prospective_cost_policy(
            _read_json(policy_path, "PROSPECTIVE_POLICY"),
            expected_coverage_start_utc=coverage_start,
            expected_coverage_end_utc=coverage_end,
            verify_local_sources=True,
        )
        if (
            readiness["policy_sha256"] != authority["authority_sha256"]
            or manifest.get("split") not in {"train", "val"}
            or manifest.get("test_accessed") is not False
            or policy["validation_policy_selection_permitted"] is not False
            or authority["future_train_only_risk_sweep_required"] is not True
            or authority["risk_utility_is_cash_pnl"] is not False
        ):
            raise RuntimeError("UNIFIED_EXIT_STEP_PROVIDER_POLICY_BINDING_INVALID")
        quote = policy["executable_bid_ask"]
        tape_path = Path(quote["parquet"]["path"])
        tape_manifest_path = Path(quote["manifest"]["path"])
        parent_times_ns, self._prices, tape_manifest = _load_tape(
            tape_path=tape_path,
            tape_sha256=quote["parquet"]["sha256"],
            manifest_path=tape_manifest_path,
            manifest_sha256=quote["manifest"]["sha256"],
            coverage_start=coverage_start,
            coverage_end=coverage_end,
        )
        compact = compact_rows.set_index("entry_row_index", drop=False)
        if (
            compact.empty
            or compact["m1_source_sha256"].ne(quote["parquet"]["sha256"]).any()
            or int(compact["entry_m1_start_row"].max()) >= len(parent_times_ns)
        ):
            raise RuntimeError("UNIFIED_EXIT_STEP_PROVIDER_LIFECYCLE_TAPE_MISMATCH")
        child_binding_values = (
            state_m1_source_path,
            state_m1_source_manifest_path,
            state_m1_source_file_sha256,
            state_m1_source_manifest_file_sha256,
            parent_m1_row_offset,
        )
        child_fields_present = tuple(
            value is not None for value in child_binding_values
        )
        child_bound = all(child_fields_present)
        if any(child_fields_present) and not child_bound:
            raise RuntimeError(
                "UNIFIED_EXIT_STEP_PROVIDER_STATE_SOURCE_BINDING_INVALID"
            )
        if child_bound:
            child_path = state_m1_source_path.expanduser().resolve()  # type: ignore[union-attr]
            child_manifest_path = state_m1_source_manifest_path.expanduser().resolve()  # type: ignore[union-attr]
            if (
                child_path.is_symlink()
                or child_manifest_path.is_symlink()
                or file_sha256(child_path) != state_m1_source_file_sha256
                or file_sha256(child_manifest_path)
                != state_m1_source_manifest_file_sha256
                or isinstance(parent_m1_row_offset, bool)
                or not isinstance(parent_m1_row_offset, int)
                or parent_m1_row_offset < 0
            ):
                raise RuntimeError(
                    "UNIFIED_EXIT_STEP_PROVIDER_STATE_SOURCE_BINDING_INVALID"
                )
            child_manifest = _read_json(child_manifest_path, "STATE_M1_MANIFEST")
            if (
                child_manifest.get("schema_version")
                != "gx1_unified_exit_pilot_m1_child_view_v1"
                or child_manifest.get("decision") != "PASS"
                or child_manifest.get("split") != manifest.get("split")
                or child_manifest.get("test_accessed") is not False
                or child_manifest.get("output_parquet") != str(child_path)
                or child_manifest.get("output_parquet_sha256")
                != state_m1_source_file_sha256
                or child_manifest.get("parent_m1_sha256") != quote["parquet"]["sha256"]
                or child_manifest.get("parent_m1_manifest_sha256")
                != quote["manifest"]["sha256"]
            ):
                raise RuntimeError(
                    "UNIFIED_EXIT_STEP_PROVIDER_STATE_SOURCE_BINDING_INVALID"
                )
            child_frame = pd.read_parquet(child_path, columns=["time"])
            child_times = pd.DatetimeIndex(
                pd.to_datetime(child_frame["time"], utc=True, errors="coerce")
            ).as_unit("ns")
            child_stop = parent_m1_row_offset + len(child_times)
            if (
                len(child_times) != child_manifest.get("row_count")
                or child_times.empty
                or child_times.hasnans
                or child_stop > len(parent_times_ns)
                or not np.array_equal(
                    child_times.asi8,
                    parent_times_ns[parent_m1_row_offset:child_stop],
                )
            ):
                raise RuntimeError(
                    "UNIFIED_EXIT_STEP_PROVIDER_STATE_SOURCE_BINDING_INVALID"
                )
            self._times_ns = np.asarray(child_times.asi8, dtype=np.int64)
            self._price_row_offset = parent_m1_row_offset
            self.state_m1_source_sha256 = state_m1_source_file_sha256
            self.state_m1_source_manifest_sha256 = state_m1_source_manifest_file_sha256
        else:
            self._times_ns = parent_times_ns
            self._price_row_offset = 0
            self.state_m1_source_sha256 = quote["parquet"]["sha256"]
            self.state_m1_source_manifest_sha256 = quote["manifest"]["sha256"]
        self.parent_m1_source_sha256 = quote["parquet"]["sha256"]
        self.parent_m1_source_manifest_sha256 = quote["manifest"]["sha256"]
        self.parent_m1_row_offset = self._price_row_offset
        if common_successor_transition_counts is not None:
            counts = np.ascontiguousarray(
                common_successor_transition_counts, dtype="<i8"
            )
            if (
                expected_successor_counts_sha256 is None
                or counts.shape != (len(compact),)
                or np.any(counts < 1)
                or hashlib.sha256(counts.tobytes()).hexdigest()
                != expected_successor_counts_sha256
            ):
                raise RuntimeError(
                    "UNIFIED_EXIT_STEP_PROVIDER_SUCCESSOR_COUNTS_INVALID"
                )
            for side in _SIDES:
                compact[f"{side}_lifecycle_state_count"] = counts + 1
            child_starts = (
                compact["entry_m1_start_row"].to_numpy(dtype=np.int64)
                - self._price_row_offset
            )
            if np.any(child_starts < 0) or np.any(
                child_starts + counts >= len(self._times_ns)
            ):
                raise RuntimeError(
                    "UNIFIED_EXIT_STEP_PROVIDER_SUCCESSOR_COUNTS_INVALID"
                )
        parameters = authority["parameters"]
        commission = float(parameters["commission"]["bps_per_execution"])
        slippage = float(parameters["execution_slippage"]["central_bps_per_execution"])
        financing = parameters["financing_or_swap"]
        risk = authority["hold_risk_penalty_annual_bps_by_side"]
        self._commission_total = (2.0 * commission,) * 2
        self._slippage_total = (2.0 * slippage,) * 2
        self._financing_cost_annual_bps = (
            float(financing["long_annual_cost_rate"]) * 10_000.0,
            float(financing["short_annual_cost_rate"]) * 10_000.0,
        )
        self._risk_penalty = (float(risk["long"]), float(risk["short"]))
        if (
            any(value < 0.0 for value in self._commission_total)
            or any(value < 0.0 for value in self._slippage_total)
            or any(value < 0.0 for value in self._financing_cost_annual_bps)
            or any(value < 0.0 for value in self._risk_penalty)
            or authority["risk_utility_formula"]
            != "annual_open_notional_penalty_bps*elapsed_wall_clock_seconds/seconds_per_year"
        ):
            raise RuntimeError("UNIFIED_EXIT_STEP_PROVIDER_PARAMETER_INVALID")
        self._rows = compact
        self._component_hashes = {
            name: authority["component_artifacts"][name]["artifact_sha256"]
            for name in REQUIRED_COMPONENTS
        }
        self._authority_sha = authority["authority_sha256"]
        objective = readiness["economics_objective_contract"]
        self._mark_to_market = objective.get("reward_accounting") in {MARK_TO_MARKET_REWARD_ACCOUNTING, LIQUIDATION_ADVANTAGE_REWARD_ACCOUNTING}
        self._annual_hurdle_rate = float(objective["annual_continuous_hurdle_rate"])
        self._initial_financing_bps = tuple(
            -annual * (60.0 / SECONDS_PER_YEAR) if self._mark_to_market else 0.0
            for annual in self._financing_cost_annual_bps
        )
        self._capital_hurdle_sha = readiness["economics_objective_contract"][
            "capital_hurdle_artifact_sha256"
        ]
        self._gap_source_sha = manifest["gap_classification_source_sha256"]
        self._closure_by_row: dict[int, dict[str, Any]] = {}
        closure_file_sha: str | None = None
        if (market_closure_authority_path is None) != (
            market_closure_authority_file_sha256 is None
        ):
            raise RuntimeError("UNIFIED_EXIT_STEP_PROVIDER_CLOSURE_BINDING_INVALID")
        if market_closure_authority_path is not None:
            closure_path = market_closure_authority_path.expanduser().resolve()
            closure_file_sha = file_sha256(closure_path)
            closure = require_market_closure_authority(
                _read_json(closure_path, "MARKET_CLOSURE_AUTHORITY"),
                expected_m1_source_sha256=self.state_m1_source_sha256,
                expected_m1_clock_sha256=m1_clock_sha256(
                    pd.DatetimeIndex(self._times_ns, tz="UTC")
                ),
            )
            if closure_file_sha != market_closure_authority_file_sha256 or (
                not child_bound and self._gap_source_sha != closure["artifact_sha256"]
            ):
                raise RuntimeError("UNIFIED_EXIT_STEP_PROVIDER_CLOSURE_BINDING_INVALID")
            self._closure_by_row = closure_intervals_by_gap_after_row(closure)
            self.market_closure_authority_sha256 = closure["artifact_sha256"]
        else:
            self.market_closure_authority_sha256 = self._gap_source_sha
        self._source_manifest_sha256 = canonical_sha256(
            {
                "cost_parameter_authority_file_sha256": file_sha256(authority_path),
                "cost_parameter_authority_sha256": self._authority_sha,
                "m1_tape_sha256": quote["parquet"]["sha256"],
                "m1_tape_manifest_sha256": quote["manifest"]["sha256"],
                "m1_tape_manifest_payload_sha256": tape_manifest[
                    "manifest_payload_sha256"
                ],
                "gap_source_manifest_sha256": self._gap_source_sha,
                "market_closure_authority_file_sha256": closure_file_sha,
                "state_m1_source_sha256": self.state_m1_source_sha256,
                "state_m1_source_manifest_sha256": self.state_m1_source_manifest_sha256,
                "parent_m1_row_offset": self._price_row_offset,
                "successor_counts_sha256": expected_successor_counts_sha256,
            }
        )
        self.economic_exit_step_manifest = seal_economic_exit_step_manifest(
            {
                "schema_version": ECONOMIC_EXIT_STEP_MANIFEST_SCHEMA_VERSION,
                "split": manifest["split"],
                "lifecycle_manifest_sha256": manifest["manifest_sha256"],
                "economics_objective_contract_sha256": readiness[
                    "economics_objective_contract"
                ]["contract_sha256"],
                "economic_step_model_sha256": self._authority_sha,
                "economic_step_source_manifest_sha256": self._source_manifest_sha256,
                "test_data_used": False,
            }
        )

    def _locate_request(
        self,
        *,
        entry_row_index: int,
        side_index: int,
        action: str,
        start_state_index: int,
        stop_state_index: int,
    ) -> tuple[pd.Series, int, int, float]:
        if (
            entry_row_index not in self._rows.index
            or side_index not in (0, 1)
            or action not in {"exit_now", "hold"}
            or isinstance(start_state_index, bool)
            or isinstance(stop_state_index, bool)
            or not isinstance(start_state_index, int)
            or not isinstance(stop_state_index, int)
            or start_state_index < 0
            or stop_state_index < start_state_index
        ):
            raise RuntimeError("UNIFIED_EXIT_ECONOMIC_STEP_SLICE_REQUEST_INVALID")
        row = self._rows.loc[entry_row_index]
        if isinstance(row, pd.DataFrame):
            raise RuntimeError("UNIFIED_EXIT_ECONOMIC_STEP_SLICE_REQUEST_INVALID")
        lifecycle_count = int(row[f"{_SIDES[side_index]}_lifecycle_state_count"])
        maximum_stop = lifecycle_count if action == "exit_now" else lifecycle_count - 1
        if stop_state_index > maximum_stop:
            raise RuntimeError("UNIFIED_EXIT_ECONOMIC_STEP_SLICE_REQUEST_INVALID")
        entry_row = int(row["entry_m1_start_row"]) - self._price_row_offset
        if entry_row < 0:
            raise RuntimeError("UNIFIED_EXIT_ECONOMIC_STEP_SLICE_REQUEST_INVALID")
        entry_price = float(
            self._prices[("ask_open", "bid_open")[side_index]][
                entry_row + self._price_row_offset
            ]
        )
        return row, lifecycle_count, entry_row, entry_price

    def __call__(
        self,
        entry_row_index: int,
        side_index: int,
        action: str,
        start_state_index: int,
        stop_state_index: int,
    ) -> dict[str, Any]:
        row, lifecycle_count, entry_row, entry_price = self._locate_request(
            entry_row_index=entry_row_index,
            side_index=side_index,
            action=action,
            start_state_index=start_state_index,
            stop_state_index=stop_state_index,
        )
        is_economic_terminal = bool(row[f"{_SIDES[side_index]}_economic_terminal"])
        steps = [
            self._step(
                entry_price=entry_price,
                state_row=entry_row + state_index,
                side_index=side_index,
                action=action,
                event_kind=(
                    "ECONOMIC_TERMINAL"
                    if action == "exit_now"
                    and is_economic_terminal
                    and state_index == lifecycle_count - 1
                    else "EXIT_NOW"
                    if action == "exit_now"
                    else "HOLD"
                ),
            )
            for state_index in range(start_state_index, stop_state_index)
        ]
        envelope = {
            "schema_version": ECONOMIC_STEP_SLICE_SCHEMA_VERSION,
            "entry_row_index": entry_row_index,
            "side_index": side_index,
            "action": action,
            "start_state_index": start_state_index,
            "stop_state_index": stop_state_index,
            "steps": steps,
            "economic_step_model_sha256": self._authority_sha,
            "economic_step_source_manifest_sha256": self._source_manifest_sha256,
        }
        envelope["slice_sha256"] = canonical_sha256(envelope)
        return envelope

    def materialize_selected_actions(
        self, requests: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Index immutable entry metadata once; retain scalar economic arithmetic."""
        if not requests:
            return []
        rows_by_entry = getattr(self, "_val_entry_metadata", None)
        if rows_by_entry is None:
            if not self._rows.index.is_unique:
                raise RuntimeError("UNIFIED_EXIT_ECONOMIC_STEP_SLICE_REQUEST_INVALID")
            # This provider owns one verified, fixed compact dataset. Tuples
            # retain its scalar dtypes without pandas selection per batch.
            columns = ("entry_m1_start_row", "long_lifecycle_state_count",
                       "short_lifecycle_state_count", "long_economic_terminal",
                       "short_economic_terminal")
            rows_by_entry = dict(zip(
                self._rows.index,
                self._rows.loc[:, list(columns)].itertuples(index=False, name=None),
            ))
            self._val_entry_metadata = rows_by_entry
        identifiers = [request["entry_row_index"] for request in requests]
        if any(type(index) is not int or index not in rows_by_entry for index in identifiers):
            raise RuntimeError("UNIFIED_EXIT_ECONOMIC_STEP_SLICE_REQUEST_INVALID")
        cache = getattr(self, "_val_hold_steps", None)
        if cache is None:
            cache = self._val_hold_steps = {}
        result = []
        for request in requests:
            entry = request["entry_row_index"]
            side = request["side_index"]
            action = request["action"]
            index = request["state_index"]
            if side not in (0, 1) or action not in {"hold", "exit_now"} or type(index) is not int or index < 0:
                raise RuntimeError("UNIFIED_EXIT_ECONOMIC_STEP_SLICE_REQUEST_INVALID")
            row = rows_by_entry[entry]
            count = int(row[1 + side])
            if index + 1 > (count if action == "exit_now" else count - 1):
                raise RuntimeError("UNIFIED_EXIT_ECONOMIC_STEP_SLICE_REQUEST_INVALID")
            entry_row = int(row[0]) - self._price_row_offset
            if entry_row < 0:
                raise RuntimeError("UNIFIED_EXIT_ECONOMIC_STEP_SLICE_REQUEST_INVALID")
            terminal = bool(row[3 + side])
            key = (entry_row + index, side)
            entry_price = float(self._prices[("ask_open", "bid_open")[side]][entry_row + self._price_row_offset])
            if action == "hold" and key in cache:
                step = cache[key]
                if self._mark_to_market:
                    # Costs can be shared across entries; liquidation value cannot.
                    step = {**step, **self._value_step_fields(entry_price, entry_row + index, side, action)}
            else:
                step = self._step(
                    entry_price=entry_price,
                    state_row=entry_row + index, side_index=side, action=action,
                    event_kind=("ECONOMIC_TERMINAL" if action == "exit_now" and terminal and index == count - 1
                                else "EXIT_NOW" if action == "exit_now" else "HOLD"),
                )
                if action == "hold":
                    # Share absolute-interval cash costs. The MTM field is
                    # recomputed for each entry on cache reuse above.
                    cache[key] = step
            envelope = {
                "schema_version": ECONOMIC_STEP_SLICE_SCHEMA_VERSION,
                "entry_row_index": entry, "side_index": side, "action": action,
                "start_state_index": index, "stop_state_index": index + 1,
                "steps": [step], "economic_step_model_sha256": self._authority_sha,
                "economic_step_source_manifest_sha256": self._source_manifest_sha256,
            }
            envelope["slice_sha256"] = canonical_sha256(envelope)
            result.append(envelope)
        return result

    def _hold_elapsed_seconds(self, state_rows: np.ndarray) -> np.ndarray:
        """Admit continuous or exact declared-closure successor transitions."""

        rows = np.ascontiguousarray(state_rows, dtype=np.int64)
        elapsed_ns = self._times_ns[rows + 1] - self._times_ns[rows]
        if np.any(elapsed_ns <= 0) or np.any(elapsed_ns % 1_000_000_000 != 0):
            raise RuntimeError("UNIFIED_EXIT_ECONOMIC_STEP_GAP_UNVERIFIED")
        noncontinuous = np.flatnonzero(elapsed_ns != 60_000_000_000)
        for position in noncontinuous.tolist():
            row = int(rows[position])
            record = self._closure_by_row.get(row)
            if (
                record is None
                or record["successor_across_gap_allowed"] is not True
                or pd.Timestamp(record["previous_bar_start_utc"]).value
                != int(self._times_ns[row])
                or pd.Timestamp(record["next_bar_start_utc"]).value
                != int(self._times_ns[row + 1])
            ):
                raise RuntimeError("UNIFIED_EXIT_ECONOMIC_STEP_GAP_UNVERIFIED")
        return np.ascontiguousarray(elapsed_ns // 1_000_000_000, dtype=np.int64)

    def materialize_training_projection(
        self,
        entry_row_index: int,
        side_index: int,
        start_state_index: int,
        stop_state_index: int,
        hold_stop_state_index: int,
    ) -> dict[str, Any]:
        """Vectorize the exact economic values consumed by the trainer."""

        row, lifecycle_count, entry_row, entry_price = self._locate_request(
            entry_row_index=entry_row_index,
            side_index=side_index,
            action="exit_now",
            start_state_index=start_state_index,
            stop_state_index=stop_state_index,
        )
        self._locate_request(
            entry_row_index=entry_row_index,
            side_index=side_index,
            action="hold",
            start_state_index=start_state_index,
            stop_state_index=hold_stop_state_index,
        )
        if hold_stop_state_index > stop_state_index:
            raise RuntimeError("UNIFIED_EXIT_ECONOMIC_STEP_SLICE_REQUEST_INVALID")

        state_rows = entry_row + np.arange(
            start_state_index, stop_state_index, dtype=np.int64
        )
        exit_price = self._prices[("bid_close", "ask_close")[side_index]][
            state_rows + self._price_row_offset
        ]
        if side_index == 0:
            gross = (exit_price - entry_price) / entry_price * 10_000.0
        else:
            gross = (entry_price - exit_price) / entry_price * 10_000.0
        exit_reward = gross + self._initial_financing_bps[side_index]
        exit_reward = exit_reward - self._commission_total[side_index]
        exit_reward = exit_reward - self._slippage_total[side_index]
        exit_reward = exit_reward - 0.0
        exit_reward = np.ascontiguousarray(exit_reward - 0.0, dtype="<f8")
        if not np.isfinite(exit_reward).all():
            raise RuntimeError("UNIFIED_EXIT_ECONOMICS_STEP_RESULT_NONFINITE")
        exit_events = np.full(exit_reward.shape, 0, dtype="u1")
        if (
            exit_events.size
            and bool(row[f"{_SIDES[side_index]}_economic_terminal"])
            and stop_state_index == lifecycle_count
        ):
            exit_events[-1] = 2

        hold_rows = entry_row + np.arange(
            start_state_index, hold_stop_state_index, dtype=np.int64
        )
        elapsed_seconds = self._hold_elapsed_seconds(hold_rows)
        scale = elapsed_seconds.astype(np.float64) / SECONDS_PER_YEAR
        financing = -self._financing_cost_annual_bps[side_index] * scale
        risk_penalty = self._risk_penalty[side_index] * scale
        hold_reward = 0.0 + financing
        hold_reward = hold_reward - 0.0
        hold_reward = hold_reward - 0.0
        hold_reward = hold_reward - 0.0
        hold_reward = np.ascontiguousarray(hold_reward - risk_penalty, dtype="<f8")
        if self._mark_to_market:
            next_value = self._liquidation_value_bps(entry_price, hold_rows + 1, side_index)
            # Use the scalar owner's math.exp arithmetic for byte-identical targets.
            durations, inverse = np.unique(elapsed_seconds, return_inverse=True)
            gamma = np.asarray([
                math.exp(-self._annual_hurdle_rate * int(seconds) / SECONDS_PER_YEAR)
                for seconds in durations
            ], dtype=np.float64)[inverse]
            hold_reward = np.ascontiguousarray(hold_reward + (1.0 - gamma) * next_value, dtype="<f8")
        if not np.isfinite(hold_reward).all():
            raise RuntimeError("UNIFIED_EXIT_ECONOMICS_STEP_RESULT_NONFINITE")
        hold_events = np.full(hold_reward.shape, 1, dtype="u1")
        return seal_economic_training_projection(
            {
                "schema_version": ECONOMIC_TRAINING_PROJECTION_SCHEMA_VERSION,
                "entry_row_index": entry_row_index,
                "side_index": side_index,
                "start_state_index": start_state_index,
                "stop_state_index": stop_state_index,
                "hold_stop_state_index": hold_stop_state_index,
                "exit_event_kind_index": exit_events,
                "exit_reward_bps": exit_reward,
                "hold_event_kind_index": hold_events,
                "hold_reward_bps": hold_reward,
                "economic_step_model_sha256": self._authority_sha,
                "economic_step_source_manifest_sha256": (self._source_manifest_sha256),
            }
        )

    def _liquidation_value_bps(
        self, entry_price: float, state_rows: int | np.ndarray, side_index: int,
    ) -> float | np.ndarray:
        """Executable value; earlier HOLD financing is already in the cash ledger."""
        prices = self._prices[("bid_close", "ask_close")[side_index]][state_rows + self._price_row_offset]
        gross = ((prices - entry_price) if side_index == 0 else (entry_price - prices)) / entry_price * 10_000.0
        value = gross + self._initial_financing_bps[side_index]
        return value - self._commission_total[side_index] - self._slippage_total[side_index]

    def _value_step_fields(
        self, entry_price: float, state_row: int, side_index: int, action: str,
    ) -> dict[str, Any]:
        if not self._mark_to_market:
            return {}
        value = (self._liquidation_value_bps(entry_price, state_row + 1, side_index)
                 if action == "hold" else 0.0)
        return {
            "schema_version": MARK_TO_MARKET_STEP_SCHEMA_VERSION,
            "successor_liquidation_value": _component(value, self._source_manifest_sha256),
        }

    def _step(
        self,
        *,
        entry_price: float,
        state_row: int,
        side_index: int,
        action: str,
        event_kind: str,
    ) -> dict[str, Any]:
        decision_ns = int(self._times_ns[state_row] + 60_000_000_000)
        hashes = self._component_hashes
        if action == "exit_now":
            exit_price = self._prices[("bid_close", "ask_close")[side_index]][
                state_row + self._price_row_offset
            ]
            gross = (
                (exit_price - entry_price) / entry_price * 10_000.0
                if side_index == 0
                else (entry_price - exit_price) / entry_price * 10_000.0
            )
            return {
                "schema_version": ECONOMIC_STEP_SCHEMA_VERSION,
                **self._value_step_fields(entry_price, state_row, side_index, action),
                "event_kind": event_kind,
                "interval_start_time_ns": decision_ns,
                "interval_end_time_ns": decision_ns,
                "gross_price_cashflow": _component(gross, hashes["executable_bid_ask"]),
                "commission": _component(
                    self._commission_total[side_index], hashes["commission"]
                ),
                "execution_slippage": _component(
                    self._slippage_total[side_index], hashes["execution_slippage"]
                ),
                "financing_or_swap": _component(self._initial_financing_bps[side_index], hashes["financing_or_swap"]),
                "guaranteed_execution_fee": _component(
                    0.0, hashes["guaranteed_execution_fee"]
                ),
                "risk_utility_penalty": _component(0.0, self._authority_sha),
                "same_capital_hurdle_running_cost": _component(
                    0.0, self._capital_hurdle_sha
                ),
                "gap": {
                    "status": "COMPLETE",
                    "classification": "instantaneous_execution",
                    "source_manifest_sha256": self._gap_source_sha,
                    "classification_artifact_sha256": self._gap_source_sha,
                },
            }
        elapsed_seconds = int(
            self._hold_elapsed_seconds(np.asarray([state_row], dtype=np.int64))[0]
        )
        next_decision_ns = decision_ns + elapsed_seconds * 1_000_000_000
        scale = float(elapsed_seconds) / SECONDS_PER_YEAR
        gap_record = self._closure_by_row.get(state_row)
        gap_classification = (
            "continuous_m1" if elapsed_seconds == 60 else "declared_market_closure"
        )
        gap_artifact_sha = (
            self._gap_source_sha
            if gap_record is None
            else str(gap_record["interval_sha256"])
        )
        return {
            "schema_version": ECONOMIC_STEP_SCHEMA_VERSION,
            **self._value_step_fields(entry_price, state_row, side_index, action),
            "event_kind": event_kind,
            "interval_start_time_ns": decision_ns,
            "interval_end_time_ns": next_decision_ns,
            "gross_price_cashflow": _component(0.0, hashes["executable_bid_ask"]),
            "commission": _component(0.0, hashes["commission"]),
            "execution_slippage": _component(0.0, hashes["execution_slippage"]),
            "financing_or_swap": _component(
                -self._financing_cost_annual_bps[side_index] * scale,
                hashes["financing_or_swap"],
            ),
            "guaranteed_execution_fee": _component(
                0.0, hashes["guaranteed_execution_fee"]
            ),
            "risk_utility_penalty": _component(
                self._risk_penalty[side_index] * scale, self._authority_sha
            ),
            "same_capital_hurdle_running_cost": _component(
                0.0, self._capital_hurdle_sha
            ),
            "gap": {
                "status": "COMPLETE",
                "classification": gap_classification,
                "source_manifest_sha256": self._gap_source_sha,
                "classification_artifact_sha256": gap_artifact_sha,
            },
        }


__all__ = (
    "ECONOMIC_STEP_SLICE_SCHEMA_VERSION",
    "LazyUnifiedExitEconomicStepProviderV1",
)
