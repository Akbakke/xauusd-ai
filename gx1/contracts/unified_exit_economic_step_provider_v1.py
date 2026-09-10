"""Lazy economic-step slices from one immutable PRETEST executable M1 tape."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.unified_exit_dataset_adapter_v2 import (
    ECONOMIC_EXIT_STEP_MANIFEST_SCHEMA_VERSION,
    seal_economic_exit_step_manifest,
)
from gx1.contracts.unified_exit_economics_objective_v2 import (
    ECONOMIC_STEP_SCHEMA_VERSION,
    SECONDS_PER_YEAR,
)
from gx1.contracts.unified_exit_fitted_q_v1 import (
    require_unified_exit_unbounded_training_readiness,
)
from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import (
    REQUIRED_COMPONENTS,
    canonical_sha256,
    file_sha256,
    require_economics_fact_manifest,
)


COST_POLICY_SCHEMA_VERSION = "gx1_unified_exit_cost_policy_v1"
COST_PARAMETER_AUTHORITY_SCHEMA_VERSION = "gx1_unified_exit_cost_parameter_authority_v1"
ECONOMIC_STEP_SLICE_SCHEMA_VERSION = "gx1_unified_exit_economic_step_slice_v1"
_SIDES = ("long", "short")


def seal_unified_exit_cost_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    observed = dict(value)
    if "policy_sha256" in observed:
        raise RuntimeError("UNIFIED_EXIT_COST_POLICY_ALREADY_SEALED")
    observed["policy_sha256"] = canonical_sha256(observed)
    return observed


def seal_cost_parameter_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    observed = dict(value)
    if "authority_sha256" in observed:
        raise RuntimeError("UNIFIED_EXIT_COST_PARAMETER_AUTHORITY_ALREADY_SEALED")
    observed["authority_sha256"] = canonical_sha256(observed)
    return observed


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"UNIFIED_EXIT_COST_POLICY_{label}_SHA256_INVALID")
    return value


def _finite_pair(value: Any, *, label: str, nonnegative: bool) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise RuntimeError(f"UNIFIED_EXIT_COST_POLICY_{label}_INVALID")
    try:
        pair = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"UNIFIED_EXIT_COST_POLICY_{label}_INVALID") from exc
    if not all(math.isfinite(item) for item in pair) or (
        nonnegative and any(item < 0.0 for item in pair)
    ):
        raise RuntimeError(f"UNIFIED_EXIT_COST_POLICY_{label}_INVALID")
    return pair


def _component(value: float, source_sha256: str) -> dict[str, Any]:
    return {
        "status": "COMPLETE",
        "value_bps": float(value),
        "source_artifact_sha256": source_sha256,
    }


class LazyUnifiedExitEconomicStepProviderV1:
    """Callable provider; shared M1 tape is O(M1), slices are O(requested states)."""

    def __init__(
        self,
        *,
        compact_rows: pd.DataFrame,
        compact_manifest: Mapping[str, Any],
        economics_readiness: Mapping[str, Any],
        cost_policy_path: Path,
    ) -> None:
        manifest = dict(compact_manifest)
        readiness = require_unified_exit_unbounded_training_readiness(
            economics_readiness, context="UNIFIED_EXIT_STEP_PROVIDER"
        )
        policy_path = cost_policy_path.expanduser().resolve()
        if not policy_path.is_file() or policy_path.is_symlink():
            raise RuntimeError("UNIFIED_EXIT_COST_POLICY_PATH_INVALID")
        try:
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("UNIFIED_EXIT_COST_POLICY_INVALID") from exc
        keys = {
            "schema_version",
            "decision",
            "instrument",
            "split",
            "coverage_start_utc",
            "coverage_end_utc",
            "pretest_only",
            "lifecycle_manifest_sha256",
            "economics_objective_contract_sha256",
            "m1_tape_path",
            "m1_tape_sha256",
            "m1_tape_manifest_path",
            "m1_tape_manifest_sha256",
            "economics_fact_manifest_path",
            "economics_fact_manifest_sha256",
            "economics_fact_manifest_identity_sha256",
            "cost_parameter_authority_path",
            "cost_parameter_authority_sha256",
            "cost_parameter_authority_identity_sha256",
            "entry_quote_field_by_side",
            "exit_quote_field_by_side",
            "spread_cost_handling",
            "commission_total_bps_by_side",
            "slippage_total_bps_by_side",
            "financing_annual_bps_by_side",
            "guaranteed_execution_fee_total_bps_by_side",
            "hold_risk_penalty_annual_bps_by_side",
            "risk_utility_formula",
            "risk_utility_is_cash_pnl",
            "component_artifact_sha256",
            "gap_source_manifest_sha256",
            "declared_market_closure_intervals",
            "same_capital_hurdle_running_cost_bps",
            "test_data_used",
            "policy_sha256",
        }
        if (
            not isinstance(policy, dict)
            or set(policy) != keys
            or policy["schema_version"] != COST_POLICY_SCHEMA_VERSION
            or policy["decision"] != "PASS"
            or policy["instrument"] != "XAU_USD"
            or policy["split"] != manifest.get("split")
            or policy["pretest_only"] is not True
            or policy["lifecycle_manifest_sha256"] != manifest.get("manifest_sha256")
            or policy["economics_objective_contract_sha256"]
            != readiness["economics_objective_contract"]["contract_sha256"]
            or policy["entry_quote_field_by_side"] != ["ask_open", "bid_open"]
            or policy["exit_quote_field_by_side"] != ["bid_close", "ask_close"]
            or policy["spread_cost_handling"]
            != "embedded_once_in_entry_and_exit_executable_quotes"
            or policy["same_capital_hurdle_running_cost_bps"] != 0.0
            or policy["test_data_used"] is not False
            or policy["policy_sha256"]
            != canonical_sha256(
                {k: v for k, v in policy.items() if k != "policy_sha256"}
            )
        ):
            raise RuntimeError("UNIFIED_EXIT_COST_POLICY_INVALID")
        coverage_start = pd.Timestamp(policy["coverage_start_utc"])
        coverage_end = pd.Timestamp(policy["coverage_end_utc"])
        if (
            pd.isna(coverage_start)
            or pd.isna(coverage_end)
            or coverage_start.tz is None
            or coverage_end.tz is None
            or coverage_start.utcoffset() != pd.Timedelta(0)
            or coverage_end.utcoffset() != pd.Timedelta(0)
            or coverage_end <= coverage_start
        ):
            raise RuntimeError("UNIFIED_EXIT_COST_POLICY_COVERAGE_INVALID")

        tape_path = Path(str(policy["m1_tape_path"] or ""))
        tape_manifest_path = Path(str(policy["m1_tape_manifest_path"] or ""))
        facts_path = Path(str(policy["economics_fact_manifest_path"] or ""))
        for path, expected in (
            (tape_path, policy["m1_tape_sha256"]),
            (tape_manifest_path, policy["m1_tape_manifest_sha256"]),
            (facts_path, policy["economics_fact_manifest_sha256"]),
        ):
            if not path.is_absolute() or not path.is_file() or path.is_symlink():
                raise RuntimeError("UNIFIED_EXIT_COST_POLICY_SOURCE_INVALID")
            if file_sha256(path) != expected:
                raise RuntimeError("UNIFIED_EXIT_COST_POLICY_SOURCE_INVALID")
        tape_manifest = json.loads(tape_manifest_path.read_text(encoding="utf-8"))
        if (
            tape_manifest.get("schema_version") != "gx1_direct_native_pretest_source_v2"
            or tape_manifest.get("instrument") != "XAU_USD"
            or tape_manifest.get("timeframe") != "M1"
            or tape_manifest.get("timestamp_semantics") != "bar_start_utc"
            or tape_manifest.get("quote_complete_m1") is not True
            or tape_manifest.get("test_accessed") is not False
            or tape_manifest.get("output_parquet") != str(tape_path)
            or tape_manifest.get("output_parquet_sha256") != policy["m1_tape_sha256"]
            or pd.Timestamp(tape_manifest.get("test_boundary_utc")) < coverage_end
        ):
            raise RuntimeError("UNIFIED_EXIT_COST_POLICY_TAPE_MANIFEST_INVALID")
        facts = json.loads(facts_path.read_text(encoding="utf-8"))
        checked_facts = require_economics_fact_manifest(
            facts,
            expected_coverage_start_utc=coverage_start,
            expected_coverage_end_utc=coverage_end,
        )
        if (
            checked_facts["manifest_sha256"]
            != policy["economics_fact_manifest_identity_sha256"]
        ):
            raise RuntimeError("UNIFIED_EXIT_COST_POLICY_FACT_BINDING_INVALID")
        bindings = checked_facts["component_artifacts"]
        artifact_hashes = policy["component_artifact_sha256"]
        if not isinstance(artifact_hashes, Mapping) or set(artifact_hashes) != set(
            REQUIRED_COMPONENTS
        ):
            raise RuntimeError("UNIFIED_EXIT_COST_POLICY_FACT_BINDING_INVALID")
        for component in REQUIRED_COMPONENTS:
            fact = json.loads(
                Path(bindings[component]["path"]).read_text(encoding="utf-8")
            )
            if fact["artifact_sha256"] != artifact_hashes[component]:
                raise RuntimeError("UNIFIED_EXIT_COST_POLICY_FACT_BINDING_INVALID")

        parameter_path = Path(str(policy["cost_parameter_authority_path"] or ""))
        if (
            not parameter_path.is_absolute()
            or not parameter_path.is_file()
            or parameter_path.is_symlink()
            or file_sha256(parameter_path) != policy["cost_parameter_authority_sha256"]
        ):
            raise RuntimeError("UNIFIED_EXIT_COST_PARAMETER_AUTHORITY_INVALID")
        try:
            parameter_authority = json.loads(parameter_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("UNIFIED_EXIT_COST_PARAMETER_AUTHORITY_INVALID") from exc
        parameter_keys = {
            "schema_version",
            "decision",
            "method",
            "split",
            "coverage_start_utc",
            "coverage_end_utc",
            "component_artifact_sha256",
            "commission_total_bps_by_side",
            "slippage_total_bps_by_side",
            "financing_annual_bps_by_side",
            "guaranteed_execution_fee_total_bps_by_side",
            "hold_risk_penalty_annual_bps_by_side",
            "risk_utility_formula",
            "risk_utility_is_cash_pnl",
            "source_method_receipt_sha256",
            "source_method_receipt_path",
            "validation_or_test_used",
            "test_data_used",
            "authority_sha256",
        }
        if (
            not isinstance(parameter_authority, dict)
            or set(parameter_authority) != parameter_keys
        ):
            raise RuntimeError("UNIFIED_EXIT_COST_PARAMETER_AUTHORITY_INVALID")
        declared_parameter_sha = parameter_authority.pop("authority_sha256")
        source_method_path = Path(
            str(parameter_authority["source_method_receipt_path"] or "")
        )
        if (
            parameter_authority["schema_version"]
            != COST_PARAMETER_AUTHORITY_SCHEMA_VERSION
            or parameter_authority["decision"] != "PASS"
            or parameter_authority["split"] != policy["split"]
            or parameter_authority["coverage_start_utc"] != policy["coverage_start_utc"]
            or parameter_authority["coverage_end_utc"] != policy["coverage_end_utc"]
            or parameter_authority["component_artifact_sha256"] != artifact_hashes
            or parameter_authority["risk_utility_formula"]
            != "annual_open_notional_penalty_bps*elapsed_wall_clock_seconds/seconds_per_year"
            or parameter_authority["risk_utility_is_cash_pnl"] is not False
            or parameter_authority["validation_or_test_used"] is not False
            or parameter_authority["test_data_used"] is not False
            or not isinstance(parameter_authority["method"], str)
            or not parameter_authority["method"]
            or not source_method_path.is_absolute()
            or not source_method_path.is_file()
            or source_method_path.is_symlink()
            or file_sha256(source_method_path)
            != parameter_authority["source_method_receipt_sha256"]
            or declared_parameter_sha != canonical_sha256(parameter_authority)
            or declared_parameter_sha
            != policy["cost_parameter_authority_identity_sha256"]
        ):
            raise RuntimeError("UNIFIED_EXIT_COST_PARAMETER_AUTHORITY_INVALID")
        _require_sha256(
            parameter_authority["source_method_receipt_sha256"],
            label="SOURCE_METHOD_RECEIPT",
        )

        commission = _finite_pair(
            policy["commission_total_bps_by_side"], label="COMMISSION", nonnegative=True
        )
        slippage = _finite_pair(
            policy["slippage_total_bps_by_side"], label="SLIPPAGE", nonnegative=True
        )
        financing = _finite_pair(
            policy["financing_annual_bps_by_side"], label="FINANCING", nonnegative=False
        )
        guaranteed_fee = _finite_pair(
            policy["guaranteed_execution_fee_total_bps_by_side"],
            label="GUARANTEED_FEE",
            nonnegative=True,
        )
        risk_penalty = _finite_pair(
            policy["hold_risk_penalty_annual_bps_by_side"],
            label="RISK_PENALTY",
            nonnegative=True,
        )
        authority_values = {
            "commission_total_bps_by_side": list(commission),
            "slippage_total_bps_by_side": list(slippage),
            "financing_annual_bps_by_side": list(financing),
            "guaranteed_execution_fee_total_bps_by_side": list(guaranteed_fee),
            "hold_risk_penalty_annual_bps_by_side": list(risk_penalty),
        }
        if (
            any(
                parameter_authority[key] != value
                for key, value in authority_values.items()
            )
            or policy["risk_utility_formula"]
            != parameter_authority["risk_utility_formula"]
            or policy["risk_utility_is_cash_pnl"] is not False
        ):
            raise RuntimeError("UNIFIED_EXIT_COST_POLICY_VALUE_NOT_SOURCE_BOUND")

        tape = pd.read_parquet(tape_path)
        required_columns = {"bid_open", "ask_open", "bid_close", "ask_close"}
        if "time" not in tape.columns or not required_columns.issubset(tape.columns):
            raise RuntimeError("UNIFIED_EXIT_COST_POLICY_TAPE_SCHEMA_INVALID")
        times = pd.DatetimeIndex(pd.to_datetime(tape["time"], utc=True)).as_unit("ns")
        prices = {
            name: np.asarray(tape[name], dtype=np.float64) for name in required_columns
        }
        if (
            len(tape) != tape_manifest.get("row_count")
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
            raise RuntimeError("UNIFIED_EXIT_COST_POLICY_TAPE_CLOCK_INVALID")
        compact = compact_rows.set_index("entry_row_index", drop=False)
        if (
            compact.empty
            or compact["m1_source_sha256"].ne(policy["m1_tape_sha256"]).any()
            or int(compact["entry_m1_start_row"].max()) >= len(tape)
        ):
            raise RuntimeError("UNIFIED_EXIT_COST_POLICY_LIFECYCLE_TAPE_MISMATCH")
        closures = policy["declared_market_closure_intervals"]
        if not isinstance(closures, list):
            raise RuntimeError("UNIFIED_EXIT_COST_POLICY_GAP_INVALID")
        closure_map = {}
        for item in closures:
            if not isinstance(item, Mapping) or set(item) != {
                "start_time_ns",
                "end_time_ns",
                "artifact_sha256",
            }:
                raise RuntimeError("UNIFIED_EXIT_COST_POLICY_GAP_INVALID")
            closure_map[(int(item["start_time_ns"]), int(item["end_time_ns"]))] = item[
                "artifact_sha256"
            ]
        self._rows = compact
        self._times_ns = np.asarray(times.asi8, dtype=np.int64)
        self._prices = prices
        self._commission = commission
        self._slippage = slippage
        self._financing = financing
        self._guaranteed_fee = guaranteed_fee
        self._risk_penalty = risk_penalty
        self._component_hashes = dict(artifact_hashes)
        self._capital_hurdle_sha = readiness["economics_objective_contract"][
            "capital_hurdle_artifact_sha256"
        ]
        self._policy = policy
        self._closure_map = closure_map
        self._source_manifest_sha256 = canonical_sha256(
            {
                "m1_tape_sha256": policy["m1_tape_sha256"],
                "m1_tape_manifest_sha256": policy["m1_tape_manifest_sha256"],
                "economics_fact_manifest_sha256": policy[
                    "economics_fact_manifest_sha256"
                ],
                "gap_source_manifest_sha256": policy["gap_source_manifest_sha256"],
            }
        )
        self.economic_exit_step_manifest = seal_economic_exit_step_manifest(
            {
                "schema_version": ECONOMIC_EXIT_STEP_MANIFEST_SCHEMA_VERSION,
                "split": policy["split"],
                "lifecycle_manifest_sha256": policy["lifecycle_manifest_sha256"],
                "economics_objective_contract_sha256": policy[
                    "economics_objective_contract_sha256"
                ],
                "economic_step_model_sha256": policy["policy_sha256"],
                "economic_step_source_manifest_sha256": self._source_manifest_sha256,
                "test_data_used": False,
            }
        )

    def __call__(
        self,
        entry_row_index: int,
        side_index: int,
        action: str,
        start_state_index: int,
        stop_state_index: int,
    ) -> dict[str, Any]:
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
        lifecycle_count = int(row[f"{_SIDES[side_index]}_lifecycle_state_count"])
        maximum_stop = lifecycle_count if action == "exit_now" else lifecycle_count - 1
        if stop_state_index > maximum_stop:
            raise RuntimeError("UNIFIED_EXIT_ECONOMIC_STEP_SLICE_REQUEST_INVALID")
        entry_row = int(row["entry_m1_start_row"])
        state_rows = range(entry_row + start_state_index, entry_row + stop_state_index)
        entry_price = self._prices[("ask_open", "bid_open")[side_index]][entry_row]
        steps = [
            self._step(
                entry_price=entry_price,
                state_row=state_row,
                side_index=side_index,
                action=action,
            )
            for state_row in state_rows
        ]
        envelope = {
            "schema_version": ECONOMIC_STEP_SLICE_SCHEMA_VERSION,
            "entry_row_index": entry_row_index,
            "side_index": side_index,
            "action": action,
            "start_state_index": start_state_index,
            "stop_state_index": stop_state_index,
            "steps": steps,
            "economic_step_model_sha256": self._policy["policy_sha256"],
            "economic_step_source_manifest_sha256": self._source_manifest_sha256,
        }
        envelope["slice_sha256"] = canonical_sha256(envelope)
        return envelope

    def _step(
        self, *, entry_price: float, state_row: int, side_index: int, action: str
    ) -> dict[str, Any]:
        decision_ns = int(self._times_ns[state_row] + 60_000_000_000)
        hashes = self._component_hashes
        if action == "exit_now":
            exit_price = self._prices[("bid_close", "ask_close")[side_index]][state_row]
            gross = (
                (exit_price - entry_price) / entry_price * 10_000.0
                if side_index == 0
                else (entry_price - exit_price) / entry_price * 10_000.0
            )
            return {
                "schema_version": ECONOMIC_STEP_SCHEMA_VERSION,
                "event_kind": "EXIT_NOW",
                "interval_start_time_ns": decision_ns,
                "interval_end_time_ns": decision_ns,
                "gross_price_cashflow": _component(gross, hashes["executable_bid_ask"]),
                "commission": _component(
                    self._commission[side_index], hashes["commission"]
                ),
                "execution_slippage": _component(
                    self._slippage[side_index], hashes["execution_slippage"]
                ),
                "financing_or_swap": _component(0.0, hashes["financing_or_swap"]),
                "guaranteed_execution_fee": _component(
                    self._guaranteed_fee[side_index], hashes["guaranteed_execution_fee"]
                ),
                "risk_utility_penalty": _component(0.0, self._policy["policy_sha256"]),
                "same_capital_hurdle_running_cost": _component(
                    0.0, self._capital_hurdle_sha
                ),
                "gap": {
                    "status": "COMPLETE",
                    "classification": "instantaneous_execution",
                    "source_manifest_sha256": self._policy[
                        "gap_source_manifest_sha256"
                    ],
                    "classification_artifact_sha256": self._policy[
                        "gap_source_manifest_sha256"
                    ],
                },
            }
        next_decision_ns = int(self._times_ns[state_row + 1] + 60_000_000_000)
        elapsed_seconds = (next_decision_ns - decision_ns) // 1_000_000_000
        if elapsed_seconds == 60:
            classification = "continuous_m1"
            gap_artifact = self._policy["gap_source_manifest_sha256"]
        elif elapsed_seconds > 60:
            classification = "declared_market_closure"
            gap_artifact = self._closure_map.get((decision_ns, next_decision_ns))
            if gap_artifact is None:
                raise RuntimeError("UNIFIED_EXIT_ECONOMIC_STEP_GAP_UNVERIFIED")
        else:
            raise RuntimeError("UNIFIED_EXIT_ECONOMIC_STEP_CLOCK_INVALID")
        scale = float(elapsed_seconds) / SECONDS_PER_YEAR
        return {
            "schema_version": ECONOMIC_STEP_SCHEMA_VERSION,
            "event_kind": "HOLD",
            "interval_start_time_ns": decision_ns,
            "interval_end_time_ns": next_decision_ns,
            "gross_price_cashflow": _component(0.0, hashes["executable_bid_ask"]),
            "commission": _component(0.0, hashes["commission"]),
            "execution_slippage": _component(0.0, hashes["execution_slippage"]),
            "financing_or_swap": _component(
                self._financing[side_index] * scale, hashes["financing_or_swap"]
            ),
            "guaranteed_execution_fee": _component(
                0.0, hashes["guaranteed_execution_fee"]
            ),
            "risk_utility_penalty": _component(
                self._risk_penalty[side_index] * scale, self._policy["policy_sha256"]
            ),
            "same_capital_hurdle_running_cost": _component(
                0.0, self._capital_hurdle_sha
            ),
            "gap": {
                "status": "COMPLETE",
                "classification": classification,
                "source_manifest_sha256": self._policy["gap_source_manifest_sha256"],
                "classification_artifact_sha256": gap_artifact,
            },
        }


__all__ = (
    "COST_PARAMETER_AUTHORITY_SCHEMA_VERSION",
    "COST_POLICY_SCHEMA_VERSION",
    "ECONOMIC_STEP_SLICE_SCHEMA_VERSION",
    "LazyUnifiedExitEconomicStepProviderV1",
    "seal_cost_parameter_authority",
    "seal_unified_exit_cost_policy",
)
