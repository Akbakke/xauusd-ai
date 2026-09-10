"""Fail-closed prospective cost policy and economics fact bundle."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.contracts.unified_exit_broker_evidence_v1 import require_unified_exit_broker_evidence_v1
from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import (
    REQUIRED_COMPONENTS,
    canonical_sha256,
    file_sha256,
    require_economics_fact_manifest,
)

PROSPECTIVE_COST_POLICY_SCHEMA_VERSION = "gx1_unified_exit_prospective_cost_policy_v1"
COST_METHOD_RECEIPT_SCHEMA_VERSION = "gx1_unified_exit_cost_method_receipt_v1"
COST_PARAMETER_AUTHORITY_SCHEMA_VERSION = "gx1_unified_exit_cost_parameter_authority_v1"
POLICY_DECISION = "PREREGISTERED_PROSPECTIVE_POLICY_NOT_HISTORICAL_TRUTH"
AUTHORITY_DECISION = "PASS_POLICY_PARAMETERS_COMPLETE_NOT_ECONOMICS_PASS"
SECONDS_PER_YEAR = 31_557_600.0
_SHA = re.compile(r"[0-9a-f]{64}")


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise RuntimeError(f"PROSPECTIVE_COST_{label}_SHA_INVALID")
    return value


def _finite(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"PROSPECTIVE_COST_{label}_INVALID")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"PROSPECTIVE_COST_{label}_INVALID") from exc
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        raise RuntimeError(f"PROSPECTIVE_COST_{label}_INVALID")
    return number


def _utc(value: Any, label: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:
        raise RuntimeError(f"PROSPECTIVE_COST_{label}_TIME_INVALID") from exc
    if pd.isna(ts) or ts.tz is None or ts.utcoffset() != pd.Timedelta(0):
        raise RuntimeError(f"PROSPECTIVE_COST_{label}_TIME_INVALID")
    return ts.as_unit("ns")


def _source(value: Any, label: str, verify: bool) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise RuntimeError(f"PROSPECTIVE_COST_{label}_BINDING_INVALID")
    path = Path(str(value["path"] or ""))
    digest = _sha(value["sha256"], label)
    if not path.is_absolute() or path.is_symlink():
        raise RuntimeError(f"PROSPECTIVE_COST_{label}_BINDING_INVALID")
    if verify and (not path.is_file() or file_sha256(path) != digest):
        raise RuntimeError(f"PROSPECTIVE_COST_{label}_SOURCE_INVALID")
    return {"path": str(path), "sha256": digest}


def _artifact(value: Any, label: str, key: str, verify: bool) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "file_sha256", key}:
        raise RuntimeError(f"PROSPECTIVE_COST_{label}_BINDING_INVALID")
    path = Path(str(value["path"] or ""))
    file_digest = _sha(value["file_sha256"], f"{label}_FILE")
    artifact_digest = _sha(value[key], f"{label}_ARTIFACT")
    if not path.is_absolute() or path.is_symlink():
        raise RuntimeError(f"PROSPECTIVE_COST_{label}_BINDING_INVALID")
    if verify and (not path.is_file() or file_sha256(path) != file_digest):
        raise RuntimeError(f"PROSPECTIVE_COST_{label}_SOURCE_INVALID")
    return {"path": str(path), "file_sha256": file_digest, key: artifact_digest}


def _seal(value: Mapping[str, Any], key: str, error: str) -> dict[str, Any]:
    result = dict(value)
    if key in result:
        raise RuntimeError(error)
    result[key] = canonical_sha256(result)
    return result


def seal_prospective_cost_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    return _seal(value, "artifact_sha256", "PROSPECTIVE_COST_POLICY_ALREADY_SEALED")


def seal_cost_method_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    return _seal(value, "receipt_sha256", "PROSPECTIVE_COST_METHOD_ALREADY_SEALED")


def seal_cost_parameter_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    return _seal(value, "authority_sha256", "PROSPECTIVE_COST_AUTHORITY_ALREADY_SEALED")


def require_prospective_cost_policy(
    value: Mapping[str, Any], *, expected_coverage_start_utc: Any,
    expected_coverage_end_utc: Any, verify_local_sources: bool = True,
) -> dict[str, Any]:
    keys = {"schema_version", "decision", "policy_scope", "historical_cost_truth_qualified",
            "economics_pass_claimed", "preregistered_at_utc", "validation_policy_selection_permitted",
            "broker_evidence", "coverage", "executable_bid_ask", "commission", "latency_slippage",
            "financing_or_swap", "guaranteed_execution_fee", "hold_risk_utility",
            "future_train_only_risk_sweep_required", "test_data_used", "artifact_sha256"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeError("PROSPECTIVE_COST_POLICY_SCHEMA_INVALID")
    observed = dict(value)
    declared = _sha(observed.pop("artifact_sha256"), "POLICY")
    if declared != canonical_sha256(observed):
        raise RuntimeError("PROSPECTIVE_COST_POLICY_HASH_INVALID")
    start, end = _utc(expected_coverage_start_utc, "START"), _utc(expected_coverage_end_utc, "END")
    _utc(observed["preregistered_at_utc"], "PREREGISTERED")
    if (observed["schema_version"] != PROSPECTIVE_COST_POLICY_SCHEMA_VERSION
            or observed["decision"] != POLICY_DECISION
            or observed["policy_scope"] != "prospective_one_year_train_and_fixed_val_sensitivity"
            or observed["historical_cost_truth_qualified"] is not False
            or observed["economics_pass_claimed"] is not False
            or observed["validation_policy_selection_permitted"] is not False
            or observed["future_train_only_risk_sweep_required"] is not True
            or observed["test_data_used"] is not False
            or observed["coverage"] != {"start_utc": start.isoformat(), "end_utc_exclusive": end.isoformat(), "test_data_used": False}):
        raise RuntimeError("PROSPECTIVE_COST_POLICY_HEADER_INVALID")

    broker_binding = _artifact(observed["broker_evidence"], "BROKER", "artifact_sha256", verify_local_sources)
    broker_path = Path(broker_binding["path"])
    if not broker_path.is_file():
        raise RuntimeError("PROSPECTIVE_COST_BROKER_SOURCE_INVALID")
    broker = require_unified_exit_broker_evidence_v1(
        json.loads(broker_path.read_text(encoding="utf-8")),
        verify_local_sources=verify_local_sources,
    )
    if broker["artifact_sha256"] != broker_binding["artifact_sha256"]:
        raise RuntimeError("PROSPECTIVE_COST_BROKER_HASH_INVALID")

    quotes, executable = broker["executable_quote_source"], observed["executable_bid_ask"]
    quote_keys = {"pricing_method", "parquet", "manifest", "manifest_schema_version", "row_count",
                  "time_min_utc", "time_max_utc", "quote_complete_m1", "test_accessed"}
    if (not isinstance(executable, Mapping) or set(executable) != quote_keys
            or executable["pricing_method"] != "side_correct_executable_bid_ask_from_bound_m1_tape"
            or any(executable[k] != quotes[k] for k in quote_keys - {"pricing_method"})
            or _utc(executable["time_min_utc"], "QUOTE_MIN") > start
            or _utc(executable["time_max_utc"], "QUOTE_MAX") < end - pd.Timedelta(minutes=1)):
        raise RuntimeError("PROSPECTIVE_COST_EXECUTABLE_QUOTES_INVALID")
    _source(executable["parquet"], "QUOTE_PARQUET", verify_local_sources)
    _source(executable["manifest"], "QUOTE_MANIFEST", verify_local_sources)

    execution = broker["execution_observations"]
    account = broker["current_prospective_terms"]["account"]
    commission = observed["commission"]
    commission_expected = {
        "bps_per_execution": 0.0,
        "zero_basis": "258_of_258_pre_cutoff_fills_zero_plus_current_account_revalidation",
        "pre_cutoff_fill_count": 258,
        "pre_cutoff_commission_present_count": 258,
        "pre_cutoff_commission_nonzero_count": 0,
        "current_revalidation_at_utc": broker["generated_at_utc"],
        "current_account_snapshot_sha256": account["sanitized_snapshot_sha256"],
        "revalidate_before_each_run": True,
        "drift_action": "FAIL_CLOSED",
        "implicit_zero_used": False,
    }
    if (commission != commission_expected or execution["cutoff_fill_count"] != 258
            or execution["commission_present_count"] != 258
            or execution["commission_nonzero_count"] != 0
            or float(account["lifetime_commission_account_units"]) != 0.0):
        raise RuntimeError("PROSPECTIVE_COST_COMMISSION_INVALID")

    slippage_expected = {
        "central_bps_per_execution": 2.0,
        "direction": "adverse",
        "application": "each_market_entry_and_each_trade_close",
        "separate_from_bid_ask_spread": True,
        "source_status": "UNKNOWN_NO_PRE_CUTOFF_DECISION_QUOTE_TO_FILL_CLOCK_BINDING",
        "parameter_origin": "explicit_conservative_nonfitted_preregistration",
        "val_sensitivity_scenarios": [
            {"name": "low", "bps_per_execution": 1.0},
            {"name": "central", "bps_per_execution": 2.0},
            {"name": "high", "bps_per_execution": 4.0},
        ],
        "val_refit_permitted": False,
        "implicit_zero_used": False,
    }
    if (observed["latency_slippage"] != slippage_expected
            or execution["latency_slippage_status"] != slippage_expected["source_status"]):
        raise RuntimeError("PROSPECTIVE_COST_SLIPPAGE_INVALID")

    instrument = broker["current_prospective_terms"]["instrument"]
    financing_expected = {
        "long_annual_cost_rate": 0.054,
        "short_annual_cost_rate": 0.0,
        "source_long_financing_rate": -0.054,
        "source_short_financing_rate": 0.0282,
        "favorable_credit_clipped_to_zero": True,
        "accrual": "actual_elapsed_wall_clock",
        "seconds_per_year": SECONDS_PER_YEAR,
        "cost_formula": "open_notional*annual_cost_rate*elapsed_wall_clock_seconds/seconds_per_year",
        "current_instrument_snapshot_sha256": instrument["sanitized_snapshot_sha256"],
        "revalidate_before_each_run": True,
        "drift_action": "FAIL_CLOSED",
        "historical_rate_series_complete": False,
        "implicit_zero_used": False,
    }
    if (observed["financing_or_swap"] != financing_expected
            or float(instrument["long_financing_rate"]) != -0.054
            or float(instrument["short_financing_rate"]) != 0.0282):
        raise RuntimeError("PROSPECTIVE_COST_FINANCING_INVALID")

    broker_policy = broker["market_order_no_gslo_policy"]
    gslo_expected = {
        "account_currency_per_execution": 0.0,
        "treatment": "structural_zero_only_under_exact_hash_bound_no_gslo_policy",
        "market_order_no_gslo_policy_sha256": broker_policy["policy_sha256"],
        "policy_source": broker_policy["source"],
        "drift_action": "FAIL_CLOSED",
        "implicit_zero_used": False,
    }
    if observed["guaranteed_execution_fee"] != gslo_expected:
        raise RuntimeError("PROSPECTIVE_COST_GSLO_INVALID")
    _source(gslo_expected["policy_source"], "GSLO_POLICY", verify_local_sources)

    risk_expected = {
        "hold_risk_penalty_annual_bps_by_side": {"long": 0.0, "short": 0.0},
        "risk_utility_formula": "annual_open_notional_penalty_bps*elapsed_wall_clock_seconds/seconds_per_year",
        "risk_utility_is_cash_pnl": False,
        "seconds_per_year": SECONDS_PER_YEAR,
        "parameter_origin": "cash_net_pnl_learning_baseline_explicit_cli_zero_no_train_risk_fit",
    }
    if observed["hold_risk_utility"] != risk_expected:
        raise RuntimeError("PROSPECTIVE_COST_RISK_INVALID")
    observed["artifact_sha256"] = declared
    return observed


def require_cost_method_receipt(value: Mapping[str, Any], *, expected_policy_sha256: str) -> dict[str, Any]:
    expected = {
        "schema_version": COST_METHOD_RECEIPT_SCHEMA_VERSION,
        "decision": "PASS_METHOD_COMPLETE_NOT_ECONOMICS_PASS",
        "policy_artifact_sha256": expected_policy_sha256,
        "spread_formula": "side_correct_entry_and_exit_price_from_executable_bid_ask_tape",
        "latency_slippage_formula": "open_notional*adverse_bps_per_execution/10000_at_each_execution",
        "commission_formula": "open_notional*commission_bps_per_execution/10000_at_each_execution",
        "financing_formula": "open_notional*annual_cost_rate_by_side*elapsed_wall_clock_seconds/31557600",
        "guaranteed_execution_fee_formula": "zero_only_while_hash_bound_no_gslo_policy_is_unchanged",
        "risk_utility_formula": "annual_open_notional_penalty_bps*elapsed_wall_clock_seconds/seconds_per_year",
        "risk_utility_is_cash_pnl": False,
        "undiscounted_net_cash_pnl_formula": "gross_executable_bid_ask_cash_pnl-minus-commission-minus-latency_slippage-minus-financing-minus-guaranteed_execution_fee",
        "same_component_double_counting_forbidden": True,
        "historical_cost_truth_qualified": False,
        "test_data_used": False,
    }
    if not isinstance(value, Mapping) or set(value) != set(expected) | {"receipt_sha256"}:
        raise RuntimeError("PROSPECTIVE_COST_METHOD_SCHEMA_INVALID")
    observed = dict(value)
    receipt = _sha(observed.pop("receipt_sha256"), "METHOD")
    if observed != expected or receipt != canonical_sha256(observed):
        raise RuntimeError("PROSPECTIVE_COST_METHOD_INVALID")
    observed["receipt_sha256"] = receipt
    return observed


def require_cost_parameter_authority(
    value: Mapping[str, Any], *, expected_coverage_start_utc: Any,
    expected_coverage_end_utc: Any, verify_local_sources: bool = True,
) -> dict[str, Any]:
    keys = {"schema_version", "decision", "historical_cost_truth_qualified", "economics_pass_claimed",
            "preregistered_at_utc", "coverage_start_utc", "coverage_end_utc_exclusive", "policy",
            "method_receipt", "economics_fact_manifest", "component_artifacts", "parameters",
            "hold_risk_penalty_annual_bps_by_side", "risk_utility_formula", "risk_utility_is_cash_pnl",
            "future_train_only_risk_sweep_required", "test_data_used", "authority_sha256"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeError("PROSPECTIVE_COST_AUTHORITY_SCHEMA_INVALID")
    observed = dict(value)
    authority_sha = _sha(observed.pop("authority_sha256"), "AUTHORITY")
    if authority_sha != canonical_sha256(observed):
        raise RuntimeError("PROSPECTIVE_COST_AUTHORITY_HASH_INVALID")
    start, end = _utc(expected_coverage_start_utc, "AUTH_START"), _utc(expected_coverage_end_utc, "AUTH_END")
    if (observed["schema_version"] != COST_PARAMETER_AUTHORITY_SCHEMA_VERSION
            or observed["decision"] != AUTHORITY_DECISION
            or observed["historical_cost_truth_qualified"] is not False
            or observed["economics_pass_claimed"] is not False
            or observed["coverage_start_utc"] != start.isoformat()
            or observed["coverage_end_utc_exclusive"] != end.isoformat()
            or observed["test_data_used"] is not False
            or observed["future_train_only_risk_sweep_required"] is not True):
        raise RuntimeError("PROSPECTIVE_COST_AUTHORITY_HEADER_INVALID")

    policy_binding = _artifact(observed["policy"], "AUTH_POLICY", "artifact_sha256", verify_local_sources)
    policy = require_prospective_cost_policy(
        json.loads(Path(policy_binding["path"]).read_text()),
        expected_coverage_start_utc=start, expected_coverage_end_utc=end,
        verify_local_sources=verify_local_sources,
    )
    if (policy["artifact_sha256"] != policy_binding["artifact_sha256"]
            or observed["preregistered_at_utc"] != policy["preregistered_at_utc"]):
        raise RuntimeError("PROSPECTIVE_COST_AUTHORITY_POLICY_INVALID")

    method_binding = _artifact(observed["method_receipt"], "AUTH_METHOD", "receipt_sha256", verify_local_sources)
    method = require_cost_method_receipt(
        json.loads(Path(method_binding["path"]).read_text()),
        expected_policy_sha256=policy["artifact_sha256"],
    )
    if method["receipt_sha256"] != method_binding["receipt_sha256"]:
        raise RuntimeError("PROSPECTIVE_COST_AUTHORITY_METHOD_INVALID")

    manifest_binding = _artifact(observed["economics_fact_manifest"], "AUTH_MANIFEST", "manifest_sha256", verify_local_sources)
    manifest = require_economics_fact_manifest(
        json.loads(Path(manifest_binding["path"]).read_text()),
        expected_coverage_start_utc=start, expected_coverage_end_utc=end,
    )
    if manifest["manifest_sha256"] != manifest_binding["manifest_sha256"]:
        raise RuntimeError("PROSPECTIVE_COST_AUTHORITY_MANIFEST_INVALID")
    components = observed["component_artifacts"]
    if (not isinstance(components, Mapping) or set(components) != set(REQUIRED_COMPONENTS)
            or any(components[name] != manifest["component_artifacts"][name] for name in REQUIRED_COMPONENTS)):
        raise RuntimeError("PROSPECTIVE_COST_AUTHORITY_COMPONENTS_INVALID")

    parameters = {
        "executable_bid_ask": {"mode": "side_correct_executable_bid_ask_from_bound_m1_tape"},
        "commission": {"bps_per_execution": 0.0},
        "execution_slippage": {"central_bps_per_execution": 2.0, "val_sensitivity_bps_per_execution": [1.0, 2.0, 4.0]},
        "financing_or_swap": {"long_annual_cost_rate": 0.054, "short_annual_cost_rate": 0.0, "favorable_credit_clipped_to_zero": True},
        "guaranteed_execution_fee": {"account_currency_per_execution": 0.0, "zero_requires_hash_bound_no_gslo_policy": True},
    }
    if (observed["parameters"] != parameters
            or observed["hold_risk_penalty_annual_bps_by_side"] != {"long": 0.0, "short": 0.0}
            or observed["hold_risk_penalty_annual_bps_by_side"] != policy["hold_risk_utility"]["hold_risk_penalty_annual_bps_by_side"]
            or observed["risk_utility_formula"] != policy["hold_risk_utility"]["risk_utility_formula"]
            or observed["risk_utility_is_cash_pnl"] is not False):
        raise RuntimeError("PROSPECTIVE_COST_AUTHORITY_PARAMETERS_INVALID")
    observed["authority_sha256"] = authority_sha
    return observed
