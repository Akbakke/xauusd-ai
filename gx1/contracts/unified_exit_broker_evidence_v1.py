"""Sanitized broker facts for prospective Exit economics.

The artifact deliberately separates current broker terms from pre-cutoff
execution observations.  It is evidence for a conservative prospective policy;
it is never a historical cost-label authority.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

UNIFIED_EXIT_BROKER_EVIDENCE_SCHEMA_VERSION = "gx1_unified_exit_broker_evidence_v1"
UNIFIED_EXIT_BROKER_EVIDENCE_CUTOFF_UTC = "2026-05-31T23:59:59+00:00"
_SHA = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_KEY_FRAGMENTS = (
    "accountid", "account_id", "userid", "user_id", "requestid", "request_id",
    "transactionid", "transaction_id", "tradeid", "trade_id", "orderid", "order_id",
    "token", "authorization", "secret", "api_key", "apikey",
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise RuntimeError(f"UNIFIED_EXIT_BROKER_EVIDENCE_{label}_SHA_INVALID")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"UNIFIED_EXIT_BROKER_EVIDENCE_{label}_INVALID")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"UNIFIED_EXIT_BROKER_EVIDENCE_{label}_INVALID") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"UNIFIED_EXIT_BROKER_EVIDENCE_{label}_INVALID")
    return result


def _utc(value: Any, label: str) -> str:
    try:
        ts = pd.Timestamp(value)
    except Exception as exc:
        raise RuntimeError(f"UNIFIED_EXIT_BROKER_EVIDENCE_{label}_TIME_INVALID") from exc
    if pd.isna(ts) or ts.tz is None or ts.utcoffset() != pd.Timedelta(0):
        raise RuntimeError(f"UNIFIED_EXIT_BROKER_EVIDENCE_{label}_TIME_INVALID")
    return ts.tz_convert("UTC").isoformat()


def _reject_sensitive_keys(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(fragment in normalized for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                raise RuntimeError(f"UNIFIED_EXIT_BROKER_EVIDENCE_SENSITIVE_KEY:{path}.{key}")
            _reject_sensitive_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_sensitive_keys(item, f"{path}[{index}]")


def _require_local_file(binding: Mapping[str, Any], label: str, verify: bool) -> None:
    if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
        raise RuntimeError(f"UNIFIED_EXIT_BROKER_EVIDENCE_{label}_SOURCE_INVALID")
    path = binding["path"]
    digest = _sha(binding["sha256"], label)
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise RuntimeError(f"UNIFIED_EXIT_BROKER_EVIDENCE_{label}_SOURCE_INVALID")
    if verify:
        source = Path(path)
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"UNIFIED_EXIT_BROKER_EVIDENCE_{label}_SOURCE_INVALID")
        observed = hashlib.sha256(source.read_bytes()).hexdigest()
        if observed != digest:
            raise RuntimeError(f"UNIFIED_EXIT_BROKER_EVIDENCE_{label}_SOURCE_HASH_MISMATCH")


def seal_unified_exit_broker_evidence_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or "artifact_sha256" in value:
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_SEAL_INPUT_INVALID")
    result = dict(value)
    _reject_sensitive_keys(result)
    result["artifact_sha256"] = _canonical_sha256(result)
    return result


def require_unified_exit_broker_evidence_v1(
    value: Mapping[str, Any], *, verify_local_sources: bool = False
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_INVALID")
    observed = dict(value)
    expected_keys = {
        "schema_version", "decision", "artifact_kind", "generated_at_utc",
        "observation_window", "privacy", "current_prospective_terms",
        "market_order_no_gslo_policy", "execution_observations",
        "financing_observations", "executable_quote_source", "qualification",
        "artifact_sha256",
    }
    if set(observed) != expected_keys:
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_KEYS_INVALID")
    _reject_sensitive_keys(observed)
    declared = _sha(observed.pop("artifact_sha256"), "ARTIFACT")
    if declared != _canonical_sha256(observed):
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_CONTENT_HASH_INVALID")
    observed["artifact_sha256"] = declared
    cutoff = _utc(UNIFIED_EXIT_BROKER_EVIDENCE_CUTOFF_UTC, "CUTOFF")
    window = observed["observation_window"]
    privacy = observed["privacy"]
    if (
        observed["schema_version"] != UNIFIED_EXIT_BROKER_EVIDENCE_SCHEMA_VERSION
        or observed["decision"] != "PASS_SANITIZED_EVIDENCE_NOT_HISTORICAL_COST_TRUTH"
        or observed["artifact_kind"] != "prospective_terms_plus_pre_cutoff_execution_observations"
        or not isinstance(observed["generated_at_utc"], str)
        or not isinstance(window, Mapping)
        or set(window) != {"start_utc", "cutoff_utc", "post_cutoff_observations_excluded", "test_data_used"}
        or _utc(window["cutoff_utc"], "CUTOFF") != cutoff
        or window["post_cutoff_observations_excluded"] is not True
        or window["test_data_used"] is not False
        or not isinstance(privacy, Mapping)
        or privacy != {"broker_identity_values_persisted": False, "credentials_persisted": False, "raw_broker_responses_persisted": False}
    ):
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_HEADER_INVALID")
    _utc(observed["generated_at_utc"], "GENERATED")
    start = pd.Timestamp(_utc(window["start_utc"], "START"))
    cutoff_ts = pd.Timestamp(cutoff)
    if start > cutoff_ts:
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_WINDOW_INVALID")

    terms = observed["current_prospective_terms"]
    if not isinstance(terms, Mapping) or set(terms) != {"scope", "account", "instrument"} or terms["scope"] != "current_terms_observed_after_cutoff_prospective_use_only":
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_TERMS_INVALID")
    account = terms["account"]
    instrument = terms["instrument"]
    if (
        not isinstance(account, Mapping)
        or set(account) != {"environment", "currency", "margin_rate", "hedging_enabled", "gslo_mode", "lifetime_commission_account_units", "lifetime_financing_account_units", "lifetime_guaranteed_execution_fees_account_units", "sanitized_snapshot_sha256"}
        or account["environment"] not in {"practice", "live"}
        or not isinstance(account["currency"], str)
        or type(account["hedging_enabled"]) is not bool
        or account["gslo_mode"] not in {"ALLOWED", "REQUIRED", "DISABLED"}
    ):
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_ACCOUNT_TERMS_INVALID")
    for key in ("margin_rate", "lifetime_commission_account_units", "lifetime_financing_account_units", "lifetime_guaranteed_execution_fees_account_units"):
        _finite(account[key], f"ACCOUNT_{key.upper()}")
    account_payload = {key: item for key, item in account.items() if key != "sanitized_snapshot_sha256"}
    if _sha(account["sanitized_snapshot_sha256"], "ACCOUNT_SNAPSHOT") != _canonical_sha256(account_payload):
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_ACCOUNT_SNAPSHOT_HASH_INVALID")
    required_instrument = {"name", "type", "display_precision", "trade_units_precision", "minimum_trade_size", "maximum_order_units", "margin_rate", "gslo_mode", "gslo_execution_premium", "minimum_gslo_distance", "financing_mode", "long_financing_rate", "short_financing_rate", "financing_days_of_week", "sanitized_snapshot_sha256"}
    if not isinstance(instrument, Mapping) or set(instrument) != required_instrument or instrument["name"] != "XAU_USD" or instrument["type"] != "METAL" or instrument["financing_mode"] != "DAILY_INSTRUMENT":
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_INSTRUMENT_TERMS_INVALID")
    for key in ("minimum_trade_size", "maximum_order_units", "margin_rate", "gslo_execution_premium", "minimum_gslo_distance", "long_financing_rate", "short_financing_rate"):
        _finite(instrument[key], f"INSTRUMENT_{key.upper()}")
    if not isinstance(instrument["financing_days_of_week"], list) or len(instrument["financing_days_of_week"]) != 7:
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_FINANCING_SCHEDULE_INVALID")
    instrument_payload = {key: item for key, item in instrument.items() if key != "sanitized_snapshot_sha256"}
    if _sha(instrument["sanitized_snapshot_sha256"], "INSTRUMENT_SNAPSHOT") != _canonical_sha256(instrument_payload):
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_INSTRUMENT_SNAPSHOT_HASH_INVALID")

    policy = observed["market_order_no_gslo_policy"]
    expected_policy_keys = {"policy", "entry_order_type", "exit_operation", "gslo_order_attached", "gslo_fee_treatment", "source", "source_commit", "policy_sha256"}
    if (
        not isinstance(policy, Mapping) or set(policy) != expected_policy_keys
        or policy["policy"] != "market_entry_and_market_trade_close_without_gslo"
        or policy["entry_order_type"] != "MARKET"
        or policy["exit_operation"] != "trade_close"
        or policy["gslo_order_attached"] is not False
        or policy["gslo_fee_treatment"] != "structural_zero_only_while_exact_no_gslo_policy_is_enforced"
        or not isinstance(policy["source_commit"], str) or re.fullmatch(r"[0-9a-f]{40}", policy["source_commit"]) is None
    ):
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_POLICY_INVALID")
    _require_local_file(policy["source"], "POLICY", verify_local_sources)
    policy_payload = {k: v for k, v in policy.items() if k != "policy_sha256"}
    if _sha(policy["policy_sha256"], "POLICY") != _canonical_sha256(policy_payload):
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_POLICY_HASH_INVALID")

    execution = observed["execution_observations"]
    execution_keys = {"scope", "rows", "cutoff_fill_count", "post_cutoff_fill_count_excluded", "post_cutoff_observation_status", "safe_population_sha256", "commission_present_count", "commission_nonzero_count", "half_spread_cost_present_count", "half_spread_cost_nonzero_count", "gslo_fee_present_count", "gslo_fee_nonzero_count", "full_vwap_residual_count", "full_vwap_residual_nonzero_count", "pricing_mode_conclusion", "latency_slippage_status"}
    if not isinstance(execution, Mapping) or set(execution) != execution_keys or execution["scope"] != "xauusd_order_fill_at_or_before_cutoff":
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_EXECUTION_INVALID")
    rows = execution["rows"]
    row_keys = {"time", "instrument", "reason", "units", "price", "full_vwap", "commission_account_units", "financing_account_units", "guaranteed_execution_fee_account_units", "quote_guaranteed_execution_fee", "half_spread_cost_account_units"}
    if not isinstance(rows, list):
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_EXECUTION_INVALID")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != row_keys or row["instrument"] != "XAU_USD" or pd.Timestamp(_utc(row["time"], "FILL")) > cutoff_ts:
            raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_EXECUTION_ROW_INVALID")
        for key in row_keys - {"time", "instrument", "reason"}:
            _finite(row[key], f"FILL_{key.upper()}")
    if execution["cutoff_fill_count"] != len(rows) or execution["safe_population_sha256"] != _canonical_sha256(rows):
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_EXECUTION_POPULATION_INVALID")
    count_fields = [key for key in execution_keys if key.endswith("_count") or key.endswith("_excluded")]
    if any(isinstance(execution[key], bool) or not isinstance(execution[key], int) or execution[key] < 0 for key in count_fields):
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_EXECUTION_COUNT_INVALID")
    expected_counts = {
        "commission_present_count": len(rows),
        "commission_nonzero_count": sum(float(row["commission_account_units"]) != 0.0 for row in rows),
        "half_spread_cost_present_count": len(rows),
        "half_spread_cost_nonzero_count": sum(float(row["half_spread_cost_account_units"]) != 0.0 for row in rows),
        "gslo_fee_present_count": len(rows),
        "gslo_fee_nonzero_count": sum(float(row["guaranteed_execution_fee_account_units"]) != 0.0 for row in rows),
        "full_vwap_residual_count": len(rows),
        "full_vwap_residual_nonzero_count": sum(not math.isclose(float(row["price"]), float(row["full_vwap"]), rel_tol=0.0, abs_tol=1e-12) for row in rows),
    }
    if any(execution[key] != expected for key, expected in expected_counts.items()) or execution["post_cutoff_fill_count_excluded"] != 0 or execution["post_cutoff_observation_status"] != "NOT_QUERIED" or execution["latency_slippage_status"] != "UNKNOWN_NO_PRE_CUTOFF_DECISION_QUOTE_TO_FILL_CLOCK_BINDING" or execution["pricing_mode_conclusion"] != "observed_zero_commission_with_nonzero_spread_cost_not_broker_plan_label":
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_EXECUTION_SUMMARY_INVALID")

    financing = observed["financing_observations"]
    financing_keys = {"scope", "rows", "daily_financing_count", "nonzero_daily_financing_count", "open_trade_financing_child_count", "safe_population_sha256", "historical_rate_series_status"}
    financing_row_keys = {"time", "financing_account_units", "xau_position_financing_account_units", "account_financing_mode", "open_trade_financing_count"}
    if not isinstance(financing, Mapping) or set(financing) != financing_keys or financing["scope"] != "xauusd_daily_financing_at_or_before_cutoff" or not isinstance(financing["rows"], list):
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_FINANCING_INVALID")
    for row in financing["rows"]:
        if not isinstance(row, Mapping) or set(row) != financing_row_keys or pd.Timestamp(_utc(row["time"], "FINANCING")) > cutoff_ts or row["account_financing_mode"] != "DAILY_INSTRUMENT":
            raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_FINANCING_ROW_INVALID")
        _finite(row["financing_account_units"], "FINANCING")
        _finite(row["xau_position_financing_account_units"], "XAU_FINANCING")
        if isinstance(row["open_trade_financing_count"], bool) or not isinstance(row["open_trade_financing_count"], int) or row["open_trade_financing_count"] < 0:
            raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_FINANCING_ROW_INVALID")
    frows = financing["rows"]
    if financing["daily_financing_count"] != len(frows) or financing["nonzero_daily_financing_count"] != sum(float(row["xau_position_financing_account_units"]) != 0.0 for row in frows) or financing["open_trade_financing_child_count"] != sum(row["open_trade_financing_count"] for row in frows) or financing["safe_population_sha256"] != _canonical_sha256(frows) or financing["historical_rate_series_status"] != "INCOMPLETE_NO_FULL_TRAIN_YEAR_RATE_HISTORY":
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_FINANCING_SUMMARY_INVALID")

    quotes = observed["executable_quote_source"]
    quote_keys = {"scope", "parquet", "manifest", "manifest_schema_version", "row_count", "time_min_utc", "time_max_utc", "columns", "quote_complete_m1", "test_accessed"}
    if not isinstance(quotes, Mapping) or set(quotes) != quote_keys or quotes["scope"] != "pretest_direct_m1_bid_ask_executable_quote_source" or quotes["quote_complete_m1"] is not True or quotes["test_accessed"] is not False or not isinstance(quotes["columns"], list) or quotes["row_count"] < 1:
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_QUOTE_SOURCE_INVALID")
    _require_local_file(quotes["parquet"], "QUOTE_PARQUET", verify_local_sources)
    _require_local_file(quotes["manifest"], "QUOTE_MANIFEST", verify_local_sources)

    qualification = observed["qualification"]
    expected_qualification = {
        "historical_cost_truth_qualified": False,
        "sanitized_evidence_package_qualified": True,
        "conservative_prospective_policy_can_qualify": True,
        "prospective_policy_conditions": [
            "executable_bid_ask_comes_from_hash_bound_pretest_m1_quotes",
            "commission_zero_is_revalidated_against_current_account_before_use",
            "latency_slippage_uses_an_explicit_conservative_nonfitted_policy",
            "financing_uses_frozen_current_terms_with_favorable_credit_clipped_to_zero",
            "market_order_no_gslo_policy_remains_hash_bound",
            "broker_term_or_policy_drift_fails_closed",
        ],
        "blocking_historical_facts": [
            "no_full_train_year_historical_financing_rate_series",
            "no_pre_cutoff_causal_decision_quote_to_fill_latency_population",
            "no_explicit_broker_pricing_plan_label",
        ],
    }
    if qualification != expected_qualification:
        raise RuntimeError("UNIFIED_EXIT_BROKER_EVIDENCE_QUALIFICATION_INVALID")
    return observed


__all__ = (
    "UNIFIED_EXIT_BROKER_EVIDENCE_CUTOFF_UTC",
    "UNIFIED_EXIT_BROKER_EVIDENCE_SCHEMA_VERSION",
    "require_unified_exit_broker_evidence_v1",
    "seal_unified_exit_broker_evidence_v1",
)
