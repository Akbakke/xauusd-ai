from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import gx1.contracts.unified_exit_broker_evidence_v1 as owner


def _source(tmp_path: Path, name: str, content: bytes) -> dict[str, str]:
    path = (tmp_path / name).resolve()
    path.write_bytes(content)
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}


def _fixture(tmp_path: Path) -> dict:
    policy_source = _source(tmp_path, "policy.py", b"MARKET no gslo\n")
    quote = _source(tmp_path, "quotes.bin", b"quotes\n")
    manifest = _source(tmp_path, "manifest.json", b"{}\n")
    account = {
        "environment": "practice",
        "currency": "EUR",
        "margin_rate": 1.0 / 30.0,
        "hedging_enabled": True,
        "gslo_mode": "ALLOWED",
        "lifetime_commission_account_units": 0.0,
        "lifetime_financing_account_units": -1.0,
        "lifetime_guaranteed_execution_fees_account_units": 0.0,
    }
    account["sanitized_snapshot_sha256"] = owner._canonical_sha256(account)
    instrument = {
        "name": "XAU_USD",
        "type": "METAL",
        "display_precision": 3,
        "trade_units_precision": 1,
        "minimum_trade_size": 0.1,
        "maximum_order_units": 20000.0,
        "margin_rate": 0.05,
        "gslo_mode": "ALLOWED",
        "gslo_execution_premium": 0.5,
        "minimum_gslo_distance": 5.32,
        "financing_mode": "DAILY_INSTRUMENT",
        "long_financing_rate": -0.054,
        "short_financing_rate": 0.0282,
        "financing_days_of_week": [
            {"day": day, "days_charged": charge}
            for day, charge in zip(
                ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"],
                [1, 1, 3, 1, 1, 0, 0],
                strict=True,
            )
        ],
    }
    instrument["sanitized_snapshot_sha256"] = owner._canonical_sha256(instrument)
    policy = {
        "policy": "market_entry_and_market_trade_close_without_gslo",
        "entry_order_type": "MARKET",
        "exit_operation": "trade_close",
        "gslo_order_attached": False,
        "gslo_fee_treatment": "structural_zero_only_while_exact_no_gslo_policy_is_enforced",
        "source": policy_source,
        "source_commit": "a" * 40,
    }
    policy["policy_sha256"] = owner._canonical_sha256(policy)
    fills = [
        {
            "time": "2026-05-20T13:33:05+00:00",
            "instrument": "XAU_USD",
            "reason": "MARKET_ORDER",
            "units": 1.0,
            "price": 4489.6,
            "full_vwap": 4489.6,
            "commission_account_units": 0.0,
            "financing_account_units": 0.0,
            "guaranteed_execution_fee_account_units": 0.0,
            "quote_guaranteed_execution_fee": 0.0,
            "half_spread_cost_account_units": 0.302,
        }
    ]
    financing_rows = [
        {
            "time": "2026-05-20T21:00:00+00:00",
            "financing_account_units": -1.0,
            "xau_position_financing_account_units": -1.0,
            "account_financing_mode": "DAILY_INSTRUMENT",
            "open_trade_financing_count": 1,
        }
    ]
    payload = {
        "schema_version": owner.UNIFIED_EXIT_BROKER_EVIDENCE_SCHEMA_VERSION,
        "decision": "PASS_SANITIZED_EVIDENCE_NOT_HISTORICAL_COST_TRUTH",
        "artifact_kind": "prospective_terms_plus_pre_cutoff_execution_observations",
        "generated_at_utc": "2026-09-10T20:00:00+00:00",
        "observation_window": {"start_utc": "2025-01-01T00:00:00+00:00", "cutoff_utc": owner.UNIFIED_EXIT_BROKER_EVIDENCE_CUTOFF_UTC, "post_cutoff_observations_excluded": True, "test_data_used": False},
        "privacy": {"broker_identity_values_persisted": False, "credentials_persisted": False, "raw_broker_responses_persisted": False},
        "current_prospective_terms": {"scope": "current_terms_observed_after_cutoff_prospective_use_only", "account": account, "instrument": instrument},
        "market_order_no_gslo_policy": policy,
        "execution_observations": {
            "scope": "xauusd_order_fill_at_or_before_cutoff",
            "rows": fills,
            "cutoff_fill_count": 1,
            "post_cutoff_fill_count_excluded": 0,
            "post_cutoff_observation_status": "NOT_QUERIED",
            "safe_population_sha256": owner._canonical_sha256(fills),
            "commission_present_count": 1,
            "commission_nonzero_count": 0,
            "half_spread_cost_present_count": 1,
            "half_spread_cost_nonzero_count": 1,
            "gslo_fee_present_count": 1,
            "gslo_fee_nonzero_count": 0,
            "full_vwap_residual_count": 1,
            "full_vwap_residual_nonzero_count": 0,
            "pricing_mode_conclusion": "observed_zero_commission_with_nonzero_spread_cost_not_broker_plan_label",
            "latency_slippage_status": "UNKNOWN_NO_PRE_CUTOFF_DECISION_QUOTE_TO_FILL_CLOCK_BINDING",
        },
        "financing_observations": {
            "scope": "xauusd_daily_financing_at_or_before_cutoff",
            "rows": financing_rows,
            "daily_financing_count": 1,
            "nonzero_daily_financing_count": 1,
            "open_trade_financing_child_count": 1,
            "safe_population_sha256": owner._canonical_sha256(financing_rows),
            "historical_rate_series_status": "INCOMPLETE_NO_FULL_TRAIN_YEAR_RATE_HISTORY",
        },
        "executable_quote_source": {
            "scope": "pretest_direct_m1_bid_ask_executable_quote_source",
            "parquet": quote,
            "manifest": manifest,
            "manifest_schema_version": "gx1_direct_native_pretest_source_v2",
            "row_count": 10,
            "time_min_utc": "2019-01-01T23:00:00+00:00",
            "time_max_utc": "2026-06-30T23:59:00+00:00",
            "columns": ["time", "bid_open", "ask_open"],
            "quote_complete_m1": True,
            "test_accessed": False,
        },
        "qualification": {
            "historical_cost_truth_qualified": False,
            "sanitized_evidence_package_qualified": True,
            "conservative_prospective_policy_can_qualify": True,
            "prospective_policy_conditions": ["executable_bid_ask_comes_from_hash_bound_pretest_m1_quotes", "commission_zero_is_revalidated_against_current_account_before_use", "latency_slippage_uses_an_explicit_conservative_nonfitted_policy", "financing_uses_frozen_current_terms_with_favorable_credit_clipped_to_zero", "market_order_no_gslo_policy_remains_hash_bound", "broker_term_or_policy_drift_fails_closed"],
            "blocking_historical_facts": ["no_full_train_year_historical_financing_rate_series", "no_pre_cutoff_causal_decision_quote_to_fill_latency_population", "no_explicit_broker_pricing_plan_label"],
        },
    }
    return owner.seal_unified_exit_broker_evidence_v1(payload)


def _reseal(value: dict) -> dict:
    payload = {key: item for key, item in value.items() if key != "artifact_sha256"}
    return owner.seal_unified_exit_broker_evidence_v1(payload)


def test_valid_sanitized_evidence_and_local_sources(tmp_path: Path) -> None:
    artifact = _fixture(tmp_path)
    checked = owner.require_unified_exit_broker_evidence_v1(artifact, verify_local_sources=True)
    assert checked["qualification"]["historical_cost_truth_qualified"] is False
    assert checked["execution_observations"]["cutoff_fill_count"] == 1


def test_post_cutoff_fill_is_rejected_even_when_resealed(tmp_path: Path) -> None:
    artifact = _fixture(tmp_path)
    artifact["execution_observations"]["rows"][0]["time"] = "2026-06-01T00:00:00+00:00"
    artifact["execution_observations"]["safe_population_sha256"] = owner._canonical_sha256(artifact["execution_observations"]["rows"])
    with pytest.raises(RuntimeError, match="EXECUTION_ROW_INVALID"):
        owner.require_unified_exit_broker_evidence_v1(_reseal(artifact))


def test_sensitive_identifier_key_is_rejected(tmp_path: Path) -> None:
    artifact = _fixture(tmp_path)
    artifact["current_prospective_terms"]["account"]["account_id"] = "forbidden"
    with pytest.raises(RuntimeError, match="SENSITIVE_KEY"):
        owner.seal_unified_exit_broker_evidence_v1({key: item for key, item in artifact.items() if key != "artifact_sha256"})


def test_local_source_hash_drift_is_rejected(tmp_path: Path) -> None:
    artifact = _fixture(tmp_path)
    Path(artifact["market_order_no_gslo_policy"]["source"]["path"]).write_text("changed\n")
    with pytest.raises(RuntimeError, match="SOURCE_HASH_MISMATCH"):
        owner.require_unified_exit_broker_evidence_v1(artifact, verify_local_sources=True)


def test_self_inconsistent_snapshot_hash_is_rejected(tmp_path: Path) -> None:
    artifact = _fixture(tmp_path)
    artifact["current_prospective_terms"]["instrument"]["long_financing_rate"] = -0.10
    with pytest.raises(RuntimeError, match="INSTRUMENT_SNAPSHOT_HASH_INVALID"):
        owner.require_unified_exit_broker_evidence_v1(_reseal(artifact))
