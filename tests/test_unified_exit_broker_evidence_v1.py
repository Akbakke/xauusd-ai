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


def prospective_broker_fixture(tmp_path: Path) -> Path:
    """Build isolated synthetic quote/terms bytes, never historical broker rows."""
    import copy
    import json
    import numpy as np
    import pandas as pd

    directory = tmp_path / "cost_inputs"
    directory.mkdir(exist_ok=True)
    broker_path = directory / "broker.json"
    if broker_path.exists():
        owner.require_unified_exit_broker_evidence_v1(
            json.loads(broker_path.read_text()), verify_local_sources=True
        )
        return broker_path
    broker = _fixture(directory)
    broker.pop("artifact_sha256")
    execution = broker["execution_observations"]
    execution["rows"] = [copy.deepcopy(execution["rows"][0]) for _ in range(258)]
    for key in ("cutoff_fill_count", "commission_present_count", "half_spread_cost_present_count",
                "half_spread_cost_nonzero_count", "gslo_fee_present_count", "full_vwap_residual_count"):
        execution[key] = len(execution["rows"])
    execution["safe_population_sha256"] = owner._canonical_sha256(execution["rows"])
    times = pd.DatetimeIndex([pd.Timestamp("2025-06-01T00:00Z"),
                             *pd.date_range("2025-06-01T23:55Z", periods=16, freq="min"),
                             pd.Timestamp("2026-06-30T23:59Z")])
    bids = 2000.0 + np.arange(len(times), dtype=np.float64) * 0.1
    tape = pd.DataFrame({"time": times, "bid_open": bids, "ask_open": bids + 0.5,
                         "bid_close": bids + 0.1, "ask_close": bids + 0.6})
    tape_path = directory / "quotes.parquet"
    tape.to_parquet(tape_path, index=False)
    tape_binding = {"path": str(tape_path), "sha256": hashlib.sha256(tape_path.read_bytes()).hexdigest()}
    manifest_path = directory / "quotes.manifest.json"
    manifest = {
        "schema_version": "gx1_direct_native_pretest_source_v2", "instrument": "XAU_USD",
        "timeframe": "M1", "timestamp_semantics": "bar_start_utc", "quote_complete_m1": True,
        "test_accessed": False, "test_boundary_utc": "2026-07-01T00:00:00+00:00",
        "output_parquet": str(tape_path), "output_parquet_sha256": tape_binding["sha256"],
        "row_count": len(tape),
    }
    manifest["manifest_payload_sha256"] = owner._canonical_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest))
    quote = broker["executable_quote_source"]
    quote.update(parquet=tape_binding,
                 manifest={"path": str(manifest_path), "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()},
                 row_count=len(tape), time_min_utc=times[0].isoformat(), time_max_utc=times[-1].isoformat())
    broker = owner.seal_unified_exit_broker_evidence_v1(broker)
    owner.require_unified_exit_broker_evidence_v1(broker, verify_local_sources=True)
    broker_path.write_text(json.dumps(broker))
    return broker_path


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


def _refresh_fixture(tmp_path, monkeypatch):
    import json
    from gx1.scripts import materialize_unified_exit_broker_evidence_v1 as producer
    frozen = _fixture(tmp_path)
    frozen_path = tmp_path/"frozen.json"
    frozen_path.write_text(json.dumps(frozen, sort_keys=True))
    env = tmp_path/"test.env"
    env.write_text("OANDA_ENV=practice\nOANDA_API_TOKEN=synthetic-token\nOANDA_ACCOUNT_ID=synthetic-account\n")
    quote = tmp_path/"new_quotes.bin";quote.write_bytes(b"synthetic complete history")
    quote_manifest = tmp_path/"new_quotes.manifest.json"
    quote_manifest.write_text(json.dumps({
        "output_columns": ["time", "bid_open", "ask_open"],
        "schema_version": "gx1_direct_native_pretest_source_v2", "row_count": 20,
        "time_min_utc": "2009-06-01T00:00:00Z", "time_max_utc": "2026-06-30T23:59:00Z",
        "quote_complete_m1": True, "test_accessed": False,
        "output_parquet": str(quote), "output_parquet_sha256": producer._sha_file(quote)}))
    current = frozen["current_prospective_terms"]
    a, i = current["account"], current["instrument"]
    account = {"currency": a["currency"], "marginRate": a["margin_rate"],
        "hedgingEnabled": a["hedging_enabled"], "guaranteedStopLossOrderMode": a["gslo_mode"],
        "commission": 0.0, "financing": -2.0, "guaranteedExecutionFees": 0.0,
        "id": "do-not-persist-account", "balance": "do-not-persist-balance"}
    instrument = {"name": "XAU_USD", "type": "METAL", "displayPrecision": 3,
        "tradeUnitsPrecision": 1, "minimumTradeSize": .1, "maximumOrderUnits": 20000,
        "marginRate": .05, "guaranteedStopLossOrderMode": "ALLOWED",
        "guaranteedStopLossOrderExecutionPremium": .5, "minimumGuaranteedStopLossDistance": 5.32,
        "financing": {"longRate": -.06, "shortRate": .0282,
            "financingDaysOfWeek": [{"dayOfWeek": row["day"], "daysCharged": row["days_charged"]}
                                   for row in i["financing_days_of_week"]]}}
    calls = []
    class Response:
        status_code = 200
        def __init__(self, payload):self.payload = payload
        def raise_for_status(self):pass
        def json(self):return self.payload
    class Session:
        def __init__(self):self.headers = {}
        def get(self, url, *, params, timeout, allow_redirects):
            calls.append((url, params))
            assert timeout == 30 and allow_redirects is False
            if url.endswith("/summary"):return Response({"account": account})
            assert url.endswith("/instruments") and params == {"instruments": "XAU_USD"}
            return Response({"instruments": [instrument]})
    monkeypatch.setattr(producer.requests, "Session", Session)
    def no_transactions(*args, **kwargs):
        raise AssertionError("Frozen observation refresh must never query transactions")
    monkeypatch.setattr(producer, "_transactions", no_transactions)
    kwargs = dict(env_file=env, output_path=tmp_path/"refreshed.json", quote_parquet=quote,
        quote_manifest=quote_manifest, observation_start_utc=frozen["observation_window"]["start_utc"],
        cutoff_utc=frozen["observation_window"]["cutoff_utc"],
        policy_source=Path(frozen["market_order_no_gslo_policy"]["source"]["path"]),
        frozen_observations_path=frozen_path, frozen_observations_sha256=producer._sha_file(frozen_path))
    return producer, frozen, kwargs, calls


def test_refresh_reads_only_current_terms_and_preserves_frozen_observations(tmp_path, monkeypatch):
    producer, frozen, kwargs, calls = _refresh_fixture(tmp_path, monkeypatch)
    refreshed = producer.materialize(**kwargs)
    assert len(calls) == 2
    for key in ["execution_observations", "financing_observations", "market_order_no_gslo_policy"]:
        assert refreshed[key] == frozen[key]
    assert refreshed["current_prospective_terms"]["instrument"]["long_financing_rate"] == -.06
    assert refreshed["current_prospective_terms"]["account"]["lifetime_financing_account_units"] == -2
    assert refreshed["executable_quote_source"]["row_count"] == 20
    assert refreshed["qualification"]["historical_cost_truth_qualified"] is False
    owner.require_unified_exit_broker_evidence_v1(refreshed, verify_local_sources=True)
    text = kwargs["output_path"].read_text()
    for secret in ["synthetic-token", "synthetic-account", "do-not-persist-account", "do-not-persist-balance"]:
        assert secret not in text


@pytest.mark.parametrize("mutation", ["hash", "window", "environment", "missing_environment", "existing_output"])
def test_refresh_fails_before_any_request_on_input_or_scope_error(tmp_path, monkeypatch, mutation):
    producer, frozen, kwargs, calls = _refresh_fixture(tmp_path, monkeypatch)
    if mutation == "hash":kwargs["frozen_observations_sha256"] = "0"*64
    elif mutation == "window":kwargs["observation_start_utc"] = "2024-01-01T00:00:00Z"
    elif mutation == "environment":
        kwargs["env_file"].write_text("OANDA_ENV=live\nOANDA_API_TOKEN=fake\nOANDA_ACCOUNT_ID=fake\n")
    elif mutation == "missing_environment":
        kwargs["env_file"].write_text("OANDA_API_TOKEN=fake\nOANDA_ACCOUNT_ID=fake\n")
    else:kwargs["output_path"].write_bytes(b"preserve existing output")
    with pytest.raises(RuntimeError, match="BROKER_EVIDENCE"):
        producer.materialize(**kwargs)
    assert calls == []
    if mutation == "existing_output":
        assert kwargs["output_path"].read_bytes() == b"preserve existing output"


def test_refresh_publication_does_not_overwrite_late_file(tmp_path, monkeypatch):
    from gx1.contracts.immutable_event_authority_v1 import ImmutableEventAuthorityError
    producer, _, kwargs, _ = _refresh_fixture(tmp_path, monkeypatch)
    original = producer._publish_file_noreplace
    def collide(source, destination):
        destination.write_bytes(b"another publisher")
        return original(source, destination)
    monkeypatch.setattr(producer, "_publish_file_noreplace", collide)
    with pytest.raises(ImmutableEventAuthorityError, match="already exists"):
        producer.materialize(**kwargs)
    assert kwargs["output_path"].read_bytes() == b"another publisher"
    assert list(tmp_path.glob(".refreshed.json.*.stage")) == []


def test_request_failure_cannot_expose_account_url_or_token():
    import traceback
    import requests
    from gx1.scripts import materialize_unified_exit_broker_evidence_v1 as producer
    class Session:
        def get(self, *args, **kwargs):
            raise requests.HTTPError("private-account-url and private-token")
    try:
        producer._get(Session(), "unused")
    except RuntimeError:
        message = traceback.format_exc()
        assert "BROKER_EVIDENCE_REQUEST_FAILED" in message
        assert "private-account-url" not in message
        assert "private-token" not in message
    else:
        pytest.fail("Request must fail closed")


def test_original_observation_collection_route_retains_its_schema(tmp_path, monkeypatch):
    producer, frozen, kwargs, calls = _refresh_fixture(tmp_path, monkeypatch)
    kwargs.update(frozen_observations_path=None, frozen_observations_sha256=None)
    row = frozen["execution_observations"]["rows"][0]
    raw = {"type": "ORDER_FILL", "time": row["time"], "instrument": "XAU_USD",
        "reason": row["reason"], "units": row["units"], "price": row["price"],
        "fullVWAP": row["full_vwap"], "commission": row["commission_account_units"],
        "financing": row["financing_account_units"],
        "guaranteedExecutionFee": row["guaranteed_execution_fee_account_units"],
        "quoteGuaranteedExecutionFee": row["quote_guaranteed_execution_fee"],
        "halfSpreadCost": row["half_spread_cost_account_units"]}
    financing = {"type": "DAILY_FINANCING", "time": "2026-05-20T21:00:00+00:00",
        "financing": -1.0, "positionFinancings": [{"instrument": "XAU_USD",
            "financing": -1.0, "accountFinancingMode": "DAILY_INSTRUMENT",
            "openTradeFinancings": [{}]}]}
    transaction_calls = []
    def transactions(*args, **options):
        transaction_calls.append(options)
        return [raw, financing]
    monkeypatch.setattr(producer, "_transactions", transactions)
    monkeypatch.setattr(producer.subprocess, "check_output", lambda *args, **kwargs: "a"*40)
    result = producer.materialize(**kwargs)
    assert len(calls) == 2 and len(transaction_calls) == 1
    assert result["execution_observations"] == frozen["execution_observations"]
    assert result["financing_observations"] == frozen["financing_observations"]
    owner.require_unified_exit_broker_evidence_v1(result, verify_local_sources=True)


def test_request_rejects_redirect_without_following_it():
    from gx1.scripts import materialize_unified_exit_broker_evidence_v1 as producer
    class Response:
        status_code = 302
        def raise_for_status(self):pass
        def json(self):raise AssertionError("Redirect body must not be accepted")
    class Session:
        def get(self, url, **kwargs):
            assert kwargs["allow_redirects"] is False
            return Response()
    with pytest.raises(RuntimeError, match="HTTP_STATUS_INVALID"):
        producer._get(Session(), "unused")
