#!/usr/bin/env python3
"""Materialize a secret-free broker evidence package for Exit economics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from gx1.contracts.unified_exit_broker_evidence_v1 import (
    UNIFIED_EXIT_BROKER_EVIDENCE_CUTOFF_UTC,
    UNIFIED_EXIT_BROKER_EVIDENCE_SCHEMA_VERSION,
    require_unified_exit_broker_evidence_v1,
    seal_unified_exit_broker_evidence_v1,
)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _load_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        result[key] = value.strip().strip('"').strip("'")
    return result


def _get(session: requests.Session, url: str, *, params: dict[str, str] | None = None) -> dict[str, Any]:
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError("BROKER_EVIDENCE_RESPONSE_INVALID")
    return value


def _transactions(
    session: requests.Session,
    base_url: str,
    account_ref: str,
    *,
    start_utc: str,
    end_utc: str,
) -> list[dict[str, Any]]:
    summary = _get(
        session,
        f"{base_url}/accounts/{account_ref}/transactions",
        params={"from": start_utc, "to": end_utc, "pageSize": "1000"},
    )
    pages = summary.get("pages")
    if not isinstance(pages, list):
        raise RuntimeError("BROKER_EVIDENCE_TRANSACTION_PAGES_INVALID")
    result: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, str) or not page.startswith(base_url):
            raise RuntimeError("BROKER_EVIDENCE_TRANSACTION_PAGE_INVALID")
        payload = _get(session, page)
        rows = payload.get("transactions")
        if not isinstance(rows, list):
            raise RuntimeError("BROKER_EVIDENCE_TRANSACTIONS_INVALID")
        result.extend(dict(row) for row in rows if isinstance(row, dict))
    return result


def _number(row: dict[str, Any], key: str) -> float:
    if key not in row:
        raise RuntimeError(f"BROKER_EVIDENCE_REQUIRED_FIELD_MISSING:{key}")
    return float(row[key])


def _sanitized_fill(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "time": pd.Timestamp(row["time"]).tz_convert("UTC").isoformat(),
        "instrument": row["instrument"],
        "reason": row["reason"],
        "units": _number(row, "units"),
        "price": _number(row, "price"),
        "full_vwap": _number(row, "fullVWAP"),
        "commission_account_units": _number(row, "commission"),
        "financing_account_units": _number(row, "financing"),
        "guaranteed_execution_fee_account_units": _number(row, "guaranteedExecutionFee"),
        "quote_guaranteed_execution_fee": _number(row, "quoteGuaranteedExecutionFee"),
        "half_spread_cost_account_units": _number(row, "halfSpreadCost"),
    }


def _sanitized_financing(row: dict[str, Any]) -> dict[str, Any] | None:
    positions = [
        item
        for item in row.get("positionFinancings", [])
        if isinstance(item, dict) and item.get("instrument") == "XAU_USD"
    ]
    if len(positions) != 1:
        return None
    position = positions[0]
    children = position.get("openTradeFinancings")
    if not isinstance(children, list):
        raise RuntimeError("BROKER_EVIDENCE_OPEN_TRADE_FINANCING_INVALID")
    return {
        "time": pd.Timestamp(row["time"]).tz_convert("UTC").isoformat(),
        "financing_account_units": _number(row, "financing"),
        "xau_position_financing_account_units": _number(position, "financing"),
        "account_financing_mode": position.get("accountFinancingMode"),
        "open_trade_financing_count": len(children),
    }


def materialize(
    *,
    env_file: Path,
    output_path: Path,
    quote_parquet: Path,
    quote_manifest: Path,
    observation_start_utc: str,
    cutoff_utc: str,
    policy_source: Path,
) -> dict[str, Any]:
    cutoff = pd.Timestamp(cutoff_utc)
    if cutoff.tz is None or cutoff.tz_convert("UTC").isoformat() != UNIFIED_EXIT_BROKER_EVIDENCE_CUTOFF_UTC:
        raise RuntimeError("BROKER_EVIDENCE_CUTOFF_INVALID")
    config = _load_env(env_file)
    environment = config.get("OANDA_ENV", "practice").lower()
    if environment not in {"practice", "live"}:
        raise RuntimeError("BROKER_EVIDENCE_ENVIRONMENT_INVALID")
    token = config.get("OANDA_API_TOKEN") or config.get("OANDA_API_KEY")
    account_ref = config.get("OANDA_ACCOUNT_ID")
    if not token or not account_ref:
        raise RuntimeError("BROKER_EVIDENCE_CREDENTIALS_UNAVAILABLE")
    base = "https://api-fxpractice.oanda.com/v3" if environment == "practice" else "https://api-fxtrade.oanda.com/v3"
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/json", "Accept-Datetime-Format": "RFC3339"})
    account_payload = _get(session, f"{base}/accounts/{account_ref}/summary")
    account = account_payload.get("account")
    instrument_payload = _get(session, f"{base}/accounts/{account_ref}/instruments", params={"instruments": "XAU_USD"})
    instruments = instrument_payload.get("instruments")
    if not isinstance(account, dict) or not isinstance(instruments, list):
        raise RuntimeError("BROKER_EVIDENCE_CURRENT_TERMS_INVALID")
    matches = [row for row in instruments if isinstance(row, dict) and row.get("name") == "XAU_USD"]
    if len(matches) != 1:
        raise RuntimeError("BROKER_EVIDENCE_XAU_INSTRUMENT_INVALID")
    instrument = matches[0]
    now = datetime.now(timezone.utc).isoformat()
    cutoff_ts = pd.Timestamp(cutoff_utc)
    transactions = _transactions(session, base, account_ref, start_utc=observation_start_utc, end_utc=cutoff_ts.tz_convert("UTC").isoformat())
    all_fills = [row for row in transactions if row.get("type") == "ORDER_FILL" and row.get("instrument") == "XAU_USD"]
    before = [row for row in all_fills if pd.Timestamp(row["time"]) <= cutoff_ts]
    after = [row for row in all_fills if pd.Timestamp(row["time"]) > cutoff_ts]
    if after:
        raise RuntimeError("BROKER_EVIDENCE_POST_CUTOFF_TRANSACTION_RETURNED")
    fills = [_sanitized_fill(row) for row in sorted(before, key=lambda row: row["time"])]
    financing_rows = []
    for row in sorted((row for row in transactions if row.get("type") == "DAILY_FINANCING" and pd.Timestamp(row["time"]) <= cutoff_ts), key=lambda row: row["time"]):
        sanitized = _sanitized_financing(row)
        if sanitized is not None:
            financing_rows.append(sanitized)

    account_safe = {
        "environment": environment,
        "currency": account["currency"],
        "margin_rate": float(account["marginRate"]),
        "hedging_enabled": account["hedgingEnabled"],
        "gslo_mode": account["guaranteedStopLossOrderMode"],
        "lifetime_commission_account_units": float(account["commission"]),
        "lifetime_financing_account_units": float(account["financing"]),
        "lifetime_guaranteed_execution_fees_account_units": float(account["guaranteedExecutionFees"]),
    }
    account_safe["sanitized_snapshot_sha256"] = _canonical_sha(account_safe)
    days = instrument.get("financing", {}).get("financingDaysOfWeek")
    if not isinstance(days, list):
        raise RuntimeError("BROKER_EVIDENCE_FINANCING_TERMS_INVALID")
    instrument_safe = {
        "name": instrument["name"],
        "type": instrument["type"],
        "display_precision": int(instrument["displayPrecision"]),
        "trade_units_precision": int(instrument["tradeUnitsPrecision"]),
        "minimum_trade_size": float(instrument["minimumTradeSize"]),
        "maximum_order_units": float(instrument["maximumOrderUnits"]),
        "margin_rate": float(instrument["marginRate"]),
        "gslo_mode": instrument["guaranteedStopLossOrderMode"],
        "gslo_execution_premium": float(instrument["guaranteedStopLossOrderExecutionPremium"]),
        "minimum_gslo_distance": float(instrument["minimumGuaranteedStopLossDistance"]),
        "financing_mode": "DAILY_INSTRUMENT",
        "long_financing_rate": float(instrument["financing"]["longRate"]),
        "short_financing_rate": float(instrument["financing"]["shortRate"]),
        "financing_days_of_week": [{"day": row["dayOfWeek"], "days_charged": int(row["daysCharged"])} for row in days],
    }
    instrument_safe["sanitized_snapshot_sha256"] = _canonical_sha(instrument_safe)

    policy_source = policy_source.resolve()
    source_commit = subprocess.check_output(["git", "-C", str(policy_source.parents[2]), "rev-parse", "HEAD"], text=True).strip()
    policy = {
        "policy": "market_entry_and_market_trade_close_without_gslo",
        "entry_order_type": "MARKET",
        "exit_operation": "trade_close",
        "gslo_order_attached": False,
        "gslo_fee_treatment": "structural_zero_only_while_exact_no_gslo_policy_is_enforced",
        "source": {"path": str(policy_source), "sha256": _sha_file(policy_source)},
        "source_commit": source_commit,
    }
    policy["policy_sha256"] = _canonical_sha(policy)

    manifest = json.loads(quote_manifest.read_text(encoding="utf-8"))
    columns = list(manifest["output_columns"])
    quote_binding = {
        "scope": "pretest_direct_m1_bid_ask_executable_quote_source",
        "parquet": {"path": str(quote_parquet.resolve()), "sha256": _sha_file(quote_parquet)},
        "manifest": {"path": str(quote_manifest.resolve()), "sha256": _sha_file(quote_manifest)},
        "manifest_schema_version": manifest["schema_version"],
        "row_count": int(manifest["row_count"]),
        "time_min_utc": pd.Timestamp(manifest["time_min_utc"]).tz_convert("UTC").isoformat(),
        "time_max_utc": pd.Timestamp(manifest["time_max_utc"]).tz_convert("UTC").isoformat(),
        "columns": columns,
        "quote_complete_m1": manifest["quote_complete_m1"],
        "test_accessed": manifest["test_accessed"],
    }
    if manifest["output_parquet_sha256"] != quote_binding["parquet"]["sha256"] or manifest["output_parquet"] != quote_binding["parquet"]["path"]:
        raise RuntimeError("BROKER_EVIDENCE_QUOTE_MANIFEST_BINDING_INVALID")

    execution = {
        "scope": "xauusd_order_fill_at_or_before_cutoff",
        "rows": fills,
        "cutoff_fill_count": len(fills),
        "post_cutoff_fill_count_excluded": 0,
        "post_cutoff_observation_status": "NOT_QUERIED",
        "safe_population_sha256": _canonical_sha(fills),
        "commission_present_count": len(fills),
        "commission_nonzero_count": sum(row["commission_account_units"] != 0.0 for row in fills),
        "half_spread_cost_present_count": len(fills),
        "half_spread_cost_nonzero_count": sum(row["half_spread_cost_account_units"] != 0.0 for row in fills),
        "gslo_fee_present_count": len(fills),
        "gslo_fee_nonzero_count": sum(row["guaranteed_execution_fee_account_units"] != 0.0 for row in fills),
        "full_vwap_residual_count": len(fills),
        "full_vwap_residual_nonzero_count": sum(row["price"] != row["full_vwap"] for row in fills),
        "pricing_mode_conclusion": "observed_zero_commission_with_nonzero_spread_cost_not_broker_plan_label",
        "latency_slippage_status": "UNKNOWN_NO_PRE_CUTOFF_DECISION_QUOTE_TO_FILL_CLOCK_BINDING",
    }
    financing = {
        "scope": "xauusd_daily_financing_at_or_before_cutoff",
        "rows": financing_rows,
        "daily_financing_count": len(financing_rows),
        "nonzero_daily_financing_count": sum(row["xau_position_financing_account_units"] != 0.0 for row in financing_rows),
        "open_trade_financing_child_count": sum(row["open_trade_financing_count"] for row in financing_rows),
        "safe_population_sha256": _canonical_sha(financing_rows),
        "historical_rate_series_status": "INCOMPLETE_NO_FULL_TRAIN_YEAR_RATE_HISTORY",
    }
    payload = {
        "schema_version": UNIFIED_EXIT_BROKER_EVIDENCE_SCHEMA_VERSION,
        "decision": "PASS_SANITIZED_EVIDENCE_NOT_HISTORICAL_COST_TRUTH",
        "artifact_kind": "prospective_terms_plus_pre_cutoff_execution_observations",
        "generated_at_utc": now,
        "observation_window": {"start_utc": pd.Timestamp(observation_start_utc).tz_convert("UTC").isoformat(), "cutoff_utc": cutoff_ts.tz_convert("UTC").isoformat(), "post_cutoff_observations_excluded": True, "test_data_used": False},
        "privacy": {"broker_identity_values_persisted": False, "credentials_persisted": False, "raw_broker_responses_persisted": False},
        "current_prospective_terms": {"scope": "current_terms_observed_after_cutoff_prospective_use_only", "account": account_safe, "instrument": instrument_safe},
        "market_order_no_gslo_policy": policy,
        "execution_observations": execution,
        "financing_observations": financing,
        "executable_quote_source": quote_binding,
        "qualification": {
            "historical_cost_truth_qualified": False,
            "sanitized_evidence_package_qualified": True,
            "conservative_prospective_policy_can_qualify": True,
            "prospective_policy_conditions": ["executable_bid_ask_comes_from_hash_bound_pretest_m1_quotes", "commission_zero_is_revalidated_against_current_account_before_use", "latency_slippage_uses_an_explicit_conservative_nonfitted_policy", "financing_uses_frozen_current_terms_with_favorable_credit_clipped_to_zero", "market_order_no_gslo_policy_remains_hash_bound", "broker_term_or_policy_drift_fails_closed"],
            "blocking_historical_facts": ["no_full_train_year_historical_financing_rate_series", "no_pre_cutoff_causal_decision_quote_to_fill_latency_population", "no_explicit_broker_pricing_plan_label"],
        },
    }
    artifact = seal_unified_exit_broker_evidence_v1(payload)
    require_unified_exit_broker_evidence_v1(artifact, verify_local_sources=True)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise RuntimeError("BROKER_EVIDENCE_OUTPUT_ALREADY_EXISTS")
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quote-parquet", type=Path, required=True)
    parser.add_argument("--quote-manifest", type=Path, required=True)
    parser.add_argument("--policy-source", type=Path, required=True)
    parser.add_argument("--observation-start-utc", default="2025-01-01T00:00:00+00:00")
    parser.add_argument("--cutoff-utc", default=UNIFIED_EXIT_BROKER_EVIDENCE_CUTOFF_UTC)
    args = parser.parse_args()
    artifact = materialize(env_file=args.env_file, output_path=args.output, quote_parquet=args.quote_parquet, quote_manifest=args.quote_manifest, observation_start_utc=args.observation_start_utc, cutoff_utc=args.cutoff_utc, policy_source=args.policy_source)
    print(json.dumps({"decision": artifact["decision"], "artifact_sha256": artifact["artifact_sha256"], "cutoff_fill_count": artifact["execution_observations"]["cutoff_fill_count"], "post_cutoff_fill_count_excluded": artifact["execution_observations"]["post_cutoff_fill_count_excluded"], "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
