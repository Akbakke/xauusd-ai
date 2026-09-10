#!/usr/bin/env python3
"""Materialize a preregistered prospective cost policy and fact manifest."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.contracts.unified_exit_broker_evidence_v1 import require_unified_exit_broker_evidence_v1
from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import (
    ECONOMICS_FACT_MANIFEST_SCHEMA_VERSION,
    ECONOMICS_FACT_SCHEMA_VERSION,
    REQUIRED_COMPONENTS,
    canonical_sha256,
    file_sha256,
    seal_economics_component_fact,
    seal_economics_fact_manifest,
)
from gx1.contracts.unified_exit_prospective_cost_policy_v1 import (
    AUTHORITY_DECISION,
    COST_METHOD_RECEIPT_SCHEMA_VERSION,
    COST_PARAMETER_AUTHORITY_SCHEMA_VERSION,
    POLICY_DECISION,
    PROSPECTIVE_COST_POLICY_SCHEMA_VERSION,
    SECONDS_PER_YEAR,
    require_cost_parameter_authority,
    seal_cost_method_receipt,
    seal_cost_parameter_authority,
    seal_prospective_cost_policy,
)

_VERIFIERS = {
    "executable_bid_ask": "gx1_executable_bid_ask_fact_verifier_v1",
    "commission": "gx1_broker_commission_fact_verifier_v1",
    "execution_slippage": "gx1_execution_slippage_fact_verifier_v1",
    "financing_or_swap": "gx1_broker_financing_fact_verifier_v1",
    "guaranteed_execution_fee": "gx1_no_gslo_execution_policy_verifier_v1",
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _binding(path: Path, artifact: MappingLike, key: str) -> dict[str, str]:
    return {"path": str(path), "file_sha256": file_sha256(path), key: str(artifact[key])}


MappingLike = dict[str, Any]


def _exact_cli(
    *, commission_bps: float, central_slippage_bps: float,
    val_slippage_bps: list[float], long_financing_rate: float,
    short_financing_rate: float, gslo_fee: float,
    long_risk_bps: float, short_risk_bps: float,
) -> None:
    expected = (0.0, 2.0, [1.0, 2.0, 4.0], 0.054, 0.0, 0.0, 0.0, 0.0)
    observed = (
        commission_bps, central_slippage_bps, val_slippage_bps,
        long_financing_rate, short_financing_rate, gslo_fee,
        long_risk_bps, short_risk_bps,
    )
    if observed != expected:
        raise RuntimeError("PROSPECTIVE_COST_EXACT_PREREGISTERED_CLI_VALUES_REQUIRED")


def materialize_prospective_cost_policy(
    *, broker_evidence_path: Path, output_dir: Path,
    coverage_start_utc: str, coverage_end_utc: str, preregistered_at_utc: str,
    commission_bps_per_execution: float,
    central_latency_slippage_bps_per_execution: float,
    val_latency_slippage_bps_per_execution: list[float],
    long_financing_annual_cost_rate: float,
    short_financing_annual_cost_rate: float,
    gslo_fee_account_currency_per_execution: float,
    hold_risk_penalty_long_annual_bps: float,
    hold_risk_penalty_short_annual_bps: float,
    verify_local_sources: bool = True,
) -> dict[str, Any]:
    _exact_cli(
        commission_bps=commission_bps_per_execution,
        central_slippage_bps=central_latency_slippage_bps_per_execution,
        val_slippage_bps=val_latency_slippage_bps_per_execution,
        long_financing_rate=long_financing_annual_cost_rate,
        short_financing_rate=short_financing_annual_cost_rate,
        gslo_fee=gslo_fee_account_currency_per_execution,
        long_risk_bps=hold_risk_penalty_long_annual_bps,
        short_risk_bps=hold_risk_penalty_short_annual_bps,
    )
    start, end, prereg = (
        pd.Timestamp(coverage_start_utc),
        pd.Timestamp(coverage_end_utc),
        pd.Timestamp(preregistered_at_utc),
    )
    if any(ts.tz is None or ts.utcoffset() != pd.Timedelta(0) for ts in (start, end, prereg)):
        raise RuntimeError("PROSPECTIVE_COST_UTC_REQUIRED")
    if end <= start:
        raise RuntimeError("PROSPECTIVE_COST_COVERAGE_INVALID")

    broker_path = broker_evidence_path.expanduser().resolve()
    broker_file_sha = file_sha256(broker_path)
    broker = require_unified_exit_broker_evidence_v1(
        json.loads(broker_path.read_text()), verify_local_sources=verify_local_sources
    )
    execution = broker["execution_observations"]
    account = broker["current_prospective_terms"]["account"]
    instrument = broker["current_prospective_terms"]["instrument"]
    if (
        execution["cutoff_fill_count"] != 258
        or execution["commission_present_count"] != 258
        or execution["commission_nonzero_count"] != 0
        or float(account["lifetime_commission_account_units"]) != 0.0
        or float(instrument["long_financing_rate"]) != -0.054
        or float(instrument["short_financing_rate"]) != 0.0282
    ):
        raise RuntimeError("PROSPECTIVE_COST_BROKER_FACTS_CHANGED")

    output = output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError("PROSPECTIVE_COST_OUTPUT_EXISTS")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        policy_path = output / "policy.json"
        method_path = output / "method_receipt.json"
        manifest_path = output / "economics.fact_manifest.json"
        authority_path = output / "parameter_authority.json"
        quote = broker["executable_quote_source"]
        broker_policy = broker["market_order_no_gslo_policy"]
        policy = seal_prospective_cost_policy(
            {
                "schema_version": PROSPECTIVE_COST_POLICY_SCHEMA_VERSION,
                "decision": POLICY_DECISION,
                "policy_scope": "prospective_one_year_train_and_fixed_val_sensitivity",
                "historical_cost_truth_qualified": False,
                "economics_pass_claimed": False,
                "preregistered_at_utc": prereg.isoformat(),
                "validation_policy_selection_permitted": False,
                "broker_evidence": {
                    "path": str(broker_path),
                    "file_sha256": broker_file_sha,
                    "artifact_sha256": broker["artifact_sha256"],
                },
                "coverage": {
                    "start_utc": start.isoformat(),
                    "end_utc_exclusive": end.isoformat(),
                    "test_data_used": False,
                },
                "executable_bid_ask": {
                    "pricing_method": "side_correct_executable_bid_ask_from_bound_m1_tape",
                    **{
                        key: quote[key]
                        for key in (
                            "parquet", "manifest", "manifest_schema_version", "row_count",
                            "time_min_utc", "time_max_utc", "quote_complete_m1", "test_accessed",
                        )
                    },
                },
                "commission": {
                    "bps_per_execution": commission_bps_per_execution,
                    "zero_basis": "258_of_258_pre_cutoff_fills_zero_plus_current_account_revalidation",
                    "pre_cutoff_fill_count": execution["cutoff_fill_count"],
                    "pre_cutoff_commission_present_count": execution["commission_present_count"],
                    "pre_cutoff_commission_nonzero_count": execution["commission_nonzero_count"],
                    "current_revalidation_at_utc": broker["generated_at_utc"],
                    "current_account_snapshot_sha256": account["sanitized_snapshot_sha256"],
                    "revalidate_before_each_run": True,
                    "drift_action": "FAIL_CLOSED",
                    "implicit_zero_used": False,
                },
                "latency_slippage": {
                    "central_bps_per_execution": central_latency_slippage_bps_per_execution,
                    "direction": "adverse",
                    "application": "each_market_entry_and_each_trade_close",
                    "separate_from_bid_ask_spread": True,
                    "source_status": execution["latency_slippage_status"],
                    "parameter_origin": "explicit_conservative_nonfitted_preregistration",
                    "val_sensitivity_scenarios": [
                        {"name": name, "bps_per_execution": value}
                        for name, value in zip(
                            ("low", "central", "high"),
                            val_latency_slippage_bps_per_execution,
                            strict=True,
                        )
                    ],
                    "val_refit_permitted": False,
                    "implicit_zero_used": False,
                },
                "financing_or_swap": {
                    "long_annual_cost_rate": long_financing_annual_cost_rate,
                    "short_annual_cost_rate": short_financing_annual_cost_rate,
                    "source_long_financing_rate": float(instrument["long_financing_rate"]),
                    "source_short_financing_rate": float(instrument["short_financing_rate"]),
                    "favorable_credit_clipped_to_zero": True,
                    "accrual": "actual_elapsed_wall_clock",
                    "seconds_per_year": SECONDS_PER_YEAR,
                    "cost_formula": "open_notional*annual_cost_rate*elapsed_wall_clock_seconds/seconds_per_year",
                    "current_instrument_snapshot_sha256": instrument["sanitized_snapshot_sha256"],
                    "revalidate_before_each_run": True,
                    "drift_action": "FAIL_CLOSED",
                    "historical_rate_series_complete": False,
                    "implicit_zero_used": False,
                },
                "guaranteed_execution_fee": {
                    "account_currency_per_execution": gslo_fee_account_currency_per_execution,
                    "treatment": "structural_zero_only_under_exact_hash_bound_no_gslo_policy",
                    "market_order_no_gslo_policy_sha256": broker_policy["policy_sha256"],
                    "policy_source": broker_policy["source"],
                    "drift_action": "FAIL_CLOSED",
                    "implicit_zero_used": False,
                },
                "hold_risk_utility": {
                    "hold_risk_penalty_annual_bps_by_side": {
                        "long": hold_risk_penalty_long_annual_bps,
                        "short": hold_risk_penalty_short_annual_bps,
                    },
                    "risk_utility_formula": "annual_open_notional_penalty_bps*elapsed_wall_clock_seconds/seconds_per_year",
                    "risk_utility_is_cash_pnl": False,
                    "seconds_per_year": SECONDS_PER_YEAR,
                    "parameter_origin": "cash_net_pnl_learning_baseline_explicit_cli_zero_no_train_risk_fit",
                },
                "future_train_only_risk_sweep_required": True,
                "test_data_used": False,
            }
        )
        _write(stage / "policy.json", policy)

        method = seal_cost_method_receipt(
            {
                "schema_version": COST_METHOD_RECEIPT_SCHEMA_VERSION,
                "decision": "PASS_METHOD_COMPLETE_NOT_ECONOMICS_PASS",
                "policy_artifact_sha256": policy["artifact_sha256"],
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
        )
        _write(stage / "method_receipt.json", method)

        parameters = {
            "executable_bid_ask": {"mode": "side_correct_executable_bid_ask_from_bound_m1_tape"},
            "commission": {"bps_per_execution": 0.0},
            "execution_slippage": {"central_bps_per_execution": 2.0, "val_sensitivity_bps_per_execution": [1.0, 2.0, 4.0]},
            "financing_or_swap": {"long_annual_cost_rate": 0.054, "short_annual_cost_rate": 0.0, "favorable_credit_clipped_to_zero": True},
            "guaranteed_execution_fee": {"account_currency_per_execution": 0.0, "zero_requires_hash_bound_no_gslo_policy": True},
        }
        fact_bindings: dict[str, dict[str, str]] = {}
        for component in REQUIRED_COMPONENTS:
            final_fact_path = output / "facts" / f"{component}.fact.json"
            stage_fact_path = stage / "facts" / f"{component}.fact.json"
            source_path = policy_path
            fact = seal_economics_component_fact(
                {
                    "schema_version": ECONOMICS_FACT_SCHEMA_VERSION,
                    "decision": "PASS",
                    "component": component,
                    "verifier_schema_version": _VERIFIERS[component],
                    "coverage_start_utc": start.isoformat(),
                    "coverage_end_utc": end.isoformat(),
                    "fact_population_sha256": canonical_sha256(
                        {"component": component, "parameters": parameters[component], "policy_artifact_sha256": policy["artifact_sha256"]}
                    ),
                    "source_evidence_path": str(source_path),
                    "source_evidence_sha256": file_sha256(stage / "policy.json"),
                    "missing_values_present": False,
                    "unknown_values_present": False,
                    "implicit_zero_used": False,
                    "zero_values_source_verified": True,
                    "test_data_used": False,
                }
            )
            _write(stage_fact_path, fact)
            fact_bindings[component] = {
                "path": str(final_fact_path),
                "file_sha256": file_sha256(stage_fact_path),
                "artifact_sha256": fact["artifact_sha256"],
            }

        manifest = seal_economics_fact_manifest(
            {
                "schema_version": ECONOMICS_FACT_MANIFEST_SCHEMA_VERSION,
                "decision": "PASS",
                "coverage_start_utc": start.isoformat(),
                "coverage_end_utc": end.isoformat(),
                "component_artifacts": fact_bindings,
                "all_required_components_complete": True,
                "test_data_used": False,
            }
        )
        _write(stage / "economics.fact_manifest.json", manifest)

        authority = seal_cost_parameter_authority(
            {
                "schema_version": COST_PARAMETER_AUTHORITY_SCHEMA_VERSION,
                "decision": AUTHORITY_DECISION,
                "historical_cost_truth_qualified": False,
                "economics_pass_claimed": False,
                "preregistered_at_utc": prereg.isoformat(),
                "coverage_start_utc": start.isoformat(),
                "coverage_end_utc_exclusive": end.isoformat(),
                "policy": {
                    "path": str(policy_path),
                    "file_sha256": file_sha256(stage / "policy.json"),
                    "artifact_sha256": policy["artifact_sha256"],
                },
                "method_receipt": {
                    "path": str(method_path),
                    "file_sha256": file_sha256(stage / "method_receipt.json"),
                    "receipt_sha256": method["receipt_sha256"],
                },
                "economics_fact_manifest": {
                    "path": str(manifest_path),
                    "file_sha256": file_sha256(stage / "economics.fact_manifest.json"),
                    "manifest_sha256": manifest["manifest_sha256"],
                },
                "component_artifacts": fact_bindings,
                "parameters": parameters,
                "hold_risk_penalty_annual_bps_by_side": {"long": 0.0, "short": 0.0},
                "risk_utility_formula": "annual_open_notional_penalty_bps*elapsed_wall_clock_seconds/seconds_per_year",
                "risk_utility_is_cash_pnl": False,
                "future_train_only_risk_sweep_required": True,
                "test_data_used": False,
            }
        )
        _write(stage / "parameter_authority.json", authority)
        os.rename(stage, output)
        require_cost_parameter_authority(
            json.loads(authority_path.read_text()),
            expected_coverage_start_utc=start,
            expected_coverage_end_utc=end,
            verify_local_sources=verify_local_sources,
        )
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        if output.exists():
            shutil.rmtree(output)
        raise
    return {
        "decision": AUTHORITY_DECISION,
        "output_dir": str(output),
        "policy": _binding(policy_path, policy, "artifact_sha256"),
        "method_receipt": _binding(method_path, method, "receipt_sha256"),
        "economics_fact_manifest": _binding(manifest_path, manifest, "manifest_sha256"),
        "parameter_authority": _binding(authority_path, authority, "authority_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker-evidence", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--coverage-start-utc", required=True)
    parser.add_argument("--coverage-end-utc", required=True)
    parser.add_argument("--preregistered-at-utc", required=True)
    parser.add_argument("--commission-bps-per-execution", required=True, type=float)
    parser.add_argument("--central-latency-slippage-bps-per-execution", required=True, type=float)
    parser.add_argument("--val-latency-slippage-bps-per-execution", required=True, type=float, action="append")
    parser.add_argument("--long-financing-annual-cost-rate", required=True, type=float)
    parser.add_argument("--short-financing-annual-cost-rate", required=True, type=float)
    parser.add_argument("--gslo-fee-account-currency-per-execution", required=True, type=float)
    parser.add_argument("--hold-risk-penalty-long-annual-bps", required=True, type=float)
    parser.add_argument("--hold-risk-penalty-short-annual-bps", required=True, type=float)
    args = parser.parse_args()
    result = materialize_prospective_cost_policy(
        broker_evidence_path=args.broker_evidence,
        output_dir=args.output_dir,
        coverage_start_utc=args.coverage_start_utc,
        coverage_end_utc=args.coverage_end_utc,
        preregistered_at_utc=args.preregistered_at_utc,
        commission_bps_per_execution=args.commission_bps_per_execution,
        central_latency_slippage_bps_per_execution=args.central_latency_slippage_bps_per_execution,
        val_latency_slippage_bps_per_execution=args.val_latency_slippage_bps_per_execution,
        long_financing_annual_cost_rate=args.long_financing_annual_cost_rate,
        short_financing_annual_cost_rate=args.short_financing_annual_cost_rate,
        gslo_fee_account_currency_per_execution=args.gslo_fee_account_currency_per_execution,
        hold_risk_penalty_long_annual_bps=args.hold_risk_penalty_long_annual_bps,
        hold_risk_penalty_short_annual_bps=args.hold_risk_penalty_short_annual_bps,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
