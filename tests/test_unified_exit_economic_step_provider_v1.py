from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts import unified_exit_economics_objective_v2 as economics
from gx1.contracts.unified_exit_economic_step_provider_v1 import (
    COST_PARAMETER_AUTHORITY_SCHEMA_VERSION,
    COST_POLICY_SCHEMA_VERSION,
    LazyUnifiedExitEconomicStepProviderV1,
    seal_cost_parameter_authority,
    seal_unified_exit_cost_policy,
)
from gx1.contracts.unified_exit_lifecycle_v2 import unified_exit_lifecycle_v2_contract
from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import (
    ECONOMICS_FACT_MANIFEST_SCHEMA_VERSION,
    ECONOMICS_FACT_SCHEMA_VERSION,
    REQUIRED_COMPONENTS,
    canonical_sha256,
    file_sha256,
    seal_economics_component_fact,
    seal_economics_fact_manifest,
)
from gx1.scripts.materialize_unified_exit_lifecycle_v2 import (
    COMPACT_LIFECYCLE_SCHEMA_VERSION,
    _compact_pointer_stream_sha256,
    build_compact_split,
)


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _readiness() -> dict:
    hurdle = economics.seal_train_fitted_capital_hurdle_artifact(
        {
            "schema_version": economics.CAPITAL_HURDLE_SCHEMA_VERSION,
            "decision": "PASS",
            "fitted_splits": ["train"],
            "validation_or_test_used": False,
            "train_split_sha256": "1" * 64,
            "train_fold_sha256": "2" * 64,
            "source_lineage_sha256": "3" * 64,
            "annual_continuous_hurdle_rate": 0.05,
            "rate_unit": "continuous_per_wall_clock_year",
            "seconds_per_year": economics.SECONDS_PER_YEAR,
            "fit_method": "prospective_policy_receipt_v1",
            "fit_evidence_sha256": "4" * 64,
        }
    )
    objective = economics.build_unified_exit_economics_objective_contract(
        capital_hurdle_artifact=hurdle,
        expected_train_split_sha256="1" * 64,
        expected_train_fold_sha256="2" * 64,
        expected_source_lineage_sha256="3" * 64,
        policy_sha256="5" * 64,
    )
    return {
        "schema_version": "gx1_unified_exit_training_economics_readiness_v3",
        "mode": "economics_objective_v2",
        "capital_hurdle_artifact": hurdle,
        "economics_objective_contract": objective,
        "expected_train_split_sha256": "1" * 64,
        "expected_train_fold_sha256": "2" * 64,
        "expected_source_lineage_sha256": "3" * 64,
        "policy_sha256": "5" * 64,
        "proper_policy_certificate_sha256": None,
        "test_data_used": False,
    }


def _provider(tmp_path: Path):
    times = pd.date_range("2024-01-01", periods=20, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "time": times,
            "bid_open": np.linspace(100.0, 100.19, len(times)),
            "ask_open": np.linspace(100.2, 100.39, len(times)),
            "bid_close": np.linspace(100.1, 100.29, len(times)),
            "ask_close": np.linspace(100.3, 100.49, len(times)),
        },
    )
    tape_path = (tmp_path / "m1_bidask.parquet").resolve()
    frame.to_parquet(tape_path)
    tape_sha = file_sha256(tape_path)
    tape_manifest = {
        "schema_version": "gx1_direct_native_pretest_source_v2",
        "instrument": "XAU_USD",
        "timeframe": "M1",
        "timestamp_semantics": "bar_start_utc",
        "quote_complete_m1": True,
        "test_accessed": False,
        "output_parquet": str(tape_path),
        "output_parquet_sha256": tape_sha,
        "row_count": len(times),
        "test_boundary_utc": (times[-1] + pd.Timedelta(minutes=1)).isoformat(),
    }
    tape_manifest_path = (tmp_path / "m1_bidask.manifest.json").resolve()
    _write(tape_manifest_path, tape_manifest)
    compact = build_compact_split(
        entry_times=[times[0]],
        m1_times=times,
        split="train",
        split_end=times[-1] + pd.Timedelta(minutes=1),
        terminal_state_count_by_entry_side={(0, 0): None, (0, 1): None},
        m1_source_sha256=tape_sha,
        gap_classification_source_sha256="6" * 64,
        entry_binding_sha256="7" * 64,
    )
    lifecycle_manifest = {
        **unified_exit_lifecycle_v2_contract(),
        "compact_schema_version": COMPACT_LIFECYCLE_SCHEMA_VERSION,
        "split": "train",
        "split_end_utc": (times[-1] + pd.Timedelta(minutes=1)).isoformat(),
        "test_accessed": False,
        "target_q_stored": False,
        "compact_pointer_stream_sha256": _compact_pointer_stream_sha256(compact),
        "schedule_lineage_sha256": "8" * 64,
    }
    lifecycle_manifest["manifest_sha256"] = canonical_sha256(lifecycle_manifest)
    readiness = _readiness()
    values = {
        "commission": [0.3, 0.4],
        "execution_slippage": [0.5, 0.6],
        "financing_or_swap": [-120.0, 80.0],
        "guaranteed_execution_fee": [0.0, 0.0],
    }
    source = (tmp_path / "source.receipt").resolve()
    source.write_bytes(b"synthetic pretest economics source\n")
    verifiers = {
        "executable_bid_ask": "gx1_executable_bid_ask_fact_verifier_v1",
        "commission": "gx1_broker_commission_fact_verifier_v1",
        "execution_slippage": "gx1_execution_slippage_fact_verifier_v1",
        "financing_or_swap": "gx1_broker_financing_fact_verifier_v1",
        "guaranteed_execution_fee": "gx1_no_gslo_execution_policy_verifier_v1",
    }
    bindings = {}
    for component in REQUIRED_COMPONENTS:
        fact = seal_economics_component_fact(
            {
                "schema_version": ECONOMICS_FACT_SCHEMA_VERSION,
                "decision": "PASS",
                "component": component,
                "verifier_schema_version": verifiers[component],
                "coverage_start_utc": times[0].isoformat(),
                "coverage_end_utc": (times[-1] + pd.Timedelta(minutes=1)).isoformat(),
                "fact_population_sha256": canonical_sha256({"component": component}),
                "source_evidence_path": str(source),
                "source_evidence_sha256": file_sha256(source),
                "missing_values_present": False,
                "unknown_values_present": False,
                "implicit_zero_used": False,
                "zero_values_source_verified": True,
                "test_data_used": False,
            }
        )
        path = (tmp_path / f"{component}.fact.json").resolve()
        _write(path, fact)
        bindings[component] = {
            "path": str(path),
            "file_sha256": file_sha256(path),
            "artifact_sha256": fact["artifact_sha256"],
        }
    facts = seal_economics_fact_manifest(
        {
            "schema_version": ECONOMICS_FACT_MANIFEST_SCHEMA_VERSION,
            "decision": "PASS",
            "coverage_start_utc": times[0].isoformat(),
            "coverage_end_utc": (times[-1] + pd.Timedelta(minutes=1)).isoformat(),
            "component_artifacts": bindings,
            "all_required_components_complete": True,
            "test_data_used": False,
        }
    )
    facts_path = (tmp_path / "facts.manifest.json").resolve()
    _write(facts_path, facts)
    source_method = (tmp_path / "cost_method.receipt").resolve()
    source_method.write_bytes(b"synthetic train-only cost parameter method\n")
    parameter_authority = seal_cost_parameter_authority(
        {
            "schema_version": COST_PARAMETER_AUTHORITY_SCHEMA_VERSION,
            "decision": "PASS",
            "method": "train_only_source_method_receipt_v1",
            "split": "train",
            "coverage_start_utc": times[0].isoformat(),
            "coverage_end_utc": (times[-1] + pd.Timedelta(minutes=1)).isoformat(),
            "component_artifact_sha256": {
                key: value["artifact_sha256"] for key, value in bindings.items()
            },
            "commission_total_bps_by_side": values["commission"],
            "slippage_total_bps_by_side": values["execution_slippage"],
            "financing_annual_bps_by_side": values["financing_or_swap"],
            "guaranteed_execution_fee_total_bps_by_side": values[
                "guaranteed_execution_fee"
            ],
            "hold_risk_penalty_annual_bps_by_side": [10.0, 20.0],
            "risk_utility_formula": (
                "annual_open_notional_penalty_bps*elapsed_wall_clock_seconds/seconds_per_year"
            ),
            "risk_utility_is_cash_pnl": False,
            "source_method_receipt_path": str(source_method),
            "source_method_receipt_sha256": file_sha256(source_method),
            "validation_or_test_used": False,
            "test_data_used": False,
        }
    )
    parameter_path = (tmp_path / "cost_parameters.json").resolve()
    _write(parameter_path, parameter_authority)
    policy = seal_unified_exit_cost_policy(
        {
            "schema_version": COST_POLICY_SCHEMA_VERSION,
            "decision": "PASS",
            "instrument": "XAU_USD",
            "split": "train",
            "coverage_start_utc": times[0].isoformat(),
            "coverage_end_utc": (times[-1] + pd.Timedelta(minutes=1)).isoformat(),
            "pretest_only": True,
            "lifecycle_manifest_sha256": lifecycle_manifest["manifest_sha256"],
            "economics_objective_contract_sha256": readiness[
                "economics_objective_contract"
            ]["contract_sha256"],
            "m1_tape_path": str(tape_path),
            "m1_tape_sha256": tape_sha,
            "m1_tape_manifest_path": str(tape_manifest_path),
            "m1_tape_manifest_sha256": file_sha256(tape_manifest_path),
            "economics_fact_manifest_path": str(facts_path),
            "economics_fact_manifest_sha256": file_sha256(facts_path),
            "economics_fact_manifest_identity_sha256": facts["manifest_sha256"],
            "cost_parameter_authority_path": str(parameter_path),
            "cost_parameter_authority_sha256": file_sha256(parameter_path),
            "cost_parameter_authority_identity_sha256": parameter_authority[
                "authority_sha256"
            ],
            "entry_quote_field_by_side": ["ask_open", "bid_open"],
            "exit_quote_field_by_side": ["bid_close", "ask_close"],
            "spread_cost_handling": "embedded_once_in_entry_and_exit_executable_quotes",
            "commission_total_bps_by_side": values["commission"],
            "slippage_total_bps_by_side": values["execution_slippage"],
            "financing_annual_bps_by_side": values["financing_or_swap"],
            "guaranteed_execution_fee_total_bps_by_side": values[
                "guaranteed_execution_fee"
            ],
            "hold_risk_penalty_annual_bps_by_side": [10.0, 20.0],
            "risk_utility_formula": (
                "annual_open_notional_penalty_bps*elapsed_wall_clock_seconds/seconds_per_year"
            ),
            "risk_utility_is_cash_pnl": False,
            "component_artifact_sha256": {
                key: value["artifact_sha256"] for key, value in bindings.items()
            },
            "gap_source_manifest_sha256": "9" * 64,
            "declared_market_closure_intervals": [],
            "same_capital_hurdle_running_cost_bps": 0.0,
            "test_data_used": False,
        }
    )
    policy_path = (tmp_path / "cost_policy.json").resolve()
    _write(policy_path, policy)
    provider = LazyUnifiedExitEconomicStepProviderV1(
        compact_rows=compact,
        compact_manifest=lifecycle_manifest,
        economics_readiness=readiness,
        cost_policy_path=policy_path,
    )
    return provider, compact, lifecycle_manifest, readiness, frame, policy_path


def test_lazy_provider_uses_executable_quotes_and_wall_clock_costs(
    tmp_path: Path,
) -> None:
    provider, compact, _manifest, readiness, frame, _ = _provider(tmp_path)
    start = int(compact.iloc[0]["entry_m1_start_row"])
    long_exit = provider(0, 0, "exit_now", 0, 1)["steps"][0]
    short_exit = provider(0, 1, "exit_now", 0, 1)["steps"][0]
    assert long_exit["gross_price_cashflow"]["value_bps"] == pytest.approx(
        (frame.iloc[start].bid_close - frame.iloc[start].ask_open)
        / frame.iloc[start].ask_open
        * 10_000.0
    )
    assert short_exit["gross_price_cashflow"]["value_bps"] == pytest.approx(
        (frame.iloc[start].bid_open - frame.iloc[start].ask_close)
        / frame.iloc[start].bid_open
        * 10_000.0
    )
    assert long_exit["commission"]["value_bps"] == 0.3
    assert long_exit["execution_slippage"]["value_bps"] == 0.5
    assert long_exit["financing_or_swap"]["value_bps"] == 0.0
    hold = provider(0, 0, "hold", 0, 1)["steps"][0]
    assert (
        hold["interval_end_time_ns"] - hold["interval_start_time_ns"] == 60_000_000_000
    )
    assert hold["financing_or_swap"]["value_bps"] == pytest.approx(
        -120.0 * 60.0 / economics.SECONDS_PER_YEAR
    )
    assert hold["same_capital_hurdle_running_cost"]["value_bps"] == 0.0
    assert (
        provider.economic_exit_step_manifest["economics_objective_contract_sha256"]
        == readiness["economics_objective_contract"]["contract_sha256"]
    )


def test_provider_rejects_unbound_policy_value(tmp_path: Path) -> None:
    (
        _provider_instance,
        compact,
        lifecycle_manifest,
        readiness,
        _frame,
        policy_path,
    ) = _provider(tmp_path)
    policy = json.loads(policy_path.read_text())
    policy.pop("policy_sha256")
    policy["slippage_total_bps_by_side"] = [9.0, 9.0]
    _write(policy_path, seal_unified_exit_cost_policy(policy))
    with pytest.raises(RuntimeError, match="VALUE_NOT_SOURCE_BOUND"):
        LazyUnifiedExitEconomicStepProviderV1(
            compact_rows=compact,
            compact_manifest=lifecycle_manifest,
            economics_readiness=readiness,
            cost_policy_path=policy_path,
        )
