from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.contracts.unified_exit_prospective_cost_policy_v1 import (
    require_cost_parameter_authority,
    require_prospective_cost_policy,
    seal_prospective_cost_policy,
)
from gx1.scripts.materialize_unified_exit_prospective_cost_policy_v1 import (
    materialize_prospective_cost_policy,
)

ROOT = Path(__file__).resolve().parents[1]
BROKER = ROOT / "docs/evidence/UNIFIED_EXIT_BROKER_EVIDENCE_V1_20260910.json"
START = "2025-06-01T00:00:00+00:00"
END = "2026-07-01T00:00:00+00:00"


def _build(tmp_path: Path, **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "broker_evidence_path": BROKER,
        "output_dir": tmp_path / "policy_bundle",
        "coverage_start_utc": START,
        "coverage_end_utc": END,
        "preregistered_at_utc": "2026-09-10T22:30:00+00:00",
        "commission_bps_per_execution": 0.0,
        "central_latency_slippage_bps_per_execution": 2.0,
        "val_latency_slippage_bps_per_execution": [1.0, 2.0, 4.0],
        "long_financing_annual_cost_rate": 0.054,
        "short_financing_annual_cost_rate": 0.0,
        "gslo_fee_account_currency_per_execution": 0.0,
        "hold_risk_penalty_long_annual_bps": 0.0,
        "hold_risk_penalty_short_annual_bps": 0.0,
        "verify_local_sources": False,
    }
    kwargs.update(overrides)
    return materialize_prospective_cost_policy(**kwargs)  # type: ignore[arg-type]


def _load(result: dict[str, object], name: str) -> dict[str, object]:
    binding = result[name]
    assert isinstance(binding, dict)
    return json.loads(Path(str(binding["path"])).read_text())


def test_valid_preregistered_bundle_and_fact_manifest(tmp_path: Path) -> None:
    result = _build(tmp_path)
    authority = _load(result, "parameter_authority")
    checked = require_cost_parameter_authority(
        authority,
        expected_coverage_start_utc=START,
        expected_coverage_end_utc=END,
        verify_local_sources=False,
    )
    assert checked["economics_pass_claimed"] is False
    assert checked["historical_cost_truth_qualified"] is False
    assert checked["parameters"]["execution_slippage"][
        "val_sensitivity_bps_per_execution"
    ] == [1.0, 2.0, 4.0]
    assert checked["hold_risk_penalty_annual_bps_by_side"] == {
        "long": 0.0,
        "short": 0.0,
    }


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("central_latency_slippage_bps_per_execution", 1.5),
        ("val_latency_slippage_bps_per_execution", [1.0, 2.0, 3.0]),
        ("commission_bps_per_execution", 0.1),
        ("long_financing_annual_cost_rate", 0.05),
        ("short_financing_annual_cost_rate", -0.0282),
        ("hold_risk_penalty_long_annual_bps", 1.0),
    ],
)
def test_cli_values_are_exactly_preregistered(
    tmp_path: Path, key: str, value: object
) -> None:
    with pytest.raises(RuntimeError, match="EXACT_PREREGISTERED_CLI"):
        _build(tmp_path, **{key: value})


def test_resealed_historical_truth_claim_fails(tmp_path: Path) -> None:
    result = _build(tmp_path)
    policy = _load(result, "policy")
    policy.pop("artifact_sha256")
    policy["historical_cost_truth_qualified"] = True
    policy = seal_prospective_cost_policy(policy)
    with pytest.raises(RuntimeError, match="HEADER_INVALID"):
        require_prospective_cost_policy(
            policy,
            expected_coverage_start_utc=START,
            expected_coverage_end_utc=END,
            verify_local_sources=False,
        )


def test_resealed_val_grid_selection_fails(tmp_path: Path) -> None:
    result = _build(tmp_path)
    policy = _load(result, "policy")
    policy.pop("artifact_sha256")
    policy["latency_slippage"]["val_sensitivity_scenarios"][2][
        "bps_per_execution"
    ] = 3.0
    policy = seal_prospective_cost_policy(policy)
    with pytest.raises(RuntimeError, match="SLIPPAGE_INVALID"):
        require_prospective_cost_policy(
            policy,
            expected_coverage_start_utc=START,
            expected_coverage_end_utc=END,
            verify_local_sources=False,
        )


def test_bound_component_byte_drift_fails(tmp_path: Path) -> None:
    result = _build(tmp_path)
    authority = _load(result, "parameter_authority")
    fact = Path(
        authority["component_artifacts"]["execution_slippage"]["path"]  # type: ignore[index]
    )
    fact.write_bytes(fact.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="FACT_MANIFEST_INVALID"):
        require_cost_parameter_authority(
            authority,
            expected_coverage_start_utc=START,
            expected_coverage_end_utc=END,
            verify_local_sources=False,
        )


def test_resealed_nonzero_risk_penalty_is_not_this_baseline(tmp_path: Path) -> None:
    result = _build(tmp_path)
    policy = _load(result, "policy")
    policy.pop("artifact_sha256")
    policy["hold_risk_utility"]["hold_risk_penalty_annual_bps_by_side"]["long"] = 1.0
    policy = seal_prospective_cost_policy(policy)
    with pytest.raises(RuntimeError, match="RISK_INVALID"):
        require_prospective_cost_policy(
            policy,
            expected_coverage_start_utc=START,
            expected_coverage_end_utc=END,
            verify_local_sources=False,
        )
