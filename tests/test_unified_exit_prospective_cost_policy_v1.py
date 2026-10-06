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

from tests.test_unified_exit_broker_evidence_v1 import prospective_broker_fixture
START = "2025-06-01T00:00:00+00:00"
END = "2026-07-01T00:00:00+00:00"


def _build(tmp_path: Path, **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "broker_evidence_path": prospective_broker_fixture(tmp_path),
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



def test_invalid_quote_coverage_never_publishes(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="EXECUTABLE_QUOTES_INVALID"):
        _build(tmp_path, coverage_start_utc="2011-06-01T00:00:00+00:00")
    assert not (tmp_path / "policy_bundle").exists()
    assert len(list(tmp_path.glob(".policy_bundle.staging.*"))) == 1


@pytest.mark.parametrize("with_sentinel", [False, True])
def test_late_output_collision_preserves_existing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_sentinel: bool,
) -> None:
    from gx1.scripts import materialize_unified_exit_prospective_cost_policy_v1 as producer
    publish = producer._publish_file_noreplace
    output = tmp_path / "policy_bundle"

    def collide(stage: Path, destination: Path) -> None:
        destination.mkdir()
        if with_sentinel:
            (destination / "keep").write_bytes(b"other attempt")
        publish(stage, destination)

    monkeypatch.setattr(producer, "_publish_file_noreplace", collide)
    with pytest.raises(RuntimeError, match="already exists"):
        _build(tmp_path)
    assert output.is_dir()
    assert sorted(p.name for p in output.iterdir()) == (["keep"] if with_sentinel else [])
    if with_sentinel:
        assert (output / "keep").read_bytes() == b"other attempt"


@pytest.mark.parametrize("defect", ["byte_drift", "extra_file", "symlink"])
def test_staging_defects_fail_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str,
) -> None:
    from gx1.scripts import materialize_unified_exit_prospective_cost_policy_v1 as producer
    write = producer._write

    def corrupt(path: Path, value: object) -> None:
        write(path, value)
        if path.name == "parameter_authority.json":
            if defect == "byte_drift":
                policy = path.parent / "policy.json"
                policy.write_bytes(policy.read_bytes() + b" ")
            elif defect == "extra_file":
                (path.parent / "unbound.json").write_text("{}")
            else:
                fact = path.parent / "facts/commission.fact.json"
                saved = tmp_path / "outside.fact.json"
                fact.rename(saved)
                fact.symlink_to(saved)

    monkeypatch.setattr(producer, "_write", corrupt)
    with pytest.raises(RuntimeError, match="SOURCE_INVALID|STAGED_INVENTORY_INVALID"):
        _build(tmp_path)
    assert not (tmp_path / "policy_bundle").exists()
    assert len(list(tmp_path.glob(".policy_bundle.staging.*"))) == 1


def test_full_strict_load_precedes_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.scripts import materialize_unified_exit_prospective_cost_policy_v1 as producer
    require = producer.require_cost_parameter_authority
    calls = []

    def inspect(*args, **kwargs):
        assert not (tmp_path / "policy_bundle").exists()
        assert kwargs["_staged_files"]
        checked = require(*args, **kwargs)
        calls.append(checked["authority_sha256"])
        return checked

    monkeypatch.setattr(producer, "require_cost_parameter_authority", inspect)
    result = _build(tmp_path)
    assert calls == [result["parameter_authority"]["authority_sha256"]]


def test_postpublication_fsync_failure_never_deletes_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.scripts import materialize_unified_exit_prospective_cost_policy_v1 as producer
    fsync = producer._fsync_directory

    def fail_parent(path: Path) -> None:
        if path == tmp_path:
            raise OSError("injected directory fsync failure")
        fsync(path)

    monkeypatch.setattr(producer, "_fsync_directory", fail_parent)
    with pytest.raises(OSError, match="injected"):
        _build(tmp_path)
    authority = json.loads((tmp_path / "policy_bundle/parameter_authority.json").read_text())
    require_cost_parameter_authority(
        authority, expected_coverage_start_utc=START, expected_coverage_end_utc=END,
        verify_local_sources=False,
    )


def _broker_with_financing(tmp_path: Path, long_rate: float, short_rate: float) -> Path:
    from gx1.contracts.unified_exit_broker_evidence_v1 import (
        _canonical_sha256, seal_unified_exit_broker_evidence_v1,
    )
    broker = json.loads(prospective_broker_fixture(tmp_path).read_text())
    broker.pop("artifact_sha256")
    instrument = broker["current_prospective_terms"]["instrument"]
    instrument.pop("sanitized_snapshot_sha256")
    instrument.update(long_financing_rate=long_rate, short_financing_rate=short_rate)
    instrument["sanitized_snapshot_sha256"] = _canonical_sha256(instrument)
    broker = seal_unified_exit_broker_evidence_v1(broker)
    path = tmp_path / "synthetic_changed_terms.json"
    path.write_text(json.dumps(broker))
    return path


@pytest.mark.parametrize(
    ("long_rate", "short_rate", "long_cost", "short_cost"),
    [(-0.0569, 0.0323, 0.0569, 0.0),
     (-0.061, -0.023, 0.061, 0.023),
     (0.015, 0.01, 0.0, 0.0),
     (0.0, 0.0, 0.0, 0.0)],
)
def test_financing_costs_follow_exact_bound_terms(
    tmp_path: Path, long_rate: float, short_rate: float,
    long_cost: float, short_cost: float,
) -> None:
    broker = _broker_with_financing(tmp_path, long_rate, short_rate)
    result = _build(tmp_path, broker_evidence_path=broker,
                    long_financing_annual_cost_rate=long_cost,
                    short_financing_annual_cost_rate=short_cost)
    policy = _load(result, "policy")
    rates = policy["financing_or_swap"]
    assert rates["source_long_financing_rate"] == long_rate
    assert rates["source_short_financing_rate"] == short_rate
    assert rates["long_annual_cost_rate"] == long_cost
    assert rates["short_annual_cost_rate"] == short_cost
    assert rates["favorable_credit_clipped_to_zero"] is True
    authority = require_cost_parameter_authority(
        _load(result, "parameter_authority"),
        expected_coverage_start_utc=START, expected_coverage_end_utc=END,
        verify_local_sources=False,
    )
    assert authority["parameters"]["financing_or_swap"] == {
        "long_annual_cost_rate": long_cost, "short_annual_cost_rate": short_cost,
        "favorable_credit_clipped_to_zero": True,
    }
    from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import canonical_sha256
    fact_path = authority["component_artifacts"]["financing_or_swap"]["path"]
    fact = json.loads(Path(fact_path).read_text())
    assert fact["fact_population_sha256"] == canonical_sha256({
        "component": "financing_or_swap",
        "parameters": authority["parameters"]["financing_or_swap"],
        "policy_artifact_sha256": policy["artifact_sha256"],
    })


def test_old_financing_cli_cannot_admit_current_terms(tmp_path: Path) -> None:
    broker = _broker_with_financing(tmp_path, -0.0569, 0.0323)
    with pytest.raises(RuntimeError, match="EXACT_PREREGISTERED_CLI"):
        _build(tmp_path, broker_evidence_path=broker)
    assert not (tmp_path / "policy_bundle").exists()
    assert not list(tmp_path.glob(".policy_bundle.staging.*"))


def test_resealed_policy_cannot_understate_observed_financing(tmp_path: Path) -> None:
    broker = _broker_with_financing(tmp_path, -0.0569, 0.0323)
    result = _build(tmp_path, broker_evidence_path=broker,
                    long_financing_annual_cost_rate=0.0569)
    policy = _load(result, "policy")
    policy.pop("artifact_sha256")
    policy["financing_or_swap"]["long_annual_cost_rate"] = 0.054
    policy = seal_prospective_cost_policy(policy)
    with pytest.raises(RuntimeError, match="FINANCING_INVALID"):
        require_prospective_cost_policy(
            policy, expected_coverage_start_utc=START,
            expected_coverage_end_utc=END, verify_local_sources=False,
        )


def test_resealed_authority_cannot_reuse_old_financing(tmp_path: Path) -> None:
    from gx1.contracts.unified_exit_prospective_cost_policy_v1 import seal_cost_parameter_authority
    broker = _broker_with_financing(tmp_path, -0.0569, 0.0323)
    result = _build(tmp_path, broker_evidence_path=broker,
                    long_financing_annual_cost_rate=0.0569)
    authority = _load(result, "parameter_authority")
    authority.pop("authority_sha256")
    authority["parameters"]["financing_or_swap"]["long_annual_cost_rate"] = 0.054
    authority = seal_cost_parameter_authority(authority)
    with pytest.raises(RuntimeError, match="AUTHORITY_PARAMETERS_INVALID"):
        require_cost_parameter_authority(
            authority, expected_coverage_start_utc=START,
            expected_coverage_end_utc=END, verify_local_sources=False,
        )
