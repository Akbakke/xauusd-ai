from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gx1.contracts import unified_exit_economics_objective_v2 as economics
from gx1.contracts.unified_exit_no_cap_economic_authority_v1 import (
    ECONOMICS_FACT_MANIFEST_SCHEMA_VERSION,
    ECONOMICS_FACT_SCHEMA_VERSION,
    REQUIRED_COMPONENTS,
    file_sha256,
    seal_economics_component_fact,
    seal_economics_fact_manifest,
)
from gx1.scripts.build_unified_exit_no_cap_authority_v1 import (
    build_no_cap_authority,
)
from gx1.scripts.materialize_unified_exit_lifecycle_v2 import (
    _load_terminal_counts,
)


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _readiness(path: Path) -> None:
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
            "fit_method": "synthetic_train_only",
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
    _write(
        path,
        {
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
        },
    )


def _facts(tmp_path: Path, *, unknown_component: str | None = None) -> Path:
    evidence = tmp_path / "literal_broker_economics.evidence"
    evidence.write_bytes(b"synthetic TRAIN broker/economic evidence\n")
    bindings = {}
    verifiers = {
        "executable_bid_ask": "gx1_executable_bid_ask_fact_verifier_v1",
        "commission": "gx1_broker_commission_fact_verifier_v1",
        "execution_slippage": "gx1_execution_slippage_fact_verifier_v1",
        "financing_or_swap": "gx1_broker_financing_fact_verifier_v1",
        "guaranteed_execution_fee": "gx1_broker_gslo_fact_verifier_v1",
    }
    for component in REQUIRED_COMPONENTS:
        fact = seal_economics_component_fact(
            {
                "schema_version": ECONOMICS_FACT_SCHEMA_VERSION,
                "decision": "PASS",
                "component": component,
                "verifier_schema_version": verifiers[component],
                "coverage_start_utc": "2024-01-01T00:00:00+00:00",
                "coverage_end_utc": "2025-01-01T00:00:00+00:00",
                "fact_population_sha256": hashlib.sha256(
                    component.encode()
                ).hexdigest(),
                "source_evidence_path": str(evidence.resolve()),
                "source_evidence_sha256": file_sha256(evidence),
                "missing_values_present": False,
                "unknown_values_present": component == unknown_component,
                "implicit_zero_used": False,
                "zero_values_source_verified": True,
                "test_data_used": False,
            }
        )
        fact_path = (tmp_path / f"{component}.fact.json").resolve()
        _write(fact_path, fact)
        bindings[component] = {
            "path": str(fact_path),
            "file_sha256": file_sha256(fact_path),
            "artifact_sha256": fact["artifact_sha256"],
        }
    manifest = seal_economics_fact_manifest(
        {
            "schema_version": ECONOMICS_FACT_MANIFEST_SCHEMA_VERSION,
            "decision": "PASS",
            "coverage_start_utc": "2024-01-01T00:00:00+00:00",
            "coverage_end_utc": "2025-01-01T00:00:00+00:00",
            "component_artifacts": bindings,
            "all_required_components_complete": True,
            "test_data_used": False,
        }
    )
    path = (tmp_path / "economics.fact_manifest.json").resolve()
    _write(path, manifest)
    return path


def test_no_cap_authority_emits_only_right_censored_lifecycles(tmp_path: Path) -> None:
    readiness = (tmp_path / "economics.readiness.json").resolve()
    _readiness(readiness)
    result = build_no_cap_authority(
        output_dir=(tmp_path / "authority").resolve(),
        dataset_run_id="pilot-one-year",
        split="train",
        entry_rows=3,
        coverage_start_utc="2024-01-01T00:00:00+00:00",
        coverage_end_utc="2025-01-01T00:00:00+00:00",
        economics_readiness_path=readiness,
        economics_fact_manifest_path=_facts(tmp_path),
        publish=True,
    )
    mapping, authority, _authority_file_sha = _load_terminal_counts(
        Path(result["authority_path"]),
        split="train",
        dataset_run_id="pilot-one-year",
        entry_rows=3,
    )
    assert set(mapping.values()) == {None}
    assert authority["no_observed_economic_terminals"] is True
    assert authority["split_end_is_right_censor"] is True
    assert authority["gap_is_right_censor"] is True
    assert authority["capacity_forces_exit"] is False
    assert authority["annual_continuous_hurdle_rate"] == pytest.approx(0.05)


@pytest.mark.parametrize("component", REQUIRED_COMPONENTS)
def test_no_cap_authority_rejects_unknown_economic_fact(
    tmp_path: Path, component: str
) -> None:
    readiness = (tmp_path / "economics.readiness.json").resolve()
    _readiness(readiness)
    with pytest.raises(RuntimeError, match="ECONOMICS_FACT_INVALID"):
        build_no_cap_authority(
            output_dir=(tmp_path / "authority").resolve(),
            dataset_run_id="pilot-one-year",
            split="train",
            entry_rows=1,
            coverage_start_utc="2024-01-01T00:00:00+00:00",
            coverage_end_utc="2025-01-01T00:00:00+00:00",
            economics_readiness_path=readiness,
            economics_fact_manifest_path=_facts(
                tmp_path, unknown_component=component
            ),
            publish=True,
        )


def test_no_cap_authority_rejects_implicit_zero(tmp_path: Path) -> None:
    readiness = (tmp_path / "economics.readiness.json").resolve()
    _readiness(readiness)
    facts_path = _facts(tmp_path)
    facts = json.loads(facts_path.read_text())
    commission_path = Path(facts["component_artifacts"]["commission"]["path"])
    commission = json.loads(commission_path.read_text())
    commission.pop("artifact_sha256")
    commission["implicit_zero_used"] = True
    _write(commission_path, seal_economics_component_fact(commission))
    facts.pop("manifest_sha256")
    facts["component_artifacts"]["commission"]["file_sha256"] = file_sha256(
        commission_path
    )
    facts["component_artifacts"]["commission"]["artifact_sha256"] = json.loads(
        commission_path.read_text()
    )["artifact_sha256"]
    _write(facts_path, seal_economics_fact_manifest(facts))
    with pytest.raises(RuntimeError, match="ECONOMICS_FACT_INVALID"):
        build_no_cap_authority(
            output_dir=(tmp_path / "authority").resolve(),
            dataset_run_id="pilot-one-year",
            split="train",
            entry_rows=1,
            coverage_start_utc="2024-01-01T00:00:00+00:00",
            coverage_end_utc="2025-01-01T00:00:00+00:00",
            economics_readiness_path=readiness,
            economics_fact_manifest_path=facts_path,
            publish=True,
        )
