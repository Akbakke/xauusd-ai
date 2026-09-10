"""No-duration-cap economic lifecycle authority for rho-contractive Exit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from gx1.contracts.unified_exit_fitted_q_v1 import (
    require_unified_exit_unbounded_training_readiness,
)


NO_CAP_AUTHORITY_SCHEMA_VERSION = "gx1_unified_exit_no_cap_authority_v1"
ECONOMICS_FACT_MANIFEST_SCHEMA_VERSION = "gx1_exit_economics_fact_manifest_v1"
ECONOMICS_FACT_SCHEMA_VERSION = "gx1_exit_economics_component_fact_v1"
NO_CAP_TERMINAL_VERIFIER_SCHEMA_VERSION = (
    "gx1_rho_contractive_no_terminal_event_verifier_v1"
)
REQUIRED_COMPONENTS = (
    "executable_bid_ask",
    "commission",
    "execution_slippage",
    "financing_or_swap",
    "guaranteed_execution_fee",
)
_VERIFIERS = {
    "executable_bid_ask": {"gx1_executable_bid_ask_fact_verifier_v1"},
    "commission": {"gx1_broker_commission_fact_verifier_v1"},
    "execution_slippage": {"gx1_execution_slippage_fact_verifier_v1"},
    "financing_or_swap": {"gx1_broker_financing_fact_verifier_v1"},
    "guaranteed_execution_fee": {
        "gx1_broker_gslo_fact_verifier_v1",
        "gx1_no_gslo_execution_policy_verifier_v1",
    },
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"UNIFIED_EXIT_NO_CAP_{label}_SHA_INVALID")
    return value


def _utc(value: Any, label: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed) or parsed.tz is None or parsed.utcoffset() != pd.Timedelta(0):
        raise RuntimeError(f"UNIFIED_EXIT_NO_CAP_{label}_INVALID")
    return parsed.as_unit("ns")


def seal_economics_component_fact(value: Mapping[str, Any]) -> dict[str, Any]:
    observed = dict(value)
    if "artifact_sha256" in observed:
        raise RuntimeError("UNIFIED_EXIT_NO_CAP_FACT_ALREADY_SEALED")
    observed["artifact_sha256"] = canonical_sha256(observed)
    return observed


def require_economics_component_fact(
    value: Mapping[str, Any],
    *,
    component: str,
    expected_coverage_start_utc: Any,
    expected_coverage_end_utc: Any,
) -> dict[str, Any]:
    keys = {
        "schema_version", "decision", "component", "verifier_schema_version",
        "coverage_start_utc", "coverage_end_utc", "fact_population_sha256",
        "source_evidence_path", "source_evidence_sha256", "missing_values_present",
        "unknown_values_present", "implicit_zero_used", "zero_values_source_verified",
        "test_data_used", "artifact_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeError("UNIFIED_EXIT_NO_CAP_ECONOMICS_FACT_INVALID")
    observed = dict(value)
    source_path = Path(str(observed["source_evidence_path"] or ""))
    start = _utc(observed["coverage_start_utc"], "FACT_COVERAGE_START")
    end = _utc(observed["coverage_end_utc"], "FACT_COVERAGE_END")
    expected_start = _utc(expected_coverage_start_utc, "EXPECTED_COVERAGE_START")
    expected_end = _utc(expected_coverage_end_utc, "EXPECTED_COVERAGE_END")
    if (
        component not in REQUIRED_COMPONENTS
        or observed["schema_version"] != ECONOMICS_FACT_SCHEMA_VERSION
        or observed["decision"] != "PASS"
        or observed["component"] != component
        or observed["verifier_schema_version"] not in _VERIFIERS[component]
        or start > expected_start
        or end < expected_end
        or end <= start
        or not source_path.is_absolute()
        or not source_path.is_file()
        or file_sha256(source_path) != observed["source_evidence_sha256"]
        or observed["missing_values_present"] is not False
        or observed["unknown_values_present"] is not False
        or observed["implicit_zero_used"] is not False
        or observed["zero_values_source_verified"] is not True
        or observed["test_data_used"] is not False
        or observed["artifact_sha256"]
        != canonical_sha256({k: v for k, v in observed.items() if k != "artifact_sha256"})
    ):
        raise RuntimeError("UNIFIED_EXIT_NO_CAP_ECONOMICS_FACT_INVALID")
    for key in ("fact_population_sha256", "source_evidence_sha256", "artifact_sha256"):
        _sha(observed[key], key.upper())
    return observed


def seal_economics_fact_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    observed = dict(value)
    if "manifest_sha256" in observed:
        raise RuntimeError("UNIFIED_EXIT_NO_CAP_FACT_MANIFEST_ALREADY_SEALED")
    observed["manifest_sha256"] = canonical_sha256(observed)
    return observed


def require_economics_fact_manifest(
    value: Mapping[str, Any],
    *,
    expected_coverage_start_utc: Any,
    expected_coverage_end_utc: Any,
) -> dict[str, Any]:
    keys = {
        "schema_version", "decision", "coverage_start_utc", "coverage_end_utc",
        "component_artifacts", "all_required_components_complete", "test_data_used",
        "manifest_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeError("UNIFIED_EXIT_NO_CAP_FACT_MANIFEST_INVALID")
    observed = dict(value)
    components = observed["component_artifacts"]
    if (
        observed["schema_version"] != ECONOMICS_FACT_MANIFEST_SCHEMA_VERSION
        or observed["decision"] != "PASS"
        or observed["all_required_components_complete"] is not True
        or observed["test_data_used"] is not False
        or not isinstance(components, Mapping)
        or set(components) != set(REQUIRED_COMPONENTS)
        or observed["manifest_sha256"]
        != canonical_sha256({k: v for k, v in observed.items() if k != "manifest_sha256"})
        or _utc(observed["coverage_start_utc"], "MANIFEST_COVERAGE_START")
        > _utc(expected_coverage_start_utc, "EXPECTED_COVERAGE_START")
        or _utc(observed["coverage_end_utc"], "MANIFEST_COVERAGE_END")
        < _utc(expected_coverage_end_utc, "EXPECTED_COVERAGE_END")
    ):
        raise RuntimeError("UNIFIED_EXIT_NO_CAP_FACT_MANIFEST_INVALID")
    for component in REQUIRED_COMPONENTS:
        binding = components[component]
        if not isinstance(binding, Mapping) or set(binding) != {
            "path", "file_sha256", "artifact_sha256"
        }:
            raise RuntimeError("UNIFIED_EXIT_NO_CAP_FACT_MANIFEST_INVALID")
        path = Path(str(binding["path"] or ""))
        if (
            not path.is_absolute()
            or not path.is_file()
            or file_sha256(path) != binding["file_sha256"]
        ):
            raise RuntimeError("UNIFIED_EXIT_NO_CAP_FACT_MANIFEST_INVALID")
        fact = json.loads(path.read_text(encoding="utf-8"))
        checked = require_economics_component_fact(
            fact,
            component=component,
            expected_coverage_start_utc=expected_coverage_start_utc,
            expected_coverage_end_utc=expected_coverage_end_utc,
        )
        if checked["artifact_sha256"] != binding["artifact_sha256"]:
            raise RuntimeError("UNIFIED_EXIT_NO_CAP_FACT_MANIFEST_INVALID")
    return observed


def seal_no_cap_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    observed = dict(value)
    if "authority_sha256" in observed:
        raise RuntimeError("UNIFIED_EXIT_NO_CAP_AUTHORITY_ALREADY_SEALED")
    observed["authority_sha256"] = canonical_sha256(observed)
    return observed


def require_no_cap_authority(
    value: Mapping[str, Any],
    *,
    expected_split: str,
    expected_dataset_run_id: str,
    expected_terminal_state_counts_sha256: str,
) -> dict[str, Any]:
    keys = {
        "schema_version", "decision", "mode", "dataset_run_id", "split",
        "coverage_start_utc", "coverage_end_utc",
        "authority_artifact_path", "authority_artifact_sha256",
        "terminal_state_counts_sha256", "economic_terminal_definition_sha256",
        "terminal_event_verifier_schema_version", "no_observed_economic_terminals",
        "economics_readiness_path", "economics_readiness_sha256",
        "economics_objective_contract_sha256", "annual_continuous_hurdle_rate",
        "economics_fact_manifest_path", "economics_fact_manifest_sha256",
        "split_end_is_right_censor", "gap_is_right_censor", "capacity_forces_exit",
        "terminal_events_recomputed_from_train_val_only", "test_data_used",
        "authority_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeError("UNIFIED_EXIT_NO_CAP_AUTHORITY_INVALID")
    observed = dict(value)
    if (
        observed["schema_version"] != NO_CAP_AUTHORITY_SCHEMA_VERSION
        or observed["decision"] != "PASS"
        or observed["mode"] != "rho_contractive_no_duration_cap"
        or observed["dataset_run_id"] != expected_dataset_run_id
        or observed["split"] != expected_split
        or _utc(observed["coverage_end_utc"], "AUTHORITY_COVERAGE_END")
        <= _utc(observed["coverage_start_utc"], "AUTHORITY_COVERAGE_START")
        or observed["terminal_state_counts_sha256"] != expected_terminal_state_counts_sha256
        or observed["terminal_event_verifier_schema_version"]
        != NO_CAP_TERMINAL_VERIFIER_SCHEMA_VERSION
        or observed["no_observed_economic_terminals"] is not True
        or float(observed["annual_continuous_hurdle_rate"]) <= 0.0
        or observed["split_end_is_right_censor"] is not True
        or observed["gap_is_right_censor"] is not True
        or observed["capacity_forces_exit"] is not False
        or observed["terminal_events_recomputed_from_train_val_only"] is not True
        or observed["test_data_used"] is not False
        or observed["authority_sha256"]
        != canonical_sha256({k: v for k, v in observed.items() if k != "authority_sha256"})
    ):
        raise RuntimeError("UNIFIED_EXIT_NO_CAP_AUTHORITY_INVALID")
    for key in (
        "authority_artifact_sha256", "terminal_state_counts_sha256",
        "economic_terminal_definition_sha256", "economics_readiness_sha256",
        "economics_objective_contract_sha256", "economics_fact_manifest_sha256",
        "authority_sha256",
    ):
        _sha(observed[key], key.upper())
    return observed


def require_no_cap_authority_sources(value: Mapping[str, Any]) -> dict[str, Any]:
    """Re-open and verify readiness and all economic fact source bytes."""

    observed = dict(value)
    readiness_path = Path(str(observed["economics_readiness_path"] or ""))
    facts_path = Path(str(observed["economics_fact_manifest_path"] or ""))
    counts_path = Path(str(observed["authority_artifact_path"] or ""))
    for path, key in (
        (readiness_path, "economics_readiness_sha256"),
        (facts_path, "economics_fact_manifest_sha256"),
        (counts_path, "authority_artifact_sha256"),
    ):
        if not path.is_absolute() or not path.is_file() or file_sha256(path) != observed[key]:
            raise RuntimeError("UNIFIED_EXIT_NO_CAP_AUTHORITY_SOURCE_INVALID")
    readiness = require_unified_exit_unbounded_training_readiness(
        json.loads(readiness_path.read_text(encoding="utf-8")),
        context="UNIFIED_EXIT_NO_CAP_AUTHORITY",
    )
    if (
        readiness["economics_objective_contract"]["contract_sha256"]
        != observed["economics_objective_contract_sha256"]
        or readiness["validated_annual_continuous_hurdle_rate"]
        != float(observed["annual_continuous_hurdle_rate"])
    ):
        raise RuntimeError("UNIFIED_EXIT_NO_CAP_AUTHORITY_SOURCE_INVALID")
    require_economics_fact_manifest(
        json.loads(facts_path.read_text(encoding="utf-8")),
        expected_coverage_start_utc=observed["coverage_start_utc"],
        expected_coverage_end_utc=observed["coverage_end_utc"],
    )
    return observed


__all__ = (
    "ECONOMICS_FACT_MANIFEST_SCHEMA_VERSION", "ECONOMICS_FACT_SCHEMA_VERSION",
    "NO_CAP_AUTHORITY_SCHEMA_VERSION", "NO_CAP_TERMINAL_VERIFIER_SCHEMA_VERSION",
    "REQUIRED_COMPONENTS", "canonical_sha256", "file_sha256",
    "require_economics_component_fact", "require_economics_fact_manifest",
    "require_no_cap_authority", "require_no_cap_authority_sources",
    "seal_economics_component_fact", "seal_economics_fact_manifest",
    "seal_no_cap_authority",
)
