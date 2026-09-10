"""Fail-closed economics and objective semantics for unbounded unified Exit.

The economic lifetime is independent of the bounded model transport chunk.
Cash PnL is preserved undiscounted.  The training utility applies an immutable,
TRAIN-fitted continuous capital hurdle through elapsed-wall-clock discounting
and keeps risk utility as a separately reported non-cash penalty.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


ECONOMICS_OBJECTIVE_SCHEMA_VERSION = "gx1_unified_exit_economics_objective_v2"
CAPITAL_HURDLE_SCHEMA_VERSION = "gx1_exit_capital_hurdle_train_fit_v1"
FROZEN_CAPITAL_HURDLE_SCHEMA_VERSION = "gx1_exit_capital_hurdle_frozen_owner_v2"
PROPER_POLICY_CERTIFICATE_SCHEMA_VERSION = "gx1_exit_absorbing_policy_certificate_v1"
ECONOMIC_STEP_SCHEMA_VERSION = "gx1_exit_economic_step_v1"
ECONOMIC_PATH_SCHEMA_VERSION = "gx1_exit_economic_path_objective_v1"
SECONDS_PER_YEAR = 31_557_600.0
NOMINAL_M1_SECONDS = 60
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_COST_COMPONENTS = (
    "commission",
    "execution_slippage",
    "financing_or_swap",
    "guaranteed_execution_fee",
)
_NONNEGATIVE_COMPONENTS = frozenset(
    {
        "commission",
        "execution_slippage",
        "guaranteed_execution_fee",
        "risk_utility_penalty",
    }
)
_EVENT_KINDS = frozenset({"ENTRY", "HOLD", "EXIT_NOW", "ECONOMIC_TERMINAL"})
_OBJECTIVE_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "target_unit",
        "policy_sha256",
        "capital_hurdle_artifact_sha256",
        "train_split_sha256",
        "train_fold_sha256",
        "source_lineage_sha256",
        "annual_continuous_hurdle_rate",
        "seconds_per_year",
        "discount_factor",
        "capacity_or_chunk_length_affects_gamma",
        "same_capital_hurdle_running_cost_allowed",
        "capital_hurdle_double_counting_forbidden",
        "undiscounted_net_cash_pnl_is_separate",
        "discounted_risk_utility_is_training_objective",
        "risk_utility_penalty_is_cash_pnl",
        "required_complete_cost_inputs",
        "explicit_gap_classification_required",
        "right_censor_is_terminal",
        "validation_or_test_may_fit_hurdle",
        "gamma_one_proper_policy_certificate",
        "contract_sha256",
    }
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise RuntimeError(f"UNIFIED_EXIT_ECONOMICS_{label}_SHA256_INVALID")
    return value


def _finite(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"UNIFIED_EXIT_ECONOMICS_{label}_INVALID")
    try:
        observed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"UNIFIED_EXIT_ECONOMICS_{label}_INVALID") from exc
    if not math.isfinite(observed) or (nonnegative and observed < 0.0):
        raise RuntimeError(f"UNIFIED_EXIT_ECONOMICS_{label}_INVALID")
    return observed


def seal_train_fitted_capital_hurdle_artifact(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal a new artifact payload; validation remains a separate operation."""

    observed = dict(value)
    if "artifact_sha256" in observed:
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_HURDLE_ALREADY_SEALED")
    observed["artifact_sha256"] = _canonical_sha256(observed)
    return observed


def seal_frozen_capital_hurdle_owner_artifact(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal a preregistered external/operator owner without calling it a fit."""

    observed = dict(value)
    if "artifact_sha256" in observed:
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_HURDLE_ALREADY_SEALED")
    observed["artifact_sha256"] = _canonical_sha256(observed)
    return observed


def require_train_fitted_capital_hurdle_artifact(
    value: Mapping[str, Any],
    *,
    expected_train_split_sha256: str,
    expected_train_fold_sha256: str,
    expected_source_lineage_sha256: str,
) -> dict[str, Any]:
    """Validate the immutable TRAIN-only owner of rho/capital hurdle."""

    train_fit_keys = {
        "schema_version",
        "decision",
        "fitted_splits",
        "validation_or_test_used",
        "train_split_sha256",
        "train_fold_sha256",
        "source_lineage_sha256",
        "annual_continuous_hurdle_rate",
        "rate_unit",
        "seconds_per_year",
        "fit_method",
        "fit_evidence_sha256",
        "artifact_sha256",
    }
    frozen_owner_keys = {
        "schema_version",
        "decision",
        "owner_kind",
        "applicable_splits",
        "fitted_splits",
        "validation_or_test_used",
        "train_split_sha256",
        "train_fold_sha256",
        "source_lineage_sha256",
        "effective_annual_return_hurdle",
        "annual_continuous_hurdle_rate",
        "rate_conversion_formula",
        "rate_unit",
        "seconds_per_year",
        "source_method",
        "source_method_artifact_sha256",
        "artifact_sha256",
    }
    if not isinstance(value, Mapping) or frozenset(value) not in {
        frozenset(train_fit_keys),
        frozenset(frozen_owner_keys),
    }:
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_HURDLE_SCHEMA_INVALID")
    observed = dict(value)
    declared_sha256 = _require_sha256(
        observed.pop("artifact_sha256"), label="HURDLE_ARTIFACT"
    )
    if declared_sha256 != _canonical_sha256(observed):
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_HURDLE_HASH_INVALID")
    for label, expected in (
        ("TRAIN_SPLIT", expected_train_split_sha256),
        ("TRAIN_FOLD", expected_train_fold_sha256),
        ("SOURCE_LINEAGE", expected_source_lineage_sha256),
    ):
        _require_sha256(expected, label=f"EXPECTED_{label}")
        if observed[label.lower() + "_sha256"] != expected:
            raise RuntimeError(f"UNIFIED_EXIT_ECONOMICS_HURDLE_{label}_MISMATCH")
    common_invalid = (
        observed["decision"] != "PASS"
        or observed["validation_or_test_used"] is not False
        or observed["rate_unit"] != "continuous_per_wall_clock_year"
        or _finite(observed["seconds_per_year"], label="SECONDS_PER_YEAR")
        != SECONDS_PER_YEAR
    )
    if observed["schema_version"] == CAPITAL_HURDLE_SCHEMA_VERSION:
        fit_method = observed["fit_method"]
        owner_invalid = (
            observed["fitted_splits"] != ["train"]
            or not isinstance(fit_method, str)
            or not fit_method
        )
        _require_sha256(observed["fit_evidence_sha256"], label="FIT_EVIDENCE")
    elif observed["schema_version"] == FROZEN_CAPITAL_HURDLE_SCHEMA_VERSION:
        effective = _finite(
            observed["effective_annual_return_hurdle"],
            label="EFFECTIVE_ANNUAL_HURDLE",
            nonnegative=True,
        )
        owner_invalid = (
            observed["owner_kind"] != "project_preregistered_prospective_policy"
            or observed["applicable_splits"] != ["train", "val"]
            or observed["fitted_splits"] != []
            or observed["rate_conversion_formula"]
            != "rho=ln(1+effective_annual_return)"
            or not isinstance(observed["source_method"], str)
            or not observed["source_method"]
            or not math.isclose(
                float(observed["annual_continuous_hurdle_rate"]),
                math.log1p(effective),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        )
        _require_sha256(
            observed["source_method_artifact_sha256"], label="SOURCE_METHOD_ARTIFACT"
        )
    else:
        owner_invalid = True
    if common_invalid or owner_invalid:
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_HURDLE_POLICY_INVALID")
    rho = _finite(
        observed["annual_continuous_hurdle_rate"],
        label="ANNUAL_HURDLE_RATE",
        nonnegative=True,
    )
    return {
        **observed,
        "annual_continuous_hurdle_rate": rho,
        "artifact_sha256": declared_sha256,
    }


def seal_proper_policy_certificate(value: Mapping[str, Any]) -> dict[str, Any]:
    observed = dict(value)
    if "certificate_sha256" in observed:
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_CERTIFICATE_ALREADY_SEALED")
    observed["certificate_sha256"] = _canonical_sha256(observed)
    return observed


def require_proper_policy_certificate(
    value: Mapping[str, Any],
    *,
    expected_policy_sha256: str,
) -> dict[str, Any]:
    """Verify a finite absorbing-chain proof for an undiscounted policy.

    Loops are allowed, so this is not a hidden maximum holding time.  The
    transient transition matrix must have spectral radius below one and a
    finite fundamental matrix, which proves almost-sure absorption with finite
    expected stopping time for every declared nonterminal state.
    """

    expected_keys = {
        "schema_version",
        "proof_kind",
        "policy_sha256",
        "economic_terminal_policy_sha256",
        "transition_matrix",
        "initial_distribution",
        "terminal_state_indices",
        "claimed_nonterminal_spectral_radius",
        "claimed_maximum_expected_steps_to_absorption",
        "numeric_tolerance",
        "certificate_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_CERTIFICATE_SCHEMA_INVALID")
    observed = dict(value)
    declared_sha256 = _require_sha256(
        observed.pop("certificate_sha256"), label="PROPER_POLICY_CERTIFICATE"
    )
    if declared_sha256 != _canonical_sha256(observed):
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_CERTIFICATE_HASH_INVALID")
    expected_policy = _require_sha256(expected_policy_sha256, label="EXPECTED_POLICY")
    if (
        observed["schema_version"] != PROPER_POLICY_CERTIFICATE_SCHEMA_VERSION
        or observed["proof_kind"] != "absorbing_markov_chain_policy_v1"
        or observed["policy_sha256"] != expected_policy
    ):
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_CERTIFICATE_POLICY_INVALID")
    _require_sha256(
        observed["economic_terminal_policy_sha256"],
        label="ECONOMIC_TERMINAL_POLICY",
    )
    tolerance = _finite(
        observed["numeric_tolerance"],
        label="CERTIFICATE_TOLERANCE",
        nonnegative=True,
    )
    if not 0.0 < tolerance <= 1e-8:
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_CERTIFICATE_TOLERANCE_INVALID")
    try:
        transition = np.asarray(observed["transition_matrix"], dtype=np.float64)
        initial = np.asarray(observed["initial_distribution"], dtype=np.float64)
        terminal = np.asarray(observed["terminal_state_indices"], dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_CERTIFICATE_ARRAY_INVALID") from exc
    if (
        transition.ndim != 2
        or transition.shape[0] != transition.shape[1]
        or not 2 <= transition.shape[0] <= 1024
        or initial.shape != (transition.shape[0],)
        or terminal.ndim != 1
        or terminal.size < 1
        or np.unique(terminal).size != terminal.size
        or np.any(terminal < 0)
        or np.any(terminal >= transition.shape[0])
        or not np.isfinite(transition).all()
        or not np.isfinite(initial).all()
        or np.any(transition < 0.0)
        or np.any(initial < 0.0)
        or not np.allclose(transition.sum(axis=1), 1.0, rtol=0.0, atol=tolerance)
        or not math.isclose(float(initial.sum()), 1.0, rel_tol=0.0, abs_tol=tolerance)
    ):
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_CERTIFICATE_ARRAY_INVALID")
    terminal_set = set(int(index) for index in terminal.tolist())
    for index in terminal_set:
        expected_row = np.zeros(transition.shape[0], dtype=np.float64)
        expected_row[index] = 1.0
        if not np.allclose(transition[index], expected_row, rtol=0.0, atol=tolerance):
            raise RuntimeError(
                "UNIFIED_EXIT_ECONOMICS_CERTIFICATE_TERMINAL_NOT_ABSORBING"
            )
    nonterminal = np.asarray(
        [index for index in range(transition.shape[0]) if index not in terminal_set],
        dtype=np.int64,
    )
    if nonterminal.size < 1:
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_CERTIFICATE_NO_TRANSIENT_STATE")
    transient = transition[np.ix_(nonterminal, nonterminal)]
    try:
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transient))))
        fundamental = np.linalg.inv(np.eye(len(nonterminal)) - transient)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_CERTIFICATE_NOT_PROPER") from exc
    expected_steps = fundamental @ np.ones(len(nonterminal), dtype=np.float64)
    maximum_expected_steps = float(np.max(expected_steps))
    claimed_radius = _finite(
        observed["claimed_nonterminal_spectral_radius"],
        label="CLAIMED_SPECTRAL_RADIUS",
        nonnegative=True,
    )
    claimed_steps = _finite(
        observed["claimed_maximum_expected_steps_to_absorption"],
        label="CLAIMED_EXPECTED_STEPS",
        nonnegative=True,
    )
    if (
        not math.isfinite(spectral_radius)
        or spectral_radius >= 1.0 - tolerance
        or not np.isfinite(fundamental).all()
        or not np.isfinite(expected_steps).all()
        or np.any(expected_steps < 1.0 - tolerance)
        or not math.isclose(
            claimed_radius, spectral_radius, rel_tol=1e-9, abs_tol=tolerance
        )
        or not math.isclose(
            claimed_steps,
            maximum_expected_steps,
            rel_tol=1e-9,
            abs_tol=tolerance,
        )
    ):
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_CERTIFICATE_NOT_PROPER")
    return {
        **observed,
        "transition_matrix": transition.tolist(),
        "initial_distribution": initial.tolist(),
        "terminal_state_indices": terminal.tolist(),
        "verified_nonterminal_spectral_radius": spectral_radius,
        "verified_maximum_expected_steps_to_absorption": maximum_expected_steps,
        "certificate_sha256": declared_sha256,
    }


def build_unified_exit_economics_objective_contract(
    *,
    capital_hurdle_artifact: Mapping[str, Any],
    expected_train_split_sha256: str,
    expected_train_fold_sha256: str,
    expected_source_lineage_sha256: str,
    policy_sha256: str,
    proper_policy_certificate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    hurdle = require_train_fitted_capital_hurdle_artifact(
        capital_hurdle_artifact,
        expected_train_split_sha256=expected_train_split_sha256,
        expected_train_fold_sha256=expected_train_fold_sha256,
        expected_source_lineage_sha256=expected_source_lineage_sha256,
    )
    policy = _require_sha256(policy_sha256, label="POLICY")
    rho = float(hurdle["annual_continuous_hurdle_rate"])
    certificate_summary = None
    if rho == 0.0:
        if proper_policy_certificate is None:
            raise RuntimeError(
                "UNIFIED_EXIT_ECONOMICS_GAMMA_ONE_PROPER_POLICY_REQUIRED"
            )
        certificate = require_proper_policy_certificate(
            proper_policy_certificate,
            expected_policy_sha256=policy,
        )
        certificate_summary = {
            "certificate_sha256": certificate["certificate_sha256"],
            "economic_terminal_policy_sha256": certificate[
                "economic_terminal_policy_sha256"
            ],
            "verified_nonterminal_spectral_radius": certificate[
                "verified_nonterminal_spectral_radius"
            ],
            "verified_maximum_expected_steps_to_absorption": certificate[
                "verified_maximum_expected_steps_to_absorption"
            ],
        }
    elif proper_policy_certificate is not None:
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_UNUSED_CERTIFICATE_FORBIDDEN")
    payload = {
        "schema_version": ECONOMICS_OBJECTIVE_SCHEMA_VERSION,
        "target_unit": "bps_of_entry_notional",
        "policy_sha256": policy,
        "capital_hurdle_artifact_sha256": hurdle["artifact_sha256"],
        "train_split_sha256": hurdle["train_split_sha256"],
        "train_fold_sha256": hurdle["train_fold_sha256"],
        "source_lineage_sha256": hurdle["source_lineage_sha256"],
        "annual_continuous_hurdle_rate": rho,
        "seconds_per_year": SECONDS_PER_YEAR,
        "discount_factor": "exp(-rho*elapsed_wall_clock_seconds/seconds_per_year)",
        "capacity_or_chunk_length_affects_gamma": False,
        "same_capital_hurdle_running_cost_allowed": False,
        "capital_hurdle_double_counting_forbidden": True,
        "undiscounted_net_cash_pnl_is_separate": True,
        "discounted_risk_utility_is_training_objective": True,
        "risk_utility_penalty_is_cash_pnl": False,
        "required_complete_cost_inputs": list(_COST_COMPONENTS),
        "explicit_gap_classification_required": True,
        "right_censor_is_terminal": False,
        "validation_or_test_may_fit_hurdle": False,
        "gamma_one_proper_policy_certificate": certificate_summary,
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


def require_unified_exit_economics_objective_contract(
    value: Mapping[str, Any],
    *,
    capital_hurdle_artifact: Mapping[str, Any],
    expected_train_split_sha256: str,
    expected_train_fold_sha256: str,
    expected_source_lineage_sha256: str,
    policy_sha256: str,
    proper_policy_certificate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected = build_unified_exit_economics_objective_contract(
        capital_hurdle_artifact=capital_hurdle_artifact,
        expected_train_split_sha256=expected_train_split_sha256,
        expected_train_fold_sha256=expected_train_fold_sha256,
        expected_source_lineage_sha256=expected_source_lineage_sha256,
        policy_sha256=policy_sha256,
        proper_policy_certificate=proper_policy_certificate,
    )
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_OBJECTIVE_CONTRACT_INVALID")
    return expected


def _require_runtime_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _OBJECTIVE_CONTRACT_KEYS:
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_OBJECTIVE_CONTRACT_INVALID")
    observed = dict(value)
    declared = observed.pop("contract_sha256")
    if (
        _require_sha256(declared, label="OBJECTIVE_CONTRACT")
        != _canonical_sha256(observed)
        or observed["schema_version"] != ECONOMICS_OBJECTIVE_SCHEMA_VERSION
        or observed["target_unit"] != "bps_of_entry_notional"
        or observed["seconds_per_year"] != SECONDS_PER_YEAR
        or observed["discount_factor"]
        != "exp(-rho*elapsed_wall_clock_seconds/seconds_per_year)"
        or observed["capacity_or_chunk_length_affects_gamma"] is not False
        or observed["same_capital_hurdle_running_cost_allowed"] is not False
        or observed["capital_hurdle_double_counting_forbidden"] is not True
        or observed["undiscounted_net_cash_pnl_is_separate"] is not True
        or observed["discounted_risk_utility_is_training_objective"] is not True
        or observed["risk_utility_penalty_is_cash_pnl"] is not False
        or observed["required_complete_cost_inputs"] != list(_COST_COMPONENTS)
        or observed["explicit_gap_classification_required"] is not True
        or observed["right_censor_is_terminal"] is not False
        or observed["validation_or_test_may_fit_hurdle"] is not False
    ):
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_OBJECTIVE_CONTRACT_INVALID")
    for name in (
        "policy_sha256",
        "capital_hurdle_artifact_sha256",
        "train_split_sha256",
        "train_fold_sha256",
        "source_lineage_sha256",
    ):
        _require_sha256(observed[name], label=name.upper())
    return {**observed, "contract_sha256": declared}


def elapsed_wall_clock_gamma(
    *, contract: Mapping[str, Any], elapsed_wall_clock_seconds: Any
) -> float:
    checked_contract = _require_runtime_contract(contract)
    elapsed = _finite(
        elapsed_wall_clock_seconds,
        label="ELAPSED_WALL_CLOCK_SECONDS",
        nonnegative=True,
    )
    rho = _finite(
        checked_contract["annual_continuous_hurdle_rate"],
        label="ANNUAL_HURDLE_RATE",
        nonnegative=True,
    )
    exponent = -rho * elapsed / SECONDS_PER_YEAR
    gamma = math.exp(exponent)
    if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_GAMMA_INVALID")
    return gamma


def _complete_component(value: Any, *, label: str, nonnegative: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "status",
        "value_bps",
        "source_artifact_sha256",
    }:
        raise RuntimeError(f"UNIFIED_EXIT_ECONOMICS_{label}_INCOMPLETE")
    observed = dict(value)
    if observed["status"] != "COMPLETE":
        raise RuntimeError(f"UNIFIED_EXIT_ECONOMICS_{label}_INCOMPLETE")
    return {
        "status": "COMPLETE",
        "value_bps": _finite(
            observed["value_bps"], label=label, nonnegative=nonnegative
        ),
        "source_artifact_sha256": _require_sha256(
            observed["source_artifact_sha256"], label=f"{label}_SOURCE"
        ),
    }


def require_economic_step_inputs(
    value: Mapping[str, Any], *, contract: Mapping[str, Any]
) -> dict[str, Any]:
    contract = _require_runtime_contract(contract)
    expected_keys = {
        "schema_version",
        "event_kind",
        "interval_start_time_ns",
        "interval_end_time_ns",
        "gross_price_cashflow",
        *_COST_COMPONENTS,
        "risk_utility_penalty",
        "same_capital_hurdle_running_cost",
        "gap",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_STEP_SCHEMA_INVALID")
    observed = dict(value)
    if (
        observed["schema_version"] != ECONOMIC_STEP_SCHEMA_VERSION
        or observed["event_kind"] not in _EVENT_KINDS
    ):
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_STEP_POLICY_INVALID")
    start = observed["interval_start_time_ns"]
    end = observed["interval_end_time_ns"]
    if (
        isinstance(start, bool)
        or not isinstance(start, (int, np.integer))
        or isinstance(end, bool)
        or not isinstance(end, (int, np.integer))
        or int(start) <= 0
        or int(end) < int(start)
    ):
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_STEP_CLOCK_INVALID")
    elapsed_ns = int(end) - int(start)
    if elapsed_ns % 1_000_000_000 != 0:
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_STEP_CLOCK_INVALID")
    elapsed_seconds = elapsed_ns // 1_000_000_000
    gap = observed["gap"]
    if not isinstance(gap, Mapping) or set(gap) != {
        "status",
        "classification",
        "source_manifest_sha256",
        "classification_artifact_sha256",
    }:
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_GAP_INPUT_INCOMPLETE")
    gap = dict(gap)
    if gap["status"] != "COMPLETE":
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_GAP_INPUT_INCOMPLETE")
    classification = gap["classification"]
    if (
        (elapsed_seconds == 0 and classification != "instantaneous_execution")
        or (elapsed_seconds == NOMINAL_M1_SECONDS and classification != "continuous_m1")
        or (
            elapsed_seconds > NOMINAL_M1_SECONDS
            and classification != "declared_market_closure"
        )
        or 0 < elapsed_seconds < NOMINAL_M1_SECONDS
    ):
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_GAP_CLASSIFICATION_INVALID")
    gap["source_manifest_sha256"] = _require_sha256(
        gap["source_manifest_sha256"], label="GAP_SOURCE_MANIFEST"
    )
    gap["classification_artifact_sha256"] = _require_sha256(
        gap["classification_artifact_sha256"],
        label="GAP_CLASSIFICATION_ARTIFACT",
    )
    components = {
        "gross_price_cashflow": _complete_component(
            observed["gross_price_cashflow"],
            label="GROSS_PRICE_CASHFLOW",
            nonnegative=False,
        ),
        **{
            name: _complete_component(
                observed[name],
                label=name.upper(),
                nonnegative=name in _NONNEGATIVE_COMPONENTS,
            )
            for name in _COST_COMPONENTS
        },
        "risk_utility_penalty": _complete_component(
            observed["risk_utility_penalty"],
            label="RISK_UTILITY_PENALTY",
            nonnegative=True,
        ),
        "same_capital_hurdle_running_cost": _complete_component(
            observed["same_capital_hurdle_running_cost"],
            label="SAME_CAPITAL_HURDLE_RUNNING_COST",
            nonnegative=True,
        ),
    }
    running_capital = components["same_capital_hurdle_running_cost"]
    if (
        contract.get("same_capital_hurdle_running_cost_allowed") is not False
        or running_capital["source_artifact_sha256"]
        != contract.get("capital_hurdle_artifact_sha256")
        or running_capital["value_bps"] != 0.0
    ):
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_CAPITAL_COST_DOUBLE_COUNTED")
    return {
        "schema_version": ECONOMIC_STEP_SCHEMA_VERSION,
        "event_kind": observed["event_kind"],
        "interval_start_time_ns": int(start),
        "interval_end_time_ns": int(end),
        "elapsed_wall_clock_seconds": int(elapsed_seconds),
        **components,
        "gap": gap,
    }


def compose_economic_step(
    value: Mapping[str, Any], *, contract: Mapping[str, Any]
) -> dict[str, Any]:
    step = require_economic_step_inputs(value, contract=contract)
    gross = step["gross_price_cashflow"]["value_bps"]
    commission = step["commission"]["value_bps"]
    slippage = step["execution_slippage"]["value_bps"]
    financing = step["financing_or_swap"]["value_bps"]
    guaranteed_fee = step["guaranteed_execution_fee"]["value_bps"]
    risk_penalty = step["risk_utility_penalty"]["value_bps"]
    net_cash = gross + financing - commission - slippage - guaranteed_fee
    utility = net_cash - risk_penalty
    gamma = elapsed_wall_clock_gamma(
        contract=contract,
        elapsed_wall_clock_seconds=step["elapsed_wall_clock_seconds"],
    )
    if not all(math.isfinite(value) for value in (net_cash, utility, gamma)):
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_STEP_RESULT_NONFINITE")
    return {
        **step,
        "undiscounted_net_cash_pnl_increment_bps": float(net_cash),
        "risk_utility_penalty_bps": float(risk_penalty),
        "undiscounted_risk_adjusted_utility_increment_bps": float(utility),
        "continuation_gamma": float(gamma),
        "capital_hurdle_applied_via_discount_only": True,
    }


def discounted_continuation_target_bps(
    *, step: Mapping[str, Any], successor_utility_bps: Any
) -> float:
    if step.get("event_kind") not in {"ENTRY", "HOLD"}:
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_CONTINUATION_EVENT_INVALID")
    immediate = _finite(
        step.get("undiscounted_risk_adjusted_utility_increment_bps"),
        label="IMMEDIATE_UTILITY",
    )
    gamma = _finite(
        step.get("continuation_gamma"),
        label="CONTINUATION_GAMMA",
        nonnegative=True,
    )
    successor = _finite(successor_utility_bps, label="SUCCESSOR_UTILITY")
    if not 0.0 < gamma <= 1.0:
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_CONTINUATION_GAMMA_INVALID")
    return float(immediate + gamma * successor)


def evaluate_economic_path(
    steps: Sequence[Mapping[str, Any]], *, contract: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_PATH_INVALID")
    composed = [compose_economic_step(step, contract=contract) for step in steps]
    if (
        len(composed) < 2
        or composed[0]["event_kind"] != "ENTRY"
        or composed[-1]["event_kind"] not in {"EXIT_NOW", "ECONOMIC_TERMINAL"}
        or any(row["event_kind"] != "HOLD" for row in composed[1:-1])
        or any(
            left["interval_end_time_ns"] != right["interval_start_time_ns"]
            for left, right in zip(composed, composed[1:])
        )
    ):
        raise RuntimeError("UNIFIED_EXIT_ECONOMICS_PATH_ORDER_INVALID")
    discount = 1.0
    net_cash = 0.0
    risk_penalty = 0.0
    discounted_utility = 0.0
    component_totals = {
        "gross_price_cashflow_bps": 0.0,
        "commission_bps": 0.0,
        "execution_slippage_bps": 0.0,
        "financing_or_swap_bps": 0.0,
        "guaranteed_execution_fee_bps": 0.0,
    }
    for row in composed:
        net_cash += row["undiscounted_net_cash_pnl_increment_bps"]
        risk_penalty += row["risk_utility_penalty_bps"]
        discounted_utility += (
            discount * row["undiscounted_risk_adjusted_utility_increment_bps"]
        )
        component_totals["gross_price_cashflow_bps"] += row["gross_price_cashflow"][
            "value_bps"
        ]
        for name in _COST_COMPONENTS:
            component_totals[f"{name}_bps"] += row[name]["value_bps"]
        discount *= row["continuation_gamma"]
    result = {
        "schema_version": ECONOMIC_PATH_SCHEMA_VERSION,
        "objective_contract_sha256": contract["contract_sha256"],
        "step_count": len(composed),
        "undiscounted_net_cash_pnl_bps": float(net_cash),
        "undiscounted_risk_utility_penalty_bps": float(risk_penalty),
        "discounted_risk_adjusted_utility_bps": float(discounted_utility),
        "terminal_discount_from_entry": float(discount),
        "component_totals": component_totals,
        "same_capital_hurdle_running_cost_bps": 0.0,
        "capital_hurdle_double_counted": False,
    }
    result["path_evidence_sha256"] = _canonical_sha256(result)
    return result


__all__ = (
    "CAPITAL_HURDLE_SCHEMA_VERSION",
    "FROZEN_CAPITAL_HURDLE_SCHEMA_VERSION",
    "ECONOMIC_PATH_SCHEMA_VERSION",
    "ECONOMIC_STEP_SCHEMA_VERSION",
    "ECONOMICS_OBJECTIVE_SCHEMA_VERSION",
    "PROPER_POLICY_CERTIFICATE_SCHEMA_VERSION",
    "SECONDS_PER_YEAR",
    "build_unified_exit_economics_objective_contract",
    "compose_economic_step",
    "discounted_continuation_target_bps",
    "elapsed_wall_clock_gamma",
    "evaluate_economic_path",
    "require_economic_step_inputs",
    "require_proper_policy_certificate",
    "require_train_fitted_capital_hurdle_artifact",
    "require_unified_exit_economics_objective_contract",
    "seal_proper_policy_certificate",
    "seal_train_fitted_capital_hurdle_artifact",
    "seal_frozen_capital_hurdle_owner_artifact",
)
