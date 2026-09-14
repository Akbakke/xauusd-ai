from __future__ import annotations

import copy
import math

import pytest

from gx1.contracts import unified_exit_economics_objective_v2 as owner


TRAIN_SPLIT = "1" * 64
TRAIN_FOLD = "2" * 64
SOURCE_LINEAGE = "3" * 64
POLICY = "4" * 64
TERMINAL_POLICY = "5" * 64
FIT_EVIDENCE = "6" * 64
COST_SOURCE = "7" * 64
GAP_SOURCE = "8" * 64
GAP_CLASSIFIER = "9" * 64
RISK_SOURCE = "a" * 64


def _hurdle(rho: float = 0.10) -> dict:
    return owner.seal_train_fitted_capital_hurdle_artifact(
        {
            "schema_version": owner.CAPITAL_HURDLE_SCHEMA_VERSION,
            "decision": "PASS",
            "fitted_splits": ["train"],
            "validation_or_test_used": False,
            "train_split_sha256": TRAIN_SPLIT,
            "train_fold_sha256": TRAIN_FOLD,
            "source_lineage_sha256": SOURCE_LINEAGE,
            "annual_continuous_hurdle_rate": rho,
            "rate_unit": "continuous_per_wall_clock_year",
            "seconds_per_year": owner.SECONDS_PER_YEAR,
            "fit_method": "train_only_capital_hurdle_fit_v1",
            "fit_evidence_sha256": FIT_EVIDENCE,
        }
    )


def _certificate(transition=None) -> dict:
    if transition is None:
        transition = [[0.5, 0.5], [0.0, 1.0]]
    return owner.seal_proper_policy_certificate(
        {
            "schema_version": owner.PROPER_POLICY_CERTIFICATE_SCHEMA_VERSION,
            "proof_kind": "absorbing_markov_chain_policy_v1",
            "policy_sha256": POLICY,
            "economic_terminal_policy_sha256": TERMINAL_POLICY,
            "transition_matrix": transition,
            "initial_distribution": [1.0, 0.0],
            "terminal_state_indices": [1],
            "claimed_nonterminal_spectral_radius": float(transition[0][0]),
            "claimed_maximum_expected_steps_to_absorption": (
                1.0 / (1.0 - float(transition[0][0]))
                if float(transition[0][0]) < 1.0
                else 1.0e30
            ),
            "numeric_tolerance": 1.0e-12,
        }
    )


def _contract(rho: float = 0.10, certificate=None, reward_accounting="terminal_cash_v2") -> dict:
    return owner.build_unified_exit_economics_objective_contract(
        capital_hurdle_artifact=_hurdle(rho),
        expected_train_split_sha256=TRAIN_SPLIT,
        expected_train_fold_sha256=TRAIN_FOLD,
        expected_source_lineage_sha256=SOURCE_LINEAGE,
        policy_sha256=POLICY,
        proper_policy_certificate=certificate,
        reward_accounting=reward_accounting,
    )


def _component(value: float, source: str = COST_SOURCE) -> dict:
    return {
        "status": "COMPLETE",
        "value_bps": value,
        "source_artifact_sha256": source,
    }


def _step(
    *,
    event: str,
    start: int,
    end: int,
    hurdle_sha256: str,
    gross: float,
    commission: float = 0.0,
    slippage: float = 0.0,
    financing: float = 0.0,
    guaranteed_fee: float = 0.0,
    risk: float = 0.0,
) -> dict:
    elapsed_seconds = (end - start) // 1_000_000_000
    classification = (
        "instantaneous_execution"
        if elapsed_seconds == 0
        else "continuous_m1"
        if elapsed_seconds == 60
        else "declared_market_closure"
    )
    return {
        "schema_version": owner.ECONOMIC_STEP_SCHEMA_VERSION,
        "event_kind": event,
        "interval_start_time_ns": start,
        "interval_end_time_ns": end,
        "gross_price_cashflow": _component(gross),
        "commission": _component(commission),
        "execution_slippage": _component(slippage),
        "financing_or_swap": _component(financing),
        "guaranteed_execution_fee": _component(guaranteed_fee),
        "risk_utility_penalty": _component(risk, RISK_SOURCE),
        "same_capital_hurdle_running_cost": _component(0.0, hurdle_sha256),
        "gap": {
            "status": "COMPLETE",
            "classification": classification,
            "source_manifest_sha256": GAP_SOURCE,
            "classification_artifact_sha256": GAP_CLASSIFIER,
        },
    }


def test_train_fitted_hurdle_is_hash_bound_and_val_test_fit_is_forbidden() -> None:
    artifact = _hurdle()
    checked = owner.require_train_fitted_capital_hurdle_artifact(
        artifact,
        expected_train_split_sha256=TRAIN_SPLIT,
        expected_train_fold_sha256=TRAIN_FOLD,
        expected_source_lineage_sha256=SOURCE_LINEAGE,
    )
    assert checked["annual_continuous_hurdle_rate"] == pytest.approx(0.10)

    tampered = copy.deepcopy(artifact)
    tampered["annual_continuous_hurdle_rate"] = 0.20
    with pytest.raises(RuntimeError, match="HURDLE_HASH_INVALID"):
        owner.require_train_fitted_capital_hurdle_artifact(
            tampered,
            expected_train_split_sha256=TRAIN_SPLIT,
            expected_train_fold_sha256=TRAIN_FOLD,
            expected_source_lineage_sha256=SOURCE_LINEAGE,
        )

    invalid_fit = copy.deepcopy(artifact)
    invalid_fit.pop("artifact_sha256")
    invalid_fit["validation_or_test_used"] = True
    invalid_fit = owner.seal_train_fitted_capital_hurdle_artifact(invalid_fit)
    with pytest.raises(RuntimeError, match="HURDLE_POLICY_INVALID"):
        owner.require_train_fitted_capital_hurdle_artifact(
            invalid_fit,
            expected_train_split_sha256=TRAIN_SPLIT,
            expected_train_fold_sha256=TRAIN_FOLD,
            expected_source_lineage_sha256=SOURCE_LINEAGE,
        )


def test_elapsed_wall_clock_gamma_uses_hurdle_and_not_chunk_capacity() -> None:
    contract = _contract()
    gamma_minute = owner.elapsed_wall_clock_gamma(
        contract=contract, elapsed_wall_clock_seconds=60
    )
    gamma_weekend = owner.elapsed_wall_clock_gamma(
        contract=contract, elapsed_wall_clock_seconds=48 * 60 * 60
    )
    assert gamma_minute == pytest.approx(
        math.exp(-0.10 * 60.0 / owner.SECONDS_PER_YEAR)
    )
    assert 0.0 < gamma_weekend < gamma_minute < 1.0
    assert contract["capacity_or_chunk_length_affects_gamma"] is False
    assert contract["same_capital_hurdle_running_cost_allowed"] is False


def test_gamma_one_requires_verified_absorbing_policy_not_boolean_and_sha() -> None:
    with pytest.raises(RuntimeError, match="GAMMA_ONE_PROPER_POLICY_REQUIRED"):
        _contract(rho=0.0)
    with pytest.raises(RuntimeError, match="CERTIFICATE_SCHEMA_INVALID"):
        _contract(
            rho=0.0,
            certificate={
                "undiscounted_proper_policy_proven": True,
                "economic_terminal_policy_sha256": TERMINAL_POLICY,
            },
        )

    contract = _contract(rho=0.0, certificate=_certificate())
    proof = contract["gamma_one_proper_policy_certificate"]
    assert proof["verified_nonterminal_spectral_radius"] == pytest.approx(0.5)
    assert proof["verified_maximum_expected_steps_to_absorption"] == pytest.approx(
        2.0
    )
    assert owner.elapsed_wall_clock_gamma(
        contract=contract, elapsed_wall_clock_seconds=10_000_000
    ) == pytest.approx(1.0)


def test_gamma_one_rejects_nonabsorbing_policy_even_when_self_sealed() -> None:
    certificate = _certificate([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(RuntimeError, match="CERTIFICATE_NOT_PROPER"):
        _contract(rho=0.0, certificate=certificate)


def test_step_requires_complete_costs_and_explicit_gap_classification() -> None:
    contract = _contract()
    start = 1_800_000_000_000_000_000
    step = _step(
        event="HOLD",
        start=start,
        end=start + 60_000_000_000,
        hurdle_sha256=contract["capital_hurdle_artifact_sha256"],
        gross=1.0,
        financing=-0.1,
    )
    composed = owner.compose_economic_step(step, contract=contract)
    assert composed["undiscounted_net_cash_pnl_increment_bps"] == pytest.approx(
        0.9
    )

    missing_commission = copy.deepcopy(step)
    missing_commission["commission"]["status"] = "UNKNOWN"
    with pytest.raises(RuntimeError, match="COMMISSION_INCOMPLETE"):
        owner.compose_economic_step(missing_commission, contract=contract)

    unclassified_gap = copy.deepcopy(step)
    unclassified_gap["gap"]["classification"] = "unknown_gap"
    with pytest.raises(RuntimeError, match="GAP_CLASSIFICATION_INVALID"):
        owner.compose_economic_step(unclassified_gap, contract=contract)

    sub_minute_gap = copy.deepcopy(step)
    sub_minute_gap["interval_end_time_ns"] = start + 30_000_000_000
    sub_minute_gap["gap"]["classification"] = "declared_market_closure"
    with pytest.raises(RuntimeError, match="GAP_CLASSIFICATION_INVALID"):
        owner.compose_economic_step(sub_minute_gap, contract=contract)


def test_same_capital_hurdle_cannot_be_discounted_and_charged_again() -> None:
    contract = _contract()
    start = 1_800_000_000_000_000_000
    step = _step(
        event="HOLD",
        start=start,
        end=start + 60_000_000_000,
        hurdle_sha256=contract["capital_hurdle_artifact_sha256"],
        gross=0.0,
    )
    step["same_capital_hurdle_running_cost"]["value_bps"] = 0.001
    with pytest.raises(RuntimeError, match="CAPITAL_COST_DOUBLE_COUNTED"):
        owner.compose_economic_step(step, contract=contract)


def test_path_keeps_undiscounted_cash_separate_from_discounted_risk_utility() -> None:
    contract = _contract()
    hurdle_sha = contract["capital_hurdle_artifact_sha256"]
    t0 = 1_800_000_000_000_000_000
    minute = 60_000_000_000
    steps = [
        _step(
            event="ENTRY",
            start=t0,
            end=t0 + minute,
            hurdle_sha256=hurdle_sha,
            gross=2.0,
            commission=0.1,
            slippage=0.2,
            risk=0.05,
        ),
        _step(
            event="HOLD",
            start=t0 + minute,
            end=t0 + 2 * minute,
            hurdle_sha256=hurdle_sha,
            gross=1.0,
            financing=-0.1,
            risk=0.2,
        ),
        _step(
            event="EXIT_NOW",
            start=t0 + 2 * minute,
            end=t0 + 2 * minute,
            hurdle_sha256=hurdle_sha,
            gross=-0.3,
            commission=0.1,
            slippage=0.2,
        ),
    ]
    report = owner.evaluate_economic_path(steps, contract=contract)
    gamma = math.exp(-0.10 * 60.0 / owner.SECONDS_PER_YEAR)
    expected_utility = 1.65 + gamma * 0.7 + gamma**2 * -0.6
    assert report["undiscounted_net_cash_pnl_bps"] == pytest.approx(2.0)
    assert report["undiscounted_risk_utility_penalty_bps"] == pytest.approx(0.25)
    assert report["discounted_risk_adjusted_utility_bps"] == pytest.approx(
        expected_utility
    )
    assert report["capital_hurdle_double_counted"] is False
    assert report["component_totals"]["financing_or_swap_bps"] == pytest.approx(
        -0.1
    )


def test_discounted_continuation_uses_incremental_utility_and_elapsed_gamma() -> None:
    contract = _contract()
    start = 1_800_000_000_000_000_000
    step = owner.compose_economic_step(
        _step(
            event="HOLD",
            start=start,
            end=start + 60_000_000_000,
            hurdle_sha256=contract["capital_hurdle_artifact_sha256"],
            gross=1.0,
            financing=-0.1,
            risk=0.2,
        ),
        contract=contract,
    )
    expected = 0.7 + step["continuation_gamma"] * 3.0
    assert owner.discounted_continuation_target_bps(
        step=step, successor_utility_bps=3.0
    ) == pytest.approx(expected)


def _marked_step(*, successor_value=0.0, **kwargs):
    step = _step(**kwargs)
    step["schema_version"] = owner.MARK_TO_MARKET_STEP_SCHEMA_VERSION
    step["successor_liquidation_value"] = _component(successor_value)
    return step


def test_marked_contract_is_explicit_and_requires_observed_successor_value():
    legacy = _contract()
    contract = _contract(reward_accounting=owner.MARK_TO_MARKET_REWARD_ACCOUNTING)
    assert contract["contract_sha256"] != legacy["contract_sha256"]
    assert "reward_accounting" not in legacy
    kwargs = dict(event="HOLD", start=1_000_000_000, end=61_000_000_000,
                  hurdle_sha256=contract["capital_hurdle_artifact_sha256"], gross=0.0)
    with pytest.raises(RuntimeError, match="STEP_SCHEMA_INVALID"):
        owner.compose_economic_step(_step(**kwargs), contract=contract)
    marked = _marked_step(**kwargs, successor_value=-100.0)
    with pytest.raises(RuntimeError, match="STEP_SCHEMA_INVALID"):
        owner.compose_economic_step(marked, contract=legacy)
    marked["successor_liquidation_value"]["status"] = "MISSING"
    with pytest.raises(RuntimeError, match="SUCCESSOR_LIQUIDATION_VALUE_INCOMPLETE"):
        owner.compose_economic_step(marked, contract=contract)


@pytest.mark.parametrize("values", [(-8.0, 16.0, -12.0, -35.0), (-8.0, -16.0, 12.0, 35.0)])
def test_marked_path_matches_discounted_value_changes_without_double_counting(values):
    contract = _contract(reward_accounting=owner.MARK_TO_MARKET_REWARD_ACCOUNTING)
    hurdle = contract["capital_hurdle_artifact_sha256"]
    t = 1_000_000_000
    steps = [_marked_step(event="ENTRY", start=t, end=t, hurdle_sha256=hurdle, gross=0.0)]
    expected = values[0]
    discount = 1.0
    financing_total = 0.0
    for i, seconds in enumerate((60, 172800, 60)):
        financing, risk = -0.03 * seconds / 60.0, 0.01 * seconds / 60.0
        nxt = t + seconds * 1_000_000_000
        steps.append(_marked_step(event="HOLD", start=t, end=nxt,
                                  hurdle_sha256=hurdle, gross=0.0,
                                  financing=financing, risk=risk,
                                  successor_value=values[i + 1]))
        expected += discount * (values[i + 1] - values[i] + financing - risk)
        financing_total += financing
        discount *= owner.elapsed_wall_clock_gamma(contract=contract, elapsed_wall_clock_seconds=seconds)
        t = nxt
    steps.append(_marked_step(event="EXIT_NOW", start=t, end=t, hurdle_sha256=hurdle,
                              gross=values[-1] + 7.02, commission=3.0, slippage=4.0,
                              financing=-0.02))
    result = owner.evaluate_economic_path(steps, contract=contract)
    assert result["discounted_risk_adjusted_utility_bps"] == pytest.approx(expected, abs=1e-11)
    assert result["undiscounted_net_cash_pnl_bps"] == pytest.approx(values[-1] + financing_total, abs=1e-11)
    # Cash remains an undiscounted ledger, distinct from the learning objective.
    assert result["discounted_risk_adjusted_utility_bps"] != result["undiscounted_net_cash_pnl_bps"]


def test_unrealized_loss_no_longer_disappears_under_indefinite_zero_cost_hold():
    legacy = _contract()
    contract = _contract(reward_accounting=owner.MARK_TO_MARKET_REWARD_ACCOUNTING)
    loss = -100.0
    kwargs = dict(event="HOLD", start=1_000_000_000, end=61_000_000_000,
                  hurdle_sha256=contract["capital_hurdle_artifact_sha256"], gross=0.0)
    old = owner.compose_economic_step(_step(**kwargs), contract=legacy)
    new = owner.compose_economic_step(_marked_step(**kwargs, successor_value=loss), contract=contract)
    gamma = new["continuation_gamma"]
    assert old["undiscounted_risk_adjusted_utility_increment_bps"] / (1.0 - gamma) == 0.0
    assert new["undiscounted_risk_adjusted_utility_increment_bps"] / (1.0 - gamma) == pytest.approx(loss)
    # Flat prices and zero costs create indifference, not a mandatory exit rule.
    assert owner.discounted_continuation_target_bps(step=new, successor_utility_bps=loss) == pytest.approx(loss)
    # A declining successor makes HOLD strictly worse than realizing now.
    falling = owner.compose_economic_step(_marked_step(**kwargs, successor_value=loss - 1.0), contract=contract)
    assert owner.discounted_continuation_target_bps(step=falling, successor_utility_bps=loss - 1.0) < loss
