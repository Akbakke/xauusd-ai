"""Regressions for liquidation-relative Exit values at the native VAL boundary."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.contracts import unified_exit_economics_objective_v2 as economics
from gx1.contracts.unified_exit_native_candidate_campaign_v1 import require_complete_val_observation
from gx1.contracts.unified_exit_random_access_val_evaluator_v1 import (
    LIQUIDATION_RELATIVE_RESULT_SCHEMA_VERSION,
    MARKED_RESULT_SCHEMA_VERSION,
    PAUSE_SCHEMA_VERSION,
    canonical_sha256,
    require_random_access_val_evaluation_result_v1,
    run_resumable_random_access_val_evaluation_v1,
)
from tests.test_unified_exit_economic_step_provider_v1 import _provider
from tests.test_unified_exit_economics_objective_v2 import _contract, _marked_step
from tests.test_unified_exit_random_access_val_evaluator_v1 import (
    _checkpoint_binding,
    _entry_policy,
    _with_route_outputs,
)
from tests.test_unified_exit_random_access_val_rollout_v1 import (
    VAL_ENTRY_COHORT_SIZE,
    _Policy,
    _fixture,
    _normalization,
    _state_provider,
)


@pytest.mark.parametrize("side", [0, 1], ids=["long", "short"])
def test_relative_value_reconstructs_marked_utility_and_cash_across_closure(side):
    """The coordinate change neither creates profit nor discounts away a loss."""
    old = _contract(reward_accounting=economics.MARK_TO_MARKET_REWARD_ACCOUNTING)
    new = _contract(reward_accounting=economics.LIQUIDATION_ADVANTAGE_REWARD_ACCOUNTING)
    assert old["contract_sha256"] != new["contract_sha256"]
    hurdle = new["capital_hurdle_artifact_sha256"]
    entry_price = (2000.20, 2000.0)[side]  # Executable ASK/BID entry.
    closes = np.array([2000.0, 2001.0, 1998.0, 1999.0]) + side * 0.20
    gross = (closes - entry_price) / entry_price * 10_000.0 * (1 if side == 0 else -1)
    commission, slippage, initial_financing = 3.0, 4.0, -0.02 - side * 0.01
    liquidation = gross - commission - slippage + initial_financing
    clock = 1_000_000_000
    steps = [_marked_step(event="ENTRY", start=clock, end=clock,
                          hurdle_sha256=hurdle, gross=0.0)]
    relative_rewards, gammas, financing_costs = [], [], []
    for index, seconds in enumerate((60, 172800, 60)):
        financing = -(0.03 + side * 0.01) * seconds / 60.0
        risk = 0.01 * seconds / 60.0
        end = clock + seconds * 1_000_000_000
        step = _marked_step(event="HOLD", start=clock, end=end,
                            hurdle_sha256=hurdle, gross=0.0, financing=financing,
                            risk=risk, successor_value=float(liquidation[index + 1]))
        steps.append(step)
        old_composed = economics.compose_economic_step(step, contract=old)
        new_composed = economics.compose_economic_step(step, contract=new)
        assert new_composed == old_composed
        assert new_composed["gap"]["classification"] == (
            "continuous_m1" if seconds == 60 else "declared_market_closure"
        )
        gamma = new_composed["continuation_gamma"]
        reward = (new_composed["undiscounted_risk_adjusted_utility_increment_bps"]
                  + gamma * liquidation[index + 1] - liquidation[index])
        assert reward == pytest.approx(
            liquidation[index + 1] - liquidation[index] + financing - risk, abs=1e-11
        )
        relative_rewards.append(reward)
        gammas.append(gamma)
        financing_costs.append(financing)
        clock = end
    steps.append(_marked_step(event="EXIT_NOW", start=clock, end=clock,
                              hurdle_sha256=hurdle, gross=float(gross[-1]),
                              commission=commission, slippage=slippage,
                              financing=initial_financing))
    old_path = economics.evaluate_economic_path(steps, contract=old)
    new_path = economics.evaluate_economic_path(steps, contract=new)
    advantage = 0.0  # This fixed realized path closes at its final observed state.
    for reward, gamma in reversed(list(zip(relative_rewards, gammas))):
        advantage = reward + gamma * advantage
    expected_cash = gross[-1] - commission - slippage + initial_financing + sum(financing_costs)
    for path in (old_path, new_path):
        assert path["undiscounted_net_cash_pnl_bps"] == pytest.approx(expected_cash, abs=1e-11)
        assert path["discounted_risk_adjusted_utility_bps"] == pytest.approx(
            liquidation[0] + advantage, abs=1e-11
        )


@pytest.mark.parametrize("gap", [None, "weekend"], ids=["continuous", "declared-closure"])
def test_native_relative_val_pause_resume_preserves_cash_marked_metrics_and_coordinates(tmp_path, gap):
    thresholds = np.full((VAL_ENTRY_COHORT_SIZE, 2), 100.0, dtype=np.float32)
    thresholds[1, 0] = -1.0  # Exercise a real close as well as open valuation.
    model, representations, adapter, contract = _fixture(
        thresholds=thresholds, counts=np.full(VAL_ENTRY_COHORT_SIZE, 2, dtype=np.int64),
        gap=gap, reward_accounting=economics.LIQUIDATION_ADVANTAGE_REWARD_ACCOUNTING,
    )
    adapter.economic_step_provider.exit_gross_override = -40.0
    model = _with_route_outputs(model)
    binding = _checkpoint_binding(contract, adapter, tmp_path)
    common = dict(model=model, entry_decision_representations=representations,
                  adapter=adapter, checkpoint_binding=binding,
                  entry_policy_decisions=_entry_policy(adapter, binding),
                  entry_route_diagnostics={}, policy_batch_size=256,
                  progress_interval_forwards=16)
    paths = dict(progress_path=tmp_path / "resumed-progress.json",
                 result_path=tmp_path / "resumed-result.json")
    paused = run_resumable_random_access_val_evaluation_v1(
        **common, **paths, max_forwards_this_invocation=1,
    )
    assert paused["schema_version"] == PAUSE_SCHEMA_VERSION
    assert paused["next_entry_scan_position"] == 256
    persisted_pause = json.loads(paths["progress_path"].read_text())
    assert persisted_pause["trade_accumulators"][0][0]["undiscounted_net_cash_pnl_bps"] == -0.1
    resumed = run_resumable_random_access_val_evaluation_v1(
        **common, **paths, max_forwards_this_invocation=100,
    )
    direct = run_resumable_random_access_val_evaluation_v1(
        **common, progress_path=tmp_path / "direct-progress.json",
        result_path=tmp_path / "direct-result.json", max_forwards_this_invocation=100,
    )
    assert resumed["schema_version"] == LIQUIDATION_RELATIVE_RESULT_SCHEMA_VERSION
    assert resumed["exit_q_value_coordinates"] == "advantage_over_executable_liquidation_bps"
    for name in ("trade_outcomes", "marked_policy_evaluation", "entry_exit_policy_metrics",
                 "exit_policy_diagnostics", "model_forward_count"):
        assert resumed[name] == direct[name]
    first = resumed["trade_outcomes"][0]
    assert first["exit_state_index"] is None
    assert first["valuation"]["model_exit_executed"] is False
    assert first["undiscounted_net_cash_pnl_bps"] == -0.1
    assert first["valuation"]["remaining_liquidation_value_bps"] == -41.0
    assert first["valuation"]["net_cash_plus_open_value_bps"] == pytest.approx(-41.1)
    replay = resumed["marked_policy_evaluation"]["single_position_replay"]
    assert replay["executed_entry_row_indices"] == [0]
    assert replay["open_position_count"] == 1
    assert replay["full_cohort_authoritative"] is True
    assert replay["net_cash_plus_open_value_bps_sum"] == pytest.approx(-41.1)
    assert resumed["entry_exit_policy_metrics"]["net_bps_sum"] is None
    assert resumed["test_data_used"] is False
    validation = dict(rollout_contract_sha256=resumed["contract_sha256"],
                      checkpoint_binding_sha256=binding["binding_sha256"],
                      execution_contract_sha256=resumed["execution_contract_sha256"])
    assert require_random_access_val_evaluation_result_v1(resumed, **validation) == resumed
    require_complete_val_observation(resumed)
    for mutation in ("wrong-coordinates", "missing-coordinates", "legacy-schema"):
        changed = copy.deepcopy(resumed)
        if mutation == "wrong-coordinates":
            changed["exit_q_value_coordinates"] = "absolute_trade_value_bps"
        elif mutation == "missing-coordinates":
            changed.pop("exit_q_value_coordinates")
        else:
            changed["schema_version"] = MARKED_RESULT_SCHEMA_VERSION
        changed.pop("semantic_result_sha256")
        changed["semantic_result_sha256"] = canonical_sha256(changed)
        with pytest.raises(RuntimeError, match="VALUE_COORDINATES_INVALID"):
            require_random_access_val_evaluation_result_v1(changed, **validation)
        with pytest.raises(RuntimeError, match="VALUE_COORDINATES_INVALID"):
            require_complete_val_observation(changed)


def test_val_entry_anchor_adds_executable_first_liquidation_exactly_once(monkeypatch):
    from gx1.scripts import run_unified_exit_random_access_val_v1 as val

    provider, readiness = _provider(reward_accounting=economics.LIQUIDATION_ADVANTAGE_REWARD_ACCOUNTING)
    objective = readiness["economics_objective_contract"]
    expected = [economics.compose_economic_step(
        provider(0, side, "exit_now", 0, 1)["steps"][0], contract=objective,
    )["undiscounted_risk_adjusted_utility_increment_bps"] for side in (0, 1)]
    calls = []
    original = provider.materialize_training_projection

    def projected(entry, side, start, stop, hold_stop):
        calls.append((entry, side, start, stop, hold_stop))
        result = original(entry, side, start, stop, hold_stop)
        assert result["hold_reward_bps"].shape == (0,)
        assert result["exit_reward_bps"].shape == (1,)
        return result

    monkeypatch.setattr(provider, "materialize_training_projection", projected)
    entry = {"entry_row_index": 0, "entry_m1_start_row": 479,
             "entry_episode_binding_sha256": "1" * 64, "entry_fill_binding_sha256": "2" * 64}
    factory = SimpleNamespace(entries=[entry], economics_objective_contract=objective,
                              economic_step_provider=provider, normalization=_normalization(),
                              materialize_state=_state_provider)
    representations = torch.tensor([[100.0, -1.0]], dtype=torch.float32)
    with torch.inference_mode():
        targets, valid, _ = val._candidate_anchor_targets(
            target_model=_Policy().eval(),
            target_entry_output={val.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY: representations},
            state_factory=factory, child_rows=[0], device=torch.device("cpu"),
        )
    assert calls == [(0, 0, 0, 1, 0), (0, 1, 0, 1, 0)]
    torch.testing.assert_close(targets, torch.tensor([[expected[0] + 2.0, expected[1], 0.0]],
                                                    dtype=torch.float32), atol=1e-5, rtol=0.0)
    assert valid.tolist() == [[True, True, True]]
