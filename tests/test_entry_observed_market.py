from __future__ import annotations

import copy

import pytest
import torch

from gx1.contracts.entry_observed_market_v1 import (
    GROSS_TARGET_COLUMNS, build_entry_observed_market_targets,
    entry_observed_market_contract, entry_observed_market_economics,
    require_entry_observed_market_economics,
)
from gx1.contracts.unified_exit_economics_objective_v2 import SECONDS_PER_YEAR
from tests.test_unified_exit_prospective_cost_policy_v1 import START, END, _build, _load


@pytest.fixture
def economics(tmp_path):
    authority = _load(_build(tmp_path), "parameter_authority")
    return entry_observed_market_economics(
        authority, coverage_start_utc=START, coverage_end_utc=END,
        verify_local_sources=False,
    )


def test_market_contract_is_teacher_free_and_preserves_existing_targets():
    value = entry_observed_market_contract()
    assert value["gross_target_columns"] == list(GROSS_TARGET_COLUMNS)
    assert value["action_order"] == ["LONG", "SHORT", "FLAT"]
    assert value["exit_model_used"] is False
    assert value["hindsight_best_exit_used"] is False
    assert value["holding_time_limit"] is False
    assert value["score_is_calibrated_probability"] is False
    assert value["retained_forecast_horizons_m5"] == [1, 5, 12, 24]


def test_net_markouts_preserve_signed_losses_exact_flat_and_elapsed_cost(economics):
    gross = torch.tensor([[12.0, -14.0], [-2.0, 1.0], [3.0, -4.0]], requires_grad=True)
    elapsed = torch.tensor([300, 5700, 172800], dtype=torch.int64)
    result = build_entry_observed_market_targets(
        gross_return_bps=gross, elapsed_wall_clock_seconds=elapsed, economics=economics,
    )
    expected = torch.tensor([
        [12.0 - 4.0 - 540.0 * 300 / SECONDS_PER_YEAR, -18.0, 0.0],
        [-2.0 - 4.0 - 540.0 * 5700 / SECONDS_PER_YEAR, -3.0, 0.0],
        [3.0 - 4.0 - 540.0 * 172800 / SECONDS_PER_YEAR, -8.0, 0.0],
    ])
    assert torch.equal(result["target_bps"], expected)
    assert bool(result["target_valid"].all())
    assert not result["target_bps"].requires_grad
    assert gross.grad is None
    assert torch.equal(gross.detach(), torch.tensor([[12.0, -14.0], [-2.0, 1.0], [3.0, -4.0]]))


@pytest.mark.parametrize("defect", ["nan", "wrong_side_count", "zero_elapsed", "float_elapsed", "negative_elapsed"])
def test_incomplete_or_invalid_outcomes_never_become_flat_targets(economics, defect):
    gross = torch.tensor([[1.0, -2.0]])
    elapsed = torch.tensor([300], dtype=torch.int64)
    if defect == "nan": gross[0, 0] = float("nan")
    elif defect == "wrong_side_count": gross = gross[:, :1]
    elif defect == "zero_elapsed": elapsed[0] = 0
    elif defect == "negative_elapsed": elapsed[0] = -1
    else: elapsed = elapsed.float()
    with pytest.raises(RuntimeError, match="TARGET_INPUT_INVALID"):
        build_entry_observed_market_targets(
            gross_return_bps=gross, elapsed_wall_clock_seconds=elapsed, economics=economics,
        )


@pytest.mark.parametrize("defect", ["execution_cost", "financing", "truth_claim", "extra_teacher"])
def test_economics_drift_is_rejected(economics, defect):
    changed = copy.deepcopy(economics)
    if defect == "execution_cost": changed["round_trip_execution_cost_bps"] = 0.0
    elif defect == "financing": changed["annual_financing_cost_bps"][0] = 0.0
    elif defect == "truth_claim": changed["historical_cost_truth_qualified"] = True
    else: changed["exit_target_model_sha256"] = "a" * 64
    with pytest.raises(RuntimeError, match="ECONOMICS_INVALID"):
        require_entry_observed_market_economics(changed)


def test_net_target_does_not_depend_on_exit_state_or_torch_rng(economics):
    gross = torch.tensor([[10.0, -12.0], [-8.0, 6.0]])
    elapsed = torch.tensor([5700, 5700], dtype=torch.int64)
    first = build_entry_observed_market_targets(
        gross_return_bps=gross, elapsed_wall_clock_seconds=elapsed, economics=economics,
    )
    torch.manual_seed(109)
    # The target API admits no model/teacher state, and consumes no random draw.
    before = torch.get_rng_state().clone()
    second = build_entry_observed_market_targets(
        gross_return_bps=gross, elapsed_wall_clock_seconds=elapsed, economics=economics,
    )
    assert torch.equal(torch.get_rng_state(), before)
    assert all(torch.equal(first[k], second[k]) for k in first)
