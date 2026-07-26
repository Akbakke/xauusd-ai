"""Profile-separated checkpoint admission and the rebalanced smoke objective.

User vedtak 2026-07-25 after V8 and V9 both produced class-degenerate
collapse while a plain unweighted-cross-entropy probe on the same substrate
cleared the majority baseline with all three classes alive:

1. smoke admits a checkpoint on active-head liveness plus non-degenerate
   class support; candidate keeps every acceptance gate exactly as before;
2. the training objective raises direction discrimination and lowers the
   distributional/conviction penalty tower, changing objective weights only.

No empirical acceptance threshold moves in either decision.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from gx1.contracts.entry_model_native_train_recipe_v1 import (
    MODEL_NATIVE_RECIPE_ENV,
)
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    _checkpoint_admission_ok,
    run_train,
)


def _admission(profile: str, **flags: bool) -> bool:
    base = {
        "aux_head_health_ok": True,
        "active_head_health_ok": True,
        "cooperation_gate_health_ok": True,
        "class_support_ok": True,
    }
    base.update(flags)
    return _checkpoint_admission_ok(profile=profile, **base)


def test_candidate_admission_still_requires_every_health_gate() -> None:
    assert _admission("candidate") is True
    for blocking in (
        "aux_head_health_ok",
        "active_head_health_ok",
        "cooperation_gate_health_ok",
    ):
        assert _admission("candidate", **{blocking: False}) is False


def test_smoke_admits_on_liveness_and_class_support_only() -> None:
    assert _admission("smoke") is True
    # Auxiliary and cooperation health stay diagnostic at smoke.
    assert _admission("smoke", aux_head_health_ok=False) is True
    assert _admission("smoke", cooperation_gate_health_ok=False) is True
    # Liveness and non-degenerate class support remain mandatory.
    assert _admission("smoke", active_head_health_ok=False) is False
    assert _admission("smoke", class_support_ok=False) is False


def test_class_support_is_not_a_candidate_shortcut() -> None:
    # Candidate never admits on class support alone.
    assert (
        _admission(
            "candidate",
            aux_head_health_ok=False,
            cooperation_gate_health_ok=False,
        )
        is False
    )


def test_unknown_profile_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="PROFILE_INVALID"):
        _admission("shadow")


def test_trainer_requires_the_exact_profile() -> None:
    assert "profile" in inspect.signature(run_train).parameters


def test_objective_rebalance_raises_discrimination_over_conviction() -> None:
    env = MODEL_NATIVE_RECIPE_ENV
    assert float(env["ENTRY_DIRECTION_CE_SCALE"]) == 12.00
    assert env["ENTRY_PRED_BALANCE_CLASS_WEIGHTS"] == "1.0,1.0,1.0"
    for key in (
        "ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT",
        "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT",
        "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT",
    ):
        assert float(env[key]) == 2.00
    # Direction cross-entropy must dominate every single penalty weight.
    ce_scale = float(env["ENTRY_DIRECTION_CE_SCALE"])
    for key in (
        "ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT",
        "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT",
        "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT",
        "ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT",
        "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT",
        "ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT",
    ):
        assert float(env[key]) <= ce_scale


def test_no_acceptance_threshold_moved_with_the_objective_rebalance() -> None:
    # Every empirical acceptance floor, policy and cost value stays fixed.
    env = MODEL_NATIVE_RECIPE_ENV
    assert env["ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE"] == "0.05"
    assert env["ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL"] == "0.35"
    assert env["ENTRY_CKPT_DIRECTION_SLICE_GUARD"] == "1"
    assert env["ENTRY_SPECIALIST_GATE_MIN_MEAN"] == "0.01"
    assert env["ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS"] == "6"
    assert env["ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE"] == "3"
    assert env["ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR"] == "0.05"
    assert env["ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR"] == "0.05"
    assert env["ENTRY_COST_LONG_TO_FLAT"] == "0.45"
    assert env["ENTRY_COST_FLAT_TO_LONG"] == "1.60"
    assert env["ENTRY_COST_SHORT_TO_FLAT"] == "0.45"
    assert env["ENTRY_COST_FLAT_TO_SHORT"] == "1.60"


def _bundle_gate(profile: str) -> bool:
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    return trainer._bundle_write_acceptance_gates_required(profile)


def test_candidate_still_refuses_a_slice_failed_bundle() -> None:
    assert _bundle_gate("candidate") is True


def test_smoke_writes_the_bundle_as_trainability_evidence() -> None:
    # Vedtak A is only fulfilled if a smoke run yields a measurable artifact.
    # The failure evidence file is still written for both profiles; only the
    # raise is separated.
    assert _bundle_gate("smoke") is False


def test_bundle_gate_rejects_an_unknown_profile() -> None:
    with pytest.raises(RuntimeError, match="PROFILE_INVALID"):
        _bundle_gate("shadow")


def test_class_balance_bundle_guard_is_not_profile_separated() -> None:
    """Non-degenerate class support is what smoke requires, so that gate stays.

    Only the direction-slice contract — an acceptance metric — is separated.
    """
    trainer_path = (
        Path(__file__).resolve().parents[1]
        / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"
    )
    source = trainer_path.read_text(encoding="utf-8")
    balance_raise = source.index("TRAIN_FAIL_DIRECTION_CLASS_BALANCE_GUARD")
    window = source[max(0, balance_raise - 2000) : balance_raise]
    assert "_bundle_write_acceptance_gates_required" not in window
