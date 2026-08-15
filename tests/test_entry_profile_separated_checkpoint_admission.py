"""Profile-separated checkpoint admission under the plain direction objective.

User vedtak 2026-07-25 after V8 and V9 both produced class-degenerate
collapse while a plain unweighted-cross-entropy probe on the same substrate
cleared the majority baseline with all three classes alive:

1. ``smoke`` admits a checkpoint on active-head liveness; ``candidate`` keeps
   every acceptance gate exactly as before;
2. direction distribution diagnostics remain read-only and cannot change the
   unweighted cross-entropy loss or checkpoint score.

No empirical acceptance threshold moves in either decision.

V30 (2026-08-14) retired the hand-written checkpoint class-balance guard, the
direction-slice acceptance contract and the ``aux_head_health_ok`` /
``class_support_ok`` admission inputs together with the recipe keys that
configured them. The admission surface is now exactly the three health gates
the trainer computes, and this file binds that surface by signature rather
than by a restated keyword list.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from gx1.contracts.entry_model_native_train_recipe_v1 import (
    MODEL_NATIVE_RECIPE_ENV,
)
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    _checkpoint_admission_ok,
    run_train,
)


def _admission_health_keywords() -> tuple[str, ...]:
    """Derive the admission health inputs from the owner's own signature."""

    parameters = inspect.signature(_checkpoint_admission_ok).parameters
    names = tuple(
        name
        for name, parameter in parameters.items()
        if name != "profile"
        and parameter.kind is inspect.Parameter.KEYWORD_ONLY
    )
    assert names, "admission owner exposes no health keyword"
    return names


def _admission(profile: str, **flags: Any) -> bool:
    base = {name: True for name in _admission_health_keywords()}
    unknown = sorted(set(flags) - set(base))
    assert not unknown, f"test used non-owner admission keywords: {unknown}"
    base.update(flags)
    return _checkpoint_admission_ok(profile=profile, **base)


def test_admission_takes_only_keyword_health_evidence() -> None:
    parameters = inspect.signature(_checkpoint_admission_ok).parameters
    assert "profile" in parameters
    positional = [
        name
        for name, parameter in parameters.items()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    assert positional == []
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in parameters.values()
    ), "admission health evidence must never carry a default"


def test_candidate_admission_still_requires_every_health_gate() -> None:
    assert _admission("candidate") is True
    for blocking in _admission_health_keywords():
        assert _admission("candidate", **{blocking: False}) is False


def test_smoke_admits_on_active_head_liveness_only() -> None:
    assert _admission("smoke") is True
    # Cooperation health stays diagnostic at smoke.
    assert _admission("smoke", cooperation_gate_health_ok=False) is True
    assert _admission("smoke", exit_cooperation_gate_health_ok=False) is True
    # Active-head liveness remains mandatory.
    assert _admission("smoke", active_head_health_ok=False) is False


def test_cooperation_health_is_not_a_candidate_shortcut() -> None:
    # Candidate never admits while any single health gate is red.
    assert (
        _admission(
            "candidate",
            active_head_health_ok=False,
            cooperation_gate_health_ok=False,
        )
        is False
    )


def test_unknown_profile_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="PROFILE_INVALID"):
        _admission("shadow")


def test_trainer_requires_the_exact_profile() -> None:
    assert "profile" in inspect.signature(run_train).parameters


def test_direction_objective_has_no_handwritten_distribution_forcing() -> None:
    env = MODEL_NATIVE_RECIPE_ENV
    forbidden = (
        "ENTRY_DIRECTION_CE_SCALE",
        "ENTRY_PRED_BALANCE_CLASS_WEIGHTS",
        "ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT",
        "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT",
        "ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT",
        "ENTRY_DIRECTION_LOGIT_ADJUST_TAU",
        "ENTRY_TAIL_DIRECTION_CE_WEIGHT",
        "ENTRY_COST_LONG_TO_FLAT",
        # Learned gate routing is admitted from empirical liveness evidence;
        # the recipe must not impose a hand-written target share.
        "ENTRY_SPECIALIST_GATE_MIN_MEAN",
        # V30 retired the hand-written checkpoint class-balance and
        # direction-slice acceptance thresholds; they may not come back.
        "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE",
        "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL",
        "ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE",
        "ENTRY_DIRECTION_SLICE_MIN_ROWS",
    )
    assert not set(forbidden) & set(env)
