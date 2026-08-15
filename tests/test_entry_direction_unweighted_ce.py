"""No hand-written direction distribution forcing anywhere in the objective.

Wave A/B retired the direction and hierarchical distribution-forcing terms.
V30 (2026-08-14) went further and retired the classification/cross-entropy
*authority* for direction entirely: the decision is now the unique argmax of
the learned raw-bps entry action-value head over the valid actions, trained by
masked mean-squared error, and no cross-entropy term scores or vetoes it.

This file is the only source-level guardrail that the retired implementation
does not come back, so it binds three surfaces at once:

1. the training-objective owner's own declaration,
2. the immutable recipe key set,
3. the trainer's executable source.

Every expectation is read from the owner; nothing is restated.
"""

from __future__ import annotations

import ast
from pathlib import Path

from gx1.contracts.entry_model_native_train_recipe_v1 import (
    MODEL_NATIVE_RECIPE_ENV,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    training_objective_contract_metadata,
)


TRAINER_PATH = (
    Path(__file__).resolve().parents[1]
    / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"
)

# Weighting/shaping keywords that would turn any cross-entropy term into a
# hand-written distribution-forcing device.
_FORBIDDEN_CE_KEYWORDS = frozenset(
    {"weight", "pos_weight", "class_weight", "label_smoothing", "ignore_index"}
)


def test_training_objective_declares_no_distribution_forcing() -> None:
    payload = training_objective_contract_metadata()

    assert payload["handwritten_distribution_forcing"] is False
    assert payload["handwritten_composite_weights"] is False
    assert payload["fixed_relative_task_weights"] is False
    # The retired cross-entropy authority: direction is decided by the learned
    # action-value argmax, never by a classification/probability loss.
    assert payload["classification_or_probability_loss_authority"] is False
    assert payload["entry_decision_authority"] == (
        "unique_argmax(entry_action_q_bps_over_valid_actions)"
    )


def test_recipe_has_no_direct_direction_distribution_forcing_keys() -> None:
    forbidden = {
        "ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT",
        "ENTRY_DIRECTION_CE_SCALE",
        "ENTRY_DIRECTION_CLASS_WEIGHT_CAP",
        "ENTRY_DIRECTION_LOGIT_ADJUST_TAU",
        "ENTRY_FLAT_CLASS_WEIGHT_FLOOR",
        "ENTRY_PRED_BALANCE_ALPHA",
        "ENTRY_PRED_BALANCE_CLASS_WEIGHTS",
        "ENTRY_PRED_BALANCE_TARGET",
        "ENTRY_TAIL_DIRECTION_CE_WEIGHT",
        "ENTRY_TAIL_DIRECTION_MIN_BATCH",
        "ENTRY_TAIL_DIRECTION_QUALITY_QUANTILE",
    }
    forbidden_prefixes = (
        "ENTRY_COST_",
        "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_",
        "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_",
        "ENTRY_DIRECTION_SLICE_RECALL_",
        "ENTRY_DIRECTION_SLICE_BALANCED_CE_",
        "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_",
        "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_",
        "ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_",
        "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_",
        "ENTRY_DIRECTION_VS_FLAT_",
        "ENTRY_DIRECTION_FLAT_STARVATION_",
    )

    assert not forbidden.intersection(MODEL_NATIVE_RECIPE_ENV)
    assert not any(
        key.startswith(forbidden_prefixes) for key in MODEL_NATIVE_RECIPE_ENV
    )


def test_every_trainer_cross_entropy_call_is_unweighted() -> None:
    """No surviving cross-entropy term may carry a weighting keyword.

    The retired guard counted a fixed number of direction CE calls, which went
    stale the moment the direction objective changed. This binds the invariant
    instead: whatever cross-entropy terms exist, none of them may be weighted,
    smoothed or class-masked.
    """

    tree = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))
    offending: list[tuple[int, str, list[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else ""
        )
        if "cross_entropy" not in name:
            continue
        used = [
            keyword.arg
            for keyword in node.keywords
            if keyword.arg in _FORBIDDEN_CE_KEYWORDS
        ]
        if used:
            offending.append((node.lineno, name, sorted(used)))

    assert offending == []


def test_trainer_has_no_retired_distribution_forcing_implementation() -> None:
    source = TRAINER_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "CostSensitiveCrossEntropyLoss",
        "_DirectionSliceBalancedSampler",
        "_build_cost_sensitive_criterion",
        "_direction_aux_ce_loss",
        "_direction_balance_term",
        "_direction_ckpt_balance_stats",
        "_direction_class_weights",
        "_direction_flat_starvation_term",
        "_direction_global_prior_match_term",
        "_direction_log_prior_offset",
        "_direction_logit_adjusted_ce_logits",
        "_direction_min_pred_rate_term",
        "_direction_slice_ckpt_score",
        "_direction_slice_hard_red_stop_ready",
        "_tail_direction_mask",
    ):
        assert forbidden not in source
