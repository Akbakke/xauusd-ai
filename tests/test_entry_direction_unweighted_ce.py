from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from gx1.contracts.entry_model_native_train_recipe_v1 import (
    MODEL_NATIVE_RECIPE_ENV,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    training_objective_contract_metadata,
)
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


TRAINER_PATH = (
    Path(__file__).resolve().parents[1]
    / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"
)


def test_training_objective_declares_plain_main_and_mtf_ce() -> None:
    payload = training_objective_contract_metadata()

    assert payload["handwritten_direction_distribution_forcing"] is False
    assert payload["main_direction_loss"] == "unweighted_cross_entropy"
    assert payload["mtf_direction_loss"] == "unweighted_cross_entropy"


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


def test_trainer_main_and_mtf_ce_calls_are_unweighted_in_train_and_val() -> None:
    tree = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))
    exact_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "cross_entropy"
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in {"logits", "mtf_dir_logits"}
            and isinstance(node.args[1], ast.Name)
            and node.args[1].id == "y"
        ):
            continue
        exact_calls.append(node)

    assert len(exact_calls) == 4
    assert all(len(call.args) == 2 and not call.keywords for call in exact_calls)


def test_trainer_has_no_retired_distribution_forcing_implementation() -> None:
    source = TRAINER_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "CostSensitiveCrossEntropyLoss",
        "_DirectionSliceBalancedSampler",
        "_build_cost_sensitive_criterion",
        "_direction_aux_ce_loss",
        "_direction_balance_term",
        "_direction_class_weights",
        "_direction_log_prior_offset",
        "_direction_logit_adjusted_ce_logits",
        "_direction_slice_ckpt_score",
        "_direction_slice_hard_red_stop_ready",
        "_tail_direction_mask",
    ):
        assert forbidden not in source


def test_direction_distribution_diagnostics_cannot_score_or_change_loss() -> None:
    stats = trainer._direction_ckpt_balance_stats(
        np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
        np.asarray([0, 0, 2, 0, 1, 2], dtype=np.int64),
    )

    assert stats["direction_distribution_diagnostic_only"] is True
    assert stats["direction_distribution_affects_loss"] is False
    assert stats["direction_distribution_affects_checkpoint_score"] is False
    assert "direction_ckpt_score" not in stats
    assert "direction_ckpt_balance_penalty" not in stats
