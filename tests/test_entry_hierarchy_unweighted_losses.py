from __future__ import annotations

import ast
from pathlib import Path

from gx1.contracts.entry_model_native_train_recipe_v1 import (
    MODEL_NATIVE_RECIPE_ENV,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    training_objective_contract_metadata,
)


REPO = Path(__file__).resolve().parents[1]
TRAINER_PATH = REPO / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"
RECIPE_PATH = REPO / "gx1/contracts/entry_model_native_train_recipe_v1.py"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _functional_calls(function: ast.FunctionDef, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]


def test_hierarchy_classification_formula_calls_are_exact_and_unweighted() -> None:
    tree = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))
    function = _function(tree, "_hierarchical_entry_task_losses")

    bce_calls = _functional_calls(function, "binary_cross_entropy_with_logits")
    ce_calls = _functional_calls(function, "cross_entropy")

    assert [tuple(ast.unparse(arg) for arg in call.args) for call in bce_calls] == [
        ("trade_logit.squeeze(1)", "y_trade"),
        ("side_bad_path_logit", "bad_target"),
        ("side_validity_logit", "validity_target"),
    ]
    assert all(not call.keywords for call in bce_calls)
    assert [tuple(ast.unparse(arg) for arg in call.args) for call in ce_calls] == [
        ("side_logits[y_side_mask]", "y_side[y_side_mask]"),
    ]
    assert not ce_calls[0].keywords


def test_auxiliary_binary_classification_is_unweighted_in_train_and_validation() -> (
    None
):
    tree = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))
    expected_first_args = {
        "tradable_logit.squeeze(1)[selector_mask]",
        "bad_path_logit.squeeze(1)[selector_mask]",
        "clean_edge_logit.squeeze(1)[selector_mask]",
        "survival_logit.squeeze(1)[selector_mask]",
    }

    for function_name in ("train_epoch", "validate"):
        function = _function(tree, function_name)
        calls = _functional_calls(function, "binary_cross_entropy_with_logits")
        aux_calls = [
            call
            for call in calls
            if call.args and ast.unparse(call.args[0]) in expected_first_args
        ]
        assert len(aux_calls) == len(expected_first_args)
        assert {ast.unparse(call.args[0]) for call in aux_calls} == expected_first_args
        assert all(len(call.args) == 2 and not call.keywords for call in aux_calls)


def test_hierarchy_contract_rejects_distribution_forcing_surface() -> None:
    payload = training_objective_contract_metadata()

    assert payload["handwritten_hierarchical_distribution_forcing"] is False
    assert payload["hier_trade_loss"] == "unweighted_binary_cross_entropy"
    assert payload["hier_side_loss"] == "masked_unweighted_cross_entropy"
    assert payload["hier_bad_path_loss"] == "unweighted_binary_cross_entropy"
    assert payload["hier_side_validity_loss"] == "unweighted_binary_cross_entropy"
    assert not {
        key for key in MODEL_NATIVE_RECIPE_ENV if key.startswith("ENTRY_HIER_")
    }
    assert not any(key.endswith("_POS_WEIGHT_CAP") for key in MODEL_NATIVE_RECIPE_ENV)


def test_retired_hierarchy_distribution_forcing_cannot_reenter_sources() -> None:
    trainer_source = TRAINER_PATH.read_text(encoding="utf-8")
    recipe_source = RECIPE_PATH.read_text(encoding="utf-8")
    forbidden = (
        "PriorMatch",
        "_batch_rate_sampling_floor",
        "_hier_trade_global_prior_match_term",
        "_hier_slice_trade_prior_match_term",
        "_hier_slice_trade_accuracy_edge_term",
        "_hier_flat_logit_margin_term",
        "_hier_slice_flat_logit_margin_term",
        "_hier_side_global_prior_match_term",
        "_hier_slice_side_prior_match_term",
        "_hier_slice_side_balanced_ce_term",
        "_hier_slice_side_true_margin_term",
        "_hier_slice_side_accuracy_edge_term",
        "ENTRY_HIER_PRED_RATE_SOFTMAX_TEMPERATURE",
        "ENTRY_HIER_BAD_PATH_POS_WEIGHT_CAP",
        "ENTRY_AUX_BAD_PATH_POS_WEIGHT_CAP",
        "ENTRY_AUX_CLEAN_EDGE_POS_WEIGHT_CAP",
        "ENTRY_AUX_SURVIVAL_POS_WEIGHT_CAP",
        "ENTRY_AUX_TRADABLE_POS_WEIGHT_CAP",
        "pos_weight=",
    )
    for marker in forbidden:
        assert marker not in trainer_source
        assert marker not in recipe_source
