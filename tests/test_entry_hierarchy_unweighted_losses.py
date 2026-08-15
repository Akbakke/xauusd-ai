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


# The Entry hierarchy (trade / side / bad-path / side-validity heads and the
# ``_hierarchical_entry_task_losses`` owner) is retired: the Entry action
# target is the frozen fitted-Q teacher and the decision authority is the
# unique argmax of ``entry_action_q_bps``.  The rule the hierarchy tests
# enforced — plain unweighted classification losses, no distribution forcing —
# still binds every surviving binary task.


def test_binary_classification_is_exact_and_unweighted_everywhere() -> None:
    tree = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))
    bce_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "binary_cross_entropy_with_logits"
    ]

    assert bce_calls, "no binary classification loss found in the trainer"
    for call in bce_calls:
        # Two positional args (logits, targets) and no class/sample weighting.
        assert len(call.args) == 2
        assert {keyword.arg for keyword in call.keywords} <= {"reduction"}
        for keyword in call.keywords:
            if keyword.arg == "reduction":
                assert ast.literal_eval(keyword.value) == "none"


def test_objective_contract_rejects_distribution_forcing_surface() -> None:
    payload = training_objective_contract_metadata()

    assert payload["handwritten_distribution_forcing"] is False
    assert payload["handwritten_composite_weights"] is False
    assert payload["handwritten_rank_losses"] is False
    assert payload["fixed_relative_task_weights"] is False
    assert payload["classification_or_probability_loss_authority"] is False
    assert payload["entry_decision_authority"] == (
        "unique_argmax(entry_action_q_bps_over_valid_actions)"
    )
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
