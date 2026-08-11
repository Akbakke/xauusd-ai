from __future__ import annotations

import ast
import json
from pathlib import Path

import torch

from gx1.contracts.entry_model_native_train_recipe_v1 import (
    MODEL_NATIVE_RECIPE_ENV,
)


TRAINER_PATH = (
    Path(__file__).resolve().parents[1]
    / "gx1"
    / "models"
    / "entry_v10"
    / "entry_v10_ctx_train_v3.py"
)


def _trainer_ast() -> ast.Module:
    return ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))


def _env_str_defaults(module: ast.Module) -> dict[str, str]:
    """Return the recipe-owned value for every key the trainer reads.

    The trainer used to hold a literal default at each call site; those 160
    literals were a shadow contract with 62 drifts. The recipe owner is now the
    single origin, so the value a trainer key resolves to is exactly the recipe
    entry, and these tests assert against that owner.
    """
    keys = {
        str(node.args[0].value)
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_env_str"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
    }
    assert keys, "trainer reads no recipe keys"
    return {key: MODEL_NATIVE_RECIPE_ENV[key] for key in keys}


def test_entry_training_requires_live_unified_exit_supervision_and_proof() -> None:
    source = TRAINER_PATH.read_text(encoding="utf-8")
    run_train_source = source.split("def run_train(", 1)[1]

    assert "_require_unified_exit_training_evidence" not in source
    assert "--unified-exit-lifecycle-manifest-json" in source
    assert run_train_source.index(
        "UnifiedExitLifecycleCorpus("
    ) < run_train_source.index("optim.AdamW(")
    assert source.index(
        "unified_exit_loss, unified_exit_stats = _unified_exit_action_loss("
    ) < source.index("\n            loss.backward()")
    assert run_train_source.index(
        "_unified_exit_movement_proof("
    ) < run_train_source.index("torch.save(best_state")


def test_unified_exit_loss_backpropagates_into_shared_entry_and_exit_head() -> None:
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    class _ExitProbe(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = torch.nn.Linear(4 + 4 + 14 + 2 + 5, 2)

        def forward_exit_action(
            self,
            *,
            entry_shared_representation,
            exit_feature_seq_x,
            exit_feature_snap_x,
            exit_feature_ctx_cat,
            exit_feature_ctx_cont,
            exit_seq_m5,
            exit_seq_m15,
            exit_seq_h1,
            exit_seq_h4,
            exit_seq_d1,
            exit_path_x,
            exit_path_lengths,
            exit_side_index,
        ):
            row = torch.arange(exit_path_x.shape[0])
            last = exit_path_x[row, exit_path_lengths.long() - 1]
            side = torch.nn.functional.one_hot(
                exit_side_index.long(),
                num_classes=2,
            ).float()
            mtf = torch.stack(
                [
                    tensor[:, -1, :].mean(dim=1)
                    for tensor in (
                        exit_seq_m5,
                        exit_seq_m15,
                        exit_seq_h1,
                        exit_seq_h4,
                        exit_seq_d1,
                    )
                ],
                dim=1,
            )
            logits = self.head(
                torch.cat(
                    [
                        entry_shared_representation,
                        exit_feature_snap_x,
                        last,
                        side,
                        mtf,
                    ],
                    dim=1,
                )
            )
            return {"exit_action_logits": logits}

    model = _ExitProbe()
    shared = torch.randn(1, 4, requires_grad=True)
    paths = torch.zeros(1, 4, 3, 14)
    paths[0, 0, :2] = 0.25
    paths[0, 1, :3] = -0.25
    paths[0, 2, :1] = 0.50
    paths[0, 3, :2] = -0.50
    batch = {
        "exit_feature_seq_x": torch.randn(1, 4, 4, 4),
        "exit_feature_snap_x": torch.randn(1, 4, 4),
        "exit_feature_ctx_cat": torch.zeros(1, 4, 5, dtype=torch.long),
        "exit_feature_ctx_cont": torch.randn(1, 4, 142),
        **{
            f"exit_seq_{tf}": torch.randn(1, 4, 2, 3, requires_grad=True)
            for tf in ("m5", "m15", "h1", "h4", "d1")
        },
        "exit_path_x": paths,
        "exit_path_lengths": torch.tensor([[2, 3, 1, 2]]),
        "exit_side_index": torch.tensor([[0, 1, 0, 1]]),
        "exit_action_target": torch.tensor([[0, 1, 0, 1]]),
        "exit_sample_valid": torch.tensor([[True, True, True, True]]),
    }
    loss, stats = trainer._unified_exit_action_loss(
        model,
        {"shared_feature_representation": shared},
        batch,
        torch.device("cpu"),
    )
    loss.backward()

    assert float(loss) > 0.0
    assert stats["rows"] == 4
    assert stats["hold_rows"] == 2
    assert stats["exit_now_rows"] == 2
    assert shared.grad is not None
    assert float(shared.grad.abs().sum()) > 0.0
    assert model.head.weight.grad is not None
    assert float(model.head.weight.grad.abs().sum()) > 0.0


def test_entry_v10_env_reads_are_owned_by_the_exact_recipe_contract() -> None:
    module = _trainer_ast()
    env_defaults = _env_str_defaults(module)
    assert set(env_defaults).issubset(MODEL_NATIVE_RECIPE_ENV)
    assert "_CANONICAL_ENTRY_TRAIN_ENV_DEFAULTS" not in TRAINER_PATH.read_text(
        encoding="utf-8"
    )


def test_entry_v10_model_has_no_hidden_mtf_or_capacity_defaults() -> None:
    import inspect

    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as model_module
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    model_signature = inspect.signature(
        model_module.EntryV10CtxHybridTransformer.__init__
    )
    for name in (
        "dropout",
        "m5_seq_len",
        "m15_seq_len",
        "h1_seq_len",
        "h4_seq_len",
        "d1_seq_len",
        "multi_tf_num_layers",
        "multi_tf_scale",
        "specialist_num_layers",
        "specialist_fusion_scale",
        "cross_family_fusion_scale",
    ):
        assert model_signature.parameters[name].default is inspect.Parameter.empty
    config_signature = inspect.signature(model_module.CtxModelConfig)
    for name in (
        "dropout",
        "m5_seq_len",
        "m15_seq_len",
        "h1_seq_len",
        "h4_seq_len",
        "d1_seq_len",
        "multi_tf_num_layers",
        "multi_tf_scale",
        "specialist_num_layers",
        "specialist_fusion_scale",
        "cross_family_fusion_scale",
    ):
        assert config_signature.parameters[name].default is inspect.Parameter.empty
    assert (
        inspect.signature(trainer.train_epoch)
        .parameters["grad_accum_steps"]
        .default
        is inspect.Parameter.empty
    )


def test_entry_v10_bad_path_and_path_quality_evidence_are_trained() -> None:
    env_defaults = _env_str_defaults(_trainer_ast())
    assert env_defaults["ENTRY_AUX_BAD_PATH_WEIGHT"] == "1.25"
    assert env_defaults["ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT"] == "2.00"
    assert env_defaults["ENTRY_PATH_QUALITY_RANK_WEIGHT"] == "2.00"
    assert env_defaults["ENTRY_OFFLINE_RL_Q_WEIGHT"] == "0.50"
    assert env_defaults["ENTRY_OFFLINE_RL_V_WEIGHT"] == "0.20"
    assert env_defaults["ENTRY_OFFLINE_RL_RANK_WEIGHT"] == "0.05"


def test_entry_v10_pred_balance_loss_runs_when_cost_sensitive_is_disabled() -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    logits = torch.tensor(
        [
            [6.0, -3.0, -3.0],
            [6.0, -3.0, -3.0],
            [6.0, -3.0, -3.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 2], dtype=torch.long)
    criterion = trainer.CostSensitiveCrossEntropyLoss(
        class_weights=None,
        cost_matrix=torch.zeros((3, 3), dtype=torch.float32),
        cost_scale=0.0,
        enabled=False,
        balance_alpha=0.20,
        balance_target="label",
        balance_class_weights=torch.tensor([1.0, 1.0, 4.0], dtype=torch.float32),
    )

    ce_only = criterion.ce(logits, targets).mean()
    loss = criterion(logits, targets)
    balance_term = trainer._direction_balance_term(torch.softmax(logits, dim=1), targets, criterion)

    assert float(balance_term.item()) > 0.0
    assert float(loss.item()) > float(ce_only.item())


def test_entry_v10_train_and_validate_apply_pred_balance_loss_directly() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")
    assert text.count("_direction_balance_term(probs, y, criterion)") >= 2


def test_entry_v10_direction_min_pred_rate_term_penalizes_active_class_collapse(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT", 2.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION", 0.50)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR", 0.05)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE", 1.0)

    targets = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    collapsed = torch.tensor(
        [
            [0.01, 0.01, 0.98],
            [0.01, 0.01, 0.98],
            [0.01, 0.01, 0.98],
            [0.01, 0.01, 0.98],
            [0.01, 0.01, 0.98],
            [0.01, 0.01, 0.98],
        ],
        dtype=torch.float32,
    )
    no_flat = torch.tensor(
        [
            [0.48, 0.50, 0.02],
            [0.48, 0.50, 0.02],
            [0.48, 0.50, 0.02],
            [0.48, 0.50, 0.02],
            [0.48, 0.50, 0.02],
            [0.48, 0.50, 0.02],
        ],
        dtype=torch.float32,
    )
    balanced_enough = torch.tensor(
        [
            [0.20, 0.20, 0.60],
            [0.20, 0.20, 0.60],
            [0.20, 0.20, 0.60],
            [0.20, 0.20, 0.60],
            [0.20, 0.20, 0.60],
            [0.20, 0.20, 0.60],
        ],
        dtype=torch.float32,
    )

    assert float(trainer._direction_min_pred_rate_term(collapsed, targets).item()) > 0.0
    assert float(trainer._direction_min_pred_rate_term(no_flat, targets).item()) > 0.0
    assert float(trainer._direction_min_pred_rate_term(balanced_enough, targets).item()) == 0.0


def test_entry_v10_direction_min_pred_rate_temperature_tracks_argmax_collapse(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_MIN_PRED_RATE_LOSS_WEIGHT", 2.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_MIN_PRED_RATE_FRACTION", 0.50)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_MIN_PRED_RATE_FLOOR", 0.05)

    targets = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    same_argmax = torch.tensor(
        [
            [0.45, 0.35, 0.20],
            [0.45, 0.35, 0.20],
            [0.45, 0.35, 0.20],
            [0.45, 0.35, 0.20],
            [0.45, 0.35, 0.20],
            [0.45, 0.35, 0.20],
        ],
        dtype=torch.float32,
    )

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE", 1.0)
    assert float(trainer._direction_min_pred_rate_term(same_argmax, targets).item()) == 0.0

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE", 0.10)
    assert float(trainer._direction_min_pred_rate_term(same_argmax, targets).item()) > 0.0


def test_entry_v10_direction_slice_min_pred_rate_term_penalizes_dead_slice_class(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT", 3.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION", 0.50)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR", 0.05)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE", 1.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_MIN_ROWS", 3)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES", "0")

    targets = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    ctx_cat = torch.tensor([[2], [2], [2], [2], [2], [2]], dtype=torch.long)
    collapsed = torch.tensor(
        [
            [0.48, 0.50, 0.02],
            [0.48, 0.50, 0.02],
            [0.48, 0.50, 0.02],
            [0.48, 0.50, 0.02],
            [0.48, 0.50, 0.02],
            [0.48, 0.50, 0.02],
        ],
        dtype=torch.float32,
    )
    covered = torch.tensor(
        [
            [0.34, 0.33, 0.33],
            [0.34, 0.33, 0.33],
            [0.34, 0.33, 0.33],
            [0.34, 0.33, 0.33],
            [0.34, 0.33, 0.33],
            [0.34, 0.33, 0.33],
        ],
        dtype=torch.float32,
    )

    assert float(trainer._direction_slice_min_pred_rate_term(collapsed, targets, ctx_cat).item()) > 0.0
    assert float(trainer._direction_slice_min_pred_rate_term(covered, targets, ctx_cat).item()) == 0.0


def test_entry_v10_direction_slice_loss_mean_max_weights_worst_slice(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_LOSS_WEIGHT", 1.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FRACTION", 0.50)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_MIN_PRED_RATE_FLOOR", 0.05)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE", 1.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_MIN_ROWS", 3)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES", "0")

    targets = torch.tensor([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2], dtype=torch.long)
    ctx_cat = torch.tensor([[0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [1]], dtype=torch.long)
    probs = torch.tensor(
        [
            [0.48, 0.50, 0.02],
            [0.48, 0.50, 0.02],
            [0.48, 0.50, 0.02],
            [0.48, 0.50, 0.02],
            [0.48, 0.50, 0.02],
            [0.48, 0.50, 0.02],
            [0.34, 0.33, 0.33],
            [0.34, 0.33, 0.33],
            [0.33, 0.34, 0.33],
            [0.33, 0.34, 0.33],
            [0.33, 0.33, 0.34],
            [0.33, 0.33, 0.34],
        ],
        dtype=torch.float32,
    )

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION", "mean")
    mean_loss = float(trainer._direction_slice_min_pred_rate_term(probs, targets, ctx_cat).item())
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION", "mean_max")
    mean_max_loss = float(trainer._direction_slice_min_pred_rate_term(probs, targets, ctx_cat).item())

    assert mean_loss > 0.0
    assert mean_max_loss > mean_loss


def test_entry_v10_direction_slice_balance_stats_penalizes_audit_slice_collapse(monkeypatch) -> None:
    import numpy as np

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_MIN_ROWS", 4)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES", "0")
    monkeypatch.setattr(trainer, "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE", 0.05)
    monkeypatch.setattr(trainer, "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL", 0.35)

    labels = np.asarray([0, 0, 1, 1, 2, 2] * 12, dtype=np.int64)
    collapsed_preds = np.asarray([0, 0, 0, 0, 0, 0] * 12, dtype=np.int64)
    covered_preds = labels.copy()
    ctx_cat = np.asarray([[2]] * len(labels), dtype=np.int64)

    collapsed = trainer._direction_slice_balance_stats(labels, collapsed_preds, ctx_cat)
    covered = trainer._direction_slice_balance_stats(labels, covered_preds, ctx_cat)

    assert collapsed["direction_slice_audited_count"] == 1
    assert collapsed["direction_slice_failure_count"] > 0
    assert collapsed["direction_slice_pred_rate_shortfall"] > 0.0
    assert collapsed["direction_slice_failure_details"]
    assert collapsed["direction_slice_failure_details"][0]["ctx_cat_index"] == 0
    assert collapsed["direction_slice_failure_details"][0]["accuracy_failed"] is True
    assert covered["direction_slice_failure_count"] == 0
    assert covered["direction_slice_failure_details"] == []
    assert trainer._direction_slice_ckpt_score(0.40, collapsed) < trainer._direction_slice_ckpt_score(0.40, covered)


def test_entry_v10_direction_slice_balance_stats_attaches_hierarchy_diagnostics(monkeypatch) -> None:
    import numpy as np

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_MIN_ROWS", 4)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_MIN_LABEL_RATE", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES", "0")
    monkeypatch.setattr(trainer, "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE", 0.05)
    monkeypatch.setattr(trainer, "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL", 0.35)

    labels = np.asarray([0, 0, 1, 1, 2, 2] * 12, dtype=np.int64)
    collapsed_preds = np.asarray([0, 0, 0, 0, 0, 0] * 12, dtype=np.int64)
    ctx_cat = np.asarray([[2]] * len(labels), dtype=np.int64)
    trade_prob = np.asarray([0.80, 0.82, 0.84, 0.86, 0.94, 0.96] * 12, dtype=np.float64)
    side_pred = np.asarray([0, 0, 1, 1, 0, 1] * 12, dtype=np.int64)
    side_long_prob = np.asarray([0.80, 0.78, 0.20, 0.18, 0.70, 0.30] * 12, dtype=np.float64)

    stats = trainer._direction_slice_balance_stats(
        labels,
        collapsed_preds,
        ctx_cat,
        trade_prob_np=trade_prob,
        side_pred_np=side_pred,
        side_long_prob_np=side_long_prob,
    )
    detail = stats["direction_slice_failure_details"][0]

    assert detail["hier_trade_target_rate"] == 4 / 6
    assert detail["hier_trade_pred_rate"] == 1.0
    assert detail["hier_trade_prob_label_flat_mean"] > 0.90
    assert detail["hier_side_pred_long_rate_on_edge"] == 0.5
    assert detail["hier_side_acc_on_edge"] == 1.0

    global_stats = trainer._direction_hierarchy_output_stats(
        labels,
        trade_prob_np=trade_prob,
        side_pred_np=side_pred,
        side_long_prob_np=side_long_prob,
    )
    assert global_stats["hier_trade_pred_rate"] == 1.0
    assert global_stats["hier_flat_label_rate"] == 1 / 3







def test_entry_v10_hier_trade_pos_weight_can_downweight_majority_trade_labels() -> None:
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    raw_majority_trade = (1.0 - 0.658120) / 0.658120

    assert trainer._bounded_pos_weight(raw_majority_trade, 12.0) == 1.0
    assert trainer._bounded_pos_weight(raw_majority_trade, 12.0, allow_below_one=True) == raw_majority_trade
    assert trainer._bounded_pos_weight(0.0, 12.0, allow_below_one=True) == 1.0 / 12.0


def test_entry_v10_direction_slice_hard_red_stop_waits_for_no_progress(monkeypatch) -> None:
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE", 3)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_MIN_EPOCHS", 6)
    red_stats = {
        "direction_slice_contract_ok": False,
        "direction_slice_failure_count": 4,
    }
    green_stats = {
        "direction_slice_contract_ok": True,
        "direction_slice_failure_count": 0,
    }

    assert not trainer._direction_slice_hard_red_stop_ready(
        epoch=5,
        epochs_since_improve=3,
        best_slice_contract_ok=False,
        val_stats=red_stats,
    )
    assert not trainer._direction_slice_hard_red_stop_ready(
        epoch=6,
        epochs_since_improve=2,
        best_slice_contract_ok=False,
        val_stats=red_stats,
    )
    assert not trainer._direction_slice_hard_red_stop_ready(
        epoch=6,
        epochs_since_improve=3,
        best_slice_contract_ok=False,
        val_stats=green_stats,
    )
    assert trainer._direction_slice_hard_red_stop_ready(
        epoch=6,
        epochs_since_improve=3,
        best_slice_contract_ok=False,
        val_stats=red_stats,
    )
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_HARD_RED_STOP_PATIENCE", 0)
    assert not trainer._direction_slice_hard_red_stop_ready(
        epoch=6,
        epochs_since_improve=3,
        best_slice_contract_ok=False,
        val_stats=red_stats,
    )


def test_entry_v10_direction_slice_failure_evidence_writes_outside_bundle(tmp_path: Path) -> None:
    import numpy as np

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    intended_bundle = tmp_path / "runs" / "slice_failed_bundle"
    stats = {
        "direction_slice_failure_count": np.int64(2),
        "direction_slice_contract_ok": False,
        "direction_slice_failure_details": [
            {
                "ctx_cat_index": np.int64(0),
                "ctx_cat_value": np.int64(4),
                "rows": np.int64(64),
                "accuracy": np.float64(0.25),
                "hier_trade_pred_rate": np.float64(1.0),
            }
        ],
        "hier_trade_pred_rate": np.float64(1.0),
        "hier_trade_prob_label_flat_mean": np.float64(0.66),
        "hier_side_acc_on_edge": np.float64(0.57),
        "unrelated": "not persisted",
    }
    snapshot = trainer._direction_slice_stats_snapshot(stats)
    evidence_path = trainer._write_direction_slice_failure_evidence(
        intended_bundle,
        {
            "schema_version": "entry_direction_slice_failure_evidence_v1",
            "decision": "FAIL_DIRECTION_SLICE_GUARD",
            "best_direction_slice_stats": snapshot,
        },
    )

    assert evidence_path == tmp_path / "runs" / "slice_failed_bundle__direction_slice_failure_evidence.json"
    assert evidence_path.is_file()
    assert not intended_bundle.exists()
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert payload["decision"] == "FAIL_DIRECTION_SLICE_GUARD"
    assert payload["bundle_written"] is False
    assert payload["promotion_shadow_live_allowed"] is False
    assert payload["best_direction_slice_stats"]["direction_slice_failure_count"] == 2
    detail = payload["best_direction_slice_stats"]["direction_slice_failure_details"][0]
    assert detail["ctx_cat_index"] == 0
    assert detail["hier_trade_pred_rate"] == 1.0
    assert payload["best_direction_slice_stats"]["hier_trade_prob_label_flat_mean"] == 0.66
    assert payload["best_direction_slice_stats"]["hier_side_acc_on_edge"] == 0.57
    assert "unrelated" not in payload["best_direction_slice_stats"]


def test_entry_v10_direction_slice_balanced_sampler_builds_hard_slice_batches() -> None:
    import numpy as np

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    labels = np.asarray([0, 1, 2] * 8 + [0, 1, 2] * 8, dtype=np.int64)
    ctx_cat = np.asarray([[0]] * 24 + [[1]] * 24, dtype=np.int64)
    sampler = trainer._DirectionSliceBalancedSampler(
        labels=labels,
        ctx_cat=ctx_cat,
        ctx_cat_indices=[0],
        batch_size=12,
        min_rows=6,
        min_label_rate=0.10,
        seed=20260715,
    )

    first_epoch = list(iter(sampler))
    first_batch = first_epoch[:12]
    batch_slices = set(ctx_cat[first_batch, 0].tolist())
    second_epoch = list(iter(sampler))

    assert len(sampler) == 48
    assert sampler.audited_slice_count == 2
    assert batch_slices == {0, 1}
    assert len(first_epoch) == len(set(first_epoch)) == len(labels)
    assert sorted(first_epoch) == list(range(len(labels)))
    assert len(second_epoch) == len(set(second_epoch)) == len(labels)
    assert sorted(second_epoch) == list(range(len(labels)))
    assert first_epoch != second_epoch
    assert np.bincount(labels[first_epoch], minlength=3).tolist() == [16, 16, 16]


def test_entry_v10_direction_slice_balanced_sampler_does_not_pad_or_replace() -> None:
    import numpy as np

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    labels = np.asarray([0, 1, 2] * 17, dtype=np.int64)
    ctx_cat = np.asarray([[0]] * 24 + [[1]] * 27, dtype=np.int64)
    sampler = trainer._DirectionSliceBalancedSampler(
        labels=labels,
        ctx_cat=ctx_cat,
        ctx_cat_indices=[0],
        batch_size=12,
        min_rows=6,
        min_label_rate=0.10,
        seed=20260723,
    )

    epoch = list(iter(sampler))

    assert len(sampler) == 51
    assert len(epoch) == len(set(epoch)) == 51
    assert sorted(epoch) == list(range(51))
    assert np.bincount(labels[epoch], minlength=3).tolist() == [17, 17, 17]


def test_entry_v10_direction_slice_balanced_sampler_fails_without_active_slice() -> None:
    import numpy as np
    import pytest

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    labels = np.asarray([0, 1, 2], dtype=np.int64)
    ctx_cat = np.asarray([[0], [0], [0]], dtype=np.int64)

    with pytest.raises(RuntimeError, match="ENTRY_DIRECTION_SLICE_BALANCED_SAMPLER_NO_ACTIVE_SLICES"):
        trainer._DirectionSliceBalancedSampler(
            labels=labels,
            ctx_cat=ctx_cat,
            ctx_cat_indices=[0],
            batch_size=8,
            min_rows=8,
            min_label_rate=0.10,
            seed=1,
        )


def test_entry_v10_direction_slice_recall_term_penalizes_low_true_class_prob(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_RECALL_LOSS_WEIGHT", 5.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_RECALL_PROB_FLOOR", 0.30)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_RECALL_MIN_LABEL_RATE", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_RECALL_MIN_ROWS", 3)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES", "0")

    targets = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    ctx_cat = torch.tensor([[2], [2], [2], [2], [2], [2]], dtype=torch.long)
    low_recall = torch.tensor(
        [
            [0.10, 0.45, 0.45],
            [0.10, 0.45, 0.45],
            [0.45, 0.10, 0.45],
            [0.45, 0.10, 0.45],
            [0.45, 0.45, 0.10],
            [0.45, 0.45, 0.10],
        ],
        dtype=torch.float32,
    )
    covered = torch.tensor(
        [
            [0.34, 0.33, 0.33],
            [0.34, 0.33, 0.33],
            [0.33, 0.34, 0.33],
            [0.33, 0.34, 0.33],
            [0.33, 0.33, 0.34],
            [0.33, 0.33, 0.34],
        ],
        dtype=torch.float32,
    )

    assert float(trainer._direction_slice_recall_prob_term(low_recall, targets, ctx_cat).item()) > 0.0
    assert float(trainer._direction_slice_recall_prob_term(covered, targets, ctx_cat).item()) == 0.0


def test_entry_v10_direction_slice_balanced_ce_term_penalizes_low_true_class_logits(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT", 2.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_LABEL_RATE", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_BALANCED_CE_MIN_ROWS", 3)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES", "0")
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION", "mean")

    targets = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    ctx_cat = torch.tensor([[2], [2], [2], [2], [2], [2]], dtype=torch.long)
    bad_logits = torch.tensor(
        [
            [-2.0, 3.0, 3.0],
            [-2.0, 3.0, 3.0],
            [3.0, -2.0, 3.0],
            [3.0, -2.0, 3.0],
            [3.0, 3.0, -2.0],
            [3.0, 3.0, -2.0],
        ],
        dtype=torch.float32,
    )
    good_logits = torch.tensor(
        [
            [5.0, -1.0, -1.0],
            [5.0, -1.0, -1.0],
            [-1.0, 5.0, -1.0],
            [-1.0, 5.0, -1.0],
            [-1.0, -1.0, 5.0],
            [-1.0, -1.0, 5.0],
        ],
        dtype=torch.float32,
    )

    bad_loss = float(trainer._direction_slice_balanced_ce_term(bad_logits, targets, ctx_cat).item())
    good_loss = float(trainer._direction_slice_balanced_ce_term(good_logits, targets, ctx_cat).item())

    assert bad_loss > good_loss
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_BALANCED_CE_WEIGHT", 0.0)
    assert float(trainer._direction_slice_balanced_ce_term(bad_logits, targets, ctx_cat).item()) == 0.0


def test_entry_v10_direction_slice_true_margin_term_penalizes_wrong_argmax(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT", 2.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_TRUE_MARGIN", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_LABEL_RATE", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_MIN_ROWS", 3)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES", "0")
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION", "mean")

    targets = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    ctx_cat = torch.tensor([[2], [2], [2], [2], [2], [2]], dtype=torch.long)
    bad_logits = torch.tensor(
        [
            [-2.0, 3.0, 3.0],
            [-2.0, 3.0, 3.0],
            [3.0, -2.0, 3.0],
            [3.0, -2.0, 3.0],
            [3.0, 3.0, -2.0],
            [3.0, 3.0, -2.0],
        ],
        dtype=torch.float32,
    )
    good_logits = torch.tensor(
        [
            [5.0, -1.0, -1.0],
            [5.0, -1.0, -1.0],
            [-1.0, 5.0, -1.0],
            [-1.0, 5.0, -1.0],
            [-1.0, -1.0, 5.0],
            [-1.0, -1.0, 5.0],
        ],
        dtype=torch.float32,
    )

    bad_loss = float(trainer._direction_slice_true_margin_term(bad_logits, targets, ctx_cat).item())
    good_loss = float(trainer._direction_slice_true_margin_term(good_logits, targets, ctx_cat).item())

    assert bad_loss > good_loss
    assert good_loss == 0.0
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_TRUE_MARGIN_WEIGHT", 0.0)
    assert float(trainer._direction_slice_true_margin_term(bad_logits, targets, ctx_cat).item()) == 0.0


def test_entry_v10_hier_slice_side_terms_penalize_collapsed_side_head(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_SIDE_CE_WEIGHT", 4.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT", 3.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT", 4.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_MARGIN", 0.02)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_SIDE_MIN_LABEL_RATE", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_SIDE_MIN_ROWS", 3)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT", 4.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_TOLERANCE", 0.02)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT", 4.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_TOLERANCE", 0.02)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_LABEL_RATE", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_MIN_ROWS", 3)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES", "0")
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION", "mean")
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE", 1.0)

    side_targets = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    side_mask = torch.ones(6, dtype=torch.bool)
    ctx_cat = torch.tensor([[2], [2], [2], [2], [2], [2]], dtype=torch.long)
    collapsed_logits = torch.tensor(
        [
            [-2.0, 4.0],
            [-2.0, 4.0],
            [-2.0, 4.0],
            [-2.0, 4.0],
            [-2.0, 4.0],
            [-2.0, 4.0],
        ],
        dtype=torch.float32,
    )
    good_logits = torch.tensor(
        [
            [4.0, -2.0],
            [4.0, -2.0],
            [4.0, -2.0],
            [-2.0, 4.0],
            [-2.0, 4.0],
            [-2.0, 4.0],
        ],
        dtype=torch.float32,
    )

    bad_ce = float(
        trainer._hier_slice_side_balanced_ce_term(collapsed_logits, side_targets, side_mask, ctx_cat).item()
    )
    good_ce = float(trainer._hier_slice_side_balanced_ce_term(good_logits, side_targets, side_mask, ctx_cat).item())
    bad_margin = float(
        trainer._hier_slice_side_true_margin_term(collapsed_logits, side_targets, side_mask, ctx_cat).item()
    )
    good_margin = float(
        trainer._hier_slice_side_true_margin_term(good_logits, side_targets, side_mask, ctx_cat).item()
    )
    bad_accuracy_edge = float(
        trainer._hier_slice_side_accuracy_edge_term(collapsed_logits, side_targets, side_mask, ctx_cat).item()
    )
    good_accuracy_edge = float(
        trainer._hier_slice_side_accuracy_edge_term(good_logits, side_targets, side_mask, ctx_cat).item()
    )
    bad_global_prior = float(
        trainer._hier_side_global_prior_match_term(collapsed_logits, side_targets, side_mask).item()
    )
    good_global_prior = float(trainer._hier_side_global_prior_match_term(good_logits, side_targets, side_mask).item())
    bad_slice_prior = float(
        trainer._hier_slice_side_prior_match_term(collapsed_logits, side_targets, side_mask, ctx_cat).item()
    )
    good_slice_prior = float(
        trainer._hier_slice_side_prior_match_term(good_logits, side_targets, side_mask, ctx_cat).item()
    )

    assert bad_ce > good_ce
    assert bad_margin > good_margin
    assert good_margin == 0.0
    assert bad_accuracy_edge > good_accuracy_edge
    assert good_accuracy_edge == 0.0
    assert bad_global_prior > good_global_prior
    assert bad_slice_prior > good_slice_prior
    assert good_global_prior == 0.0
    assert good_slice_prior == 0.0
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_SIDE_CE_WEIGHT", 0.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_SIDE_TRUE_MARGIN_WEIGHT", 0.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_SIDE_ACCURACY_EDGE_WEIGHT", 0.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SIDE_GLOBAL_PRIOR_MATCH_WEIGHT", 0.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_SIDE_PRIOR_MATCH_WEIGHT", 0.0)
    assert float(trainer._hier_slice_side_balanced_ce_term(collapsed_logits, side_targets, side_mask, ctx_cat).item()) == 0.0
    assert float(trainer._hier_slice_side_true_margin_term(collapsed_logits, side_targets, side_mask, ctx_cat).item()) == 0.0
    assert float(trainer._hier_slice_side_accuracy_edge_term(collapsed_logits, side_targets, side_mask, ctx_cat).item()) == 0.0
    assert float(trainer._hier_side_global_prior_match_term(collapsed_logits, side_targets, side_mask).item()) == 0.0
    assert float(trainer._hier_slice_side_prior_match_term(collapsed_logits, side_targets, side_mask, ctx_cat).item()) == 0.0


def test_entry_v10_hier_trade_prior_terms_penalize_all_trade_head(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT", 4.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_TOLERANCE", 0.02)
    monkeypatch.setattr(trainer, "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT", 4.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_TOLERANCE", 0.02)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_LABEL_RATE", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_MIN_ROWS", 3)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT", 4.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_MARGIN", 0.02)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES", "0")
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION", "mean")
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE", 0.05)

    y_trade = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32)
    ctx_cat = torch.tensor([[2], [2], [2], [2], [2], [2]], dtype=torch.long)
    collapsed_trade = torch.full((6, 1), 4.0, dtype=torch.float32)
    matched_trade = torch.tensor([[4.0], [4.0], [4.0], [-4.0], [-4.0], [-4.0]], dtype=torch.float32)

    bad_global_prior = float(trainer._hier_trade_global_prior_match_term(collapsed_trade, y_trade).item())
    good_global_prior = float(trainer._hier_trade_global_prior_match_term(matched_trade, y_trade).item())
    bad_slice_prior = float(trainer._hier_slice_trade_prior_match_term(collapsed_trade, y_trade, ctx_cat).item())
    good_slice_prior = float(trainer._hier_slice_trade_prior_match_term(matched_trade, y_trade, ctx_cat).item())
    bad_accuracy_edge = float(
        trainer._hier_slice_trade_accuracy_edge_term(collapsed_trade, y_trade, ctx_cat).item()
    )
    good_accuracy_edge = float(
        trainer._hier_slice_trade_accuracy_edge_term(matched_trade, y_trade, ctx_cat).item()
    )

    assert bad_global_prior > good_global_prior
    assert bad_slice_prior > good_slice_prior
    assert bad_accuracy_edge > good_accuracy_edge
    assert good_global_prior == 0.0
    assert good_slice_prior == 0.0
    assert good_accuracy_edge == 0.0
    monkeypatch.setattr(trainer, "ENTRY_HIER_TRADE_GLOBAL_PRIOR_MATCH_WEIGHT", 0.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_TRADE_PRIOR_MATCH_WEIGHT", 0.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_TRADE_ACCURACY_EDGE_WEIGHT", 0.0)
    assert float(trainer._hier_trade_global_prior_match_term(collapsed_trade, y_trade).item()) == 0.0
    assert float(trainer._hier_slice_trade_prior_match_term(collapsed_trade, y_trade, ctx_cat).item()) == 0.0
    assert float(trainer._hier_slice_trade_accuracy_edge_term(collapsed_trade, y_trade, ctx_cat).item()) == 0.0


def test_entry_v10_hier_flat_logit_margin_terms_penalize_flat_as_trade(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT", 8.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_FLAT_LOGIT_MARGIN", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_HIER_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT", 8.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_LABEL_RATE", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_MIN_ROWS", 3)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES", "0")
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION", "mean")

    y_trade = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float32)
    ctx_cat = torch.tensor([[2], [2], [2], [2], [2], [2]], dtype=torch.long)
    bad_trade = torch.full((6, 1), 1.0, dtype=torch.float32)
    good_trade = torch.tensor([[1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0]], dtype=torch.float32)

    bad_global = float(trainer._hier_flat_logit_margin_term(bad_trade, y_trade).item())
    good_global = float(trainer._hier_flat_logit_margin_term(good_trade, y_trade).item())
    bad_slice = float(trainer._hier_slice_flat_logit_margin_term(bad_trade, y_trade, ctx_cat).item())
    good_slice = float(trainer._hier_slice_flat_logit_margin_term(good_trade, y_trade, ctx_cat).item())

    assert bad_global > good_global
    assert bad_slice > good_slice
    assert good_global == 0.0
    assert good_slice == 0.0
    monkeypatch.setattr(trainer, "ENTRY_HIER_FLAT_LOGIT_MARGIN_WEIGHT", 0.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SLICE_FLAT_LOGIT_MARGIN_WEIGHT", 0.0)
    assert float(trainer._hier_flat_logit_margin_term(bad_trade, y_trade).item()) == 0.0
    assert float(trainer._hier_slice_flat_logit_margin_term(bad_trade, y_trade, ctx_cat).item()) == 0.0










def test_entry_v10_direction_slice_accuracy_edge_term_penalizes_below_majority(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT", 4.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MARGIN", 0.02)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS", 6)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES", "0")
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION", "mean")
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE", 0.05)

    targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)
    ctx_cat = torch.tensor([[2]] * len(targets), dtype=torch.long)
    below_majority = torch.tensor(
        [
            [-1.0, 2.0, 0.0],
            [-1.0, 2.0, 0.0],
            [-1.0, 2.0, 0.0],
            [-1.0, 2.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0, 0.0, -1.0],
            [2.0, 0.0, -1.0],
            [2.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    above_majority = torch.tensor(
        [
            [3.0, -1.0, -1.0],
            [3.0, -1.0, -1.0],
            [3.0, -1.0, -1.0],
            [3.0, -1.0, -1.0],
            [-1.0, 3.0, -1.0],
            [-1.0, 3.0, -1.0],
            [-1.0, 3.0, -1.0],
            [-1.0, -1.0, 3.0],
            [-1.0, -1.0, 3.0],
            [-1.0, -1.0, 3.0],
        ],
        dtype=torch.float32,
    )

    bad_loss = float(trainer._direction_slice_accuracy_edge_term(below_majority, targets, ctx_cat).item())
    good_loss = float(trainer._direction_slice_accuracy_edge_term(above_majority, targets, ctx_cat).item())

    assert bad_loss > 0.0
    assert good_loss == 0.0
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_WEIGHT", 0.0)
    assert float(trainer._direction_slice_accuracy_edge_term(below_majority, targets, ctx_cat).item()) == 0.0


def test_entry_v10_direction_slice_confusion_pair_term_penalizes_wrong_class(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT", 4.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_MARGIN", 0.02)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_LABEL_RATE", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_ACCURACY_EDGE_MIN_ROWS", 6)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES", "0")
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION", "mean")
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE", 1.0)

    targets = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    ctx_cat = torch.tensor([[3]] * len(targets), dtype=torch.long)
    confused_logits = torch.tensor(
        [
            [-1.0, 4.0, -1.0],
            [-1.0, 4.0, -1.0],
            [4.0, -1.0, -1.0],
            [4.0, -1.0, -1.0],
            [4.0, -1.0, -1.0],
            [4.0, -1.0, -1.0],
        ],
        dtype=torch.float32,
    )
    correct_logits = torch.tensor(
        [
            [4.0, -1.0, -1.0],
            [4.0, -1.0, -1.0],
            [-1.0, 4.0, -1.0],
            [-1.0, 4.0, -1.0],
            [-1.0, -1.0, 4.0],
            [-1.0, -1.0, 4.0],
        ],
        dtype=torch.float32,
    )

    bad_loss = float(trainer._direction_slice_confusion_pair_term(confused_logits, targets, ctx_cat).item())
    good_loss = float(trainer._direction_slice_confusion_pair_term(correct_logits, targets, ctx_cat).item())

    assert bad_loss > 0.0
    assert good_loss == 0.0
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_CONFUSION_PAIR_WEIGHT", 0.0)
    assert float(trainer._direction_slice_confusion_pair_term(confused_logits, targets, ctx_cat).item()) == 0.0


def test_entry_v10_direction_slice_prior_match_penalizes_slice_distribution_drift(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT", 3.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_TOLERANCE", 0.02)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_LABEL_RATE", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_MIN_ROWS", 6)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES", "0")
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION", "mean")
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE", 1.0)

    targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)
    ctx_cat = torch.tensor([[2]] * len(targets), dtype=torch.long)
    collapsed = torch.tensor([[0.10, 0.80, 0.10]] * len(targets), dtype=torch.float32)
    matched = torch.tensor([[0.40, 0.30, 0.30]] * len(targets), dtype=torch.float32)

    bad_loss = float(trainer._direction_slice_prior_match_term(collapsed, targets, ctx_cat).item())
    good_loss = float(trainer._direction_slice_prior_match_term(matched, targets, ctx_cat).item())

    assert bad_loss > 0.0
    assert good_loss == 0.0
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_PRIOR_MATCH_WEIGHT", 0.0)
    assert float(trainer._direction_slice_prior_match_term(collapsed, targets, ctx_cat).item()) == 0.0


def test_entry_v10_direction_global_prior_match_penalizes_global_distribution_drift(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT", 8.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_TOLERANCE", 0.02)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_MIN_LABEL_RATE", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE", 1.0)

    targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)
    collapsed = torch.tensor([[0.10, 0.89, 0.01]] * len(targets), dtype=torch.float32)
    matched = torch.tensor([[0.40, 0.30, 0.30]] * len(targets), dtype=torch.float32)

    bad_loss = float(trainer._direction_global_prior_match_term(collapsed, targets).item())
    good_loss = float(trainer._direction_global_prior_match_term(matched, targets).item())

    assert bad_loss > 0.0
    assert good_loss == 0.0
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_GLOBAL_PRIOR_MATCH_WEIGHT", 0.0)
    assert float(trainer._direction_global_prior_match_term(collapsed, targets).item()) == 0.0


def test_entry_v10_direction_vs_flat_margin_term_penalizes_directional_flat_argmax(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_VS_FLAT_MARGIN_WEIGHT", 4.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_VS_FLAT_MARGIN", 0.10)

    targets = torch.tensor([0, 1, 2], dtype=torch.long)
    collapsed = torch.tensor(
        [
            [0.0, -1.0, 1.0],
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    side_above_flat = torch.tensor(
        [
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, 0.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    flat_wrong_side = torch.tensor(
        [
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, 0.0],
            [1.2, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    wrong_side = torch.tensor(
        [
            [-1.0, 2.0, 0.0],
            [2.0, -1.0, 0.0],
            [-1.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )

    assert float(trainer._direction_vs_flat_margin_term(collapsed, targets).item()) > 4.0
    assert float(trainer._direction_vs_flat_margin_term(flat_wrong_side, targets).item()) > 2.0
    assert float(trainer._direction_vs_flat_margin_term(wrong_side, targets).item()) > 4.0
    assert float(trainer._direction_vs_flat_margin_term(side_above_flat, targets).item()) < 1.0


def test_entry_v10_direction_utility_margin_term_penalizes_wrong_utility_side(monkeypatch) -> None:
    import pytest
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT", 4.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_UTILITY_MIN_GAP_BPS", 15.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_UTILITY_LOGIT_MARGIN", 0.10)

    long_utility = torch.tensor([40.0, 0.0, 10.0], dtype=torch.float32)
    short_utility = torch.tensor([0.0, 40.0, 0.0], dtype=torch.float32)
    wrong_side = torch.tensor(
        [
            [-1.0, 2.0, 0.0],
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    side_or_flat_ok = torch.tensor(
        [
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, 0.0],
            [-1.0, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )

    bad_loss = float(trainer._direction_utility_margin_term(wrong_side, long_utility, short_utility).item())
    good_loss = float(trainer._direction_utility_margin_term(side_or_flat_ok, long_utility, short_utility).item())

    assert bad_loss > good_loss + 5.0
    assert good_loss < 1.0

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT", 0.0)
    assert float(trainer._direction_utility_margin_term(wrong_side, long_utility, short_utility).item()) == 0.0

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT", 4.0)
    with pytest.raises(RuntimeError, match="ENTRY_DIRECTION_UTILITY_MARGIN_SHAPE_MISMATCH"):
        trainer._direction_utility_margin_term(wrong_side, long_utility[:2], short_utility)


def test_entry_v10_direction_side_utility_conviction_penalizes_side_label_flat_or_wrong_side(
    monkeypatch,
) -> None:
    import pytest
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT", 6.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_MIN_GAP_BPS", 15.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_LOGIT_MARGIN", 0.10)

    targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    long_utility = torch.tensor([40.0, 0.0, 40.0, 5.0], dtype=torch.float32)
    short_utility = torch.tensor([0.0, 40.0, 0.0, 0.0], dtype=torch.float32)
    flat_or_wrong = torch.tensor(
        [
            [-1.0, 0.0, 2.0],
            [0.0, -1.0, 2.0],
            [-1.0, 0.0, 2.0],
            [-1.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    side_ok = torch.tensor(
        [
            [2.0, 0.0, -1.0],
            [0.0, 2.0, -1.0],
            [-1.0, 0.0, 2.0],
            [-1.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )

    bad_loss = float(
        trainer._direction_side_utility_conviction_term(
            flat_or_wrong,
            targets,
            long_utility,
            short_utility,
        ).item()
    )
    good_loss = float(
        trainer._direction_side_utility_conviction_term(
            side_ok,
            targets,
            long_utility,
            short_utility,
        ).item()
    )

    assert bad_loss > good_loss + 10.0
    assert good_loss < 1.0

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT", 0.0)
    assert (
        float(
            trainer._direction_side_utility_conviction_term(
                flat_or_wrong,
                targets,
                long_utility,
                short_utility,
            ).item()
        )
        == 0.0
    )

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT", 6.0)
    with pytest.raises(RuntimeError, match="ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_SHAPE_MISMATCH"):
        trainer._direction_side_utility_conviction_term(
            flat_or_wrong,
            targets[:2],
            long_utility,
            short_utility,
        )


def test_entry_v10_direction_utility_trade_conviction_requires_tradable_side_edge(monkeypatch) -> None:
    import pytest
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT", 8.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_GAP_BPS", 15.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MIN_UTILITY_BPS", 0.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_MAX_BAD_PATH", 0.50)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_LOGIT_MARGIN", 0.10)

    long_utility = torch.tensor([40.0, -5.0, 0.0, 30.0], dtype=torch.float32)
    short_utility = torch.tensor([0.0, -40.0, 40.0, 0.0], dtype=torch.float32)
    long_bad = torch.tensor([0.20, 0.20, 0.90, 0.80], dtype=torch.float32)
    short_bad = torch.tensor([0.90, 0.90, 0.20, 0.20], dtype=torch.float32)
    flat_or_wrong = torch.tensor(
        [
            [-1.0, 0.0, 2.0],
            [-1.0, 0.0, 2.0],
            [0.0, -1.0, 2.0],
            [-1.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    side_ok = torch.tensor(
        [
            [4.0, 0.0, -2.0],
            [-1.0, 0.0, 2.0],
            [0.0, 4.0, -2.0],
            [-1.0, 0.0, 2.0],
        ],
        dtype=torch.float32,
    )

    bad_loss = float(
        trainer._direction_utility_trade_conviction_term(
            flat_or_wrong,
            long_utility,
            short_utility,
            long_bad,
            short_bad,
        ).item()
    )
    good_loss = float(
        trainer._direction_utility_trade_conviction_term(
            side_ok,
            long_utility,
            short_utility,
            long_bad,
            short_bad,
        ).item()
    )

    assert bad_loss > good_loss + 10.0
    assert good_loss < 1.0

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT", 0.0)
    assert (
        float(
            trainer._direction_utility_trade_conviction_term(
                flat_or_wrong,
                long_utility,
                short_utility,
                long_bad,
                short_bad,
            ).item()
        )
        == 0.0
    )

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT", 8.0)
    with pytest.raises(RuntimeError, match="ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_SHAPE_MISMATCH"):
        trainer._direction_utility_trade_conviction_term(
            flat_or_wrong,
            long_utility[:2],
            short_utility,
            long_bad,
            short_bad,
        )


def test_entry_v10_direction_utility_triad_ce_teaches_no_edge_flat(monkeypatch) -> None:
    import pytest
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT", 8.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_GAP_BPS", 15.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MIN_UTILITY_BPS", 0.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_UTILITY_TRIAD_CE_MAX_BAD_PATH", 0.50)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_UTILITY_TRIAD_CE_CLASS_WEIGHT_CAP", 4.0)

    long_utility = torch.tensor([40.0, 0.0, 0.0], dtype=torch.float32)
    short_utility = torch.tensor([0.0, 40.0, 0.0], dtype=torch.float32)
    long_bad = torch.tensor([0.20, 0.90, 0.90], dtype=torch.float32)
    short_bad = torch.tensor([0.90, 0.20, 0.90], dtype=torch.float32)
    wrong_logits = torch.tensor(
        [
            [-1.0, 0.0, 3.0],
            [0.0, -1.0, 3.0],
            [3.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    correct_logits = torch.tensor(
        [
            [4.0, -1.0, -2.0],
            [-1.0, 4.0, -2.0],
            [-1.0, -2.0, 4.0],
        ],
        dtype=torch.float32,
    )

    bad_loss = float(
        trainer._direction_utility_triad_ce_term(
            wrong_logits,
            long_utility,
            short_utility,
            long_bad,
            short_bad,
        ).item()
    )
    good_loss = float(
        trainer._direction_utility_triad_ce_term(
            correct_logits,
            long_utility,
            short_utility,
            long_bad,
            short_bad,
        ).item()
    )

    assert bad_loss > good_loss + 20.0
    assert good_loss < 0.25

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT", 0.0)
    assert (
        float(
            trainer._direction_utility_triad_ce_term(
                wrong_logits,
                long_utility,
                short_utility,
                long_bad,
                short_bad,
            ).item()
        )
        == 0.0
    )

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT", 8.0)
    with pytest.raises(RuntimeError, match="ENTRY_DIRECTION_UTILITY_TRIAD_CE_SHAPE_MISMATCH"):
        trainer._direction_utility_triad_ce_term(
            wrong_logits,
            long_utility[:2],
            short_utility,
            long_bad,
            short_bad,
        )


def test_entry_v10_direction_flat_starvation_term_penalizes_zero_flat_predictions(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT", 8.0)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_FLAT_STARVATION_MIN_LABEL_RATE", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_FLAT_STARVATION_MIN_ROWS", 2)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FRACTION", 0.50)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_FLAT_STARVATION_PRED_FLOOR", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_FLAT_STARVATION_LOGIT_MARGIN", 0.10)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_MIN_PRED_RATE_SOFTMAX_TEMPERATURE", 0.05)
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_CTX_CAT_INDICES", "0")
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_SLICE_LOSS_AGGREGATION", "mean")

    targets = torch.tensor([2, 2, 2, 0, 1, 0], dtype=torch.long)
    ctx_cat = torch.tensor([[1], [1], [2], [1], [2], [2]], dtype=torch.long)
    no_flat = torch.tensor(
        [
            [2.0, 1.0, -1.0],
            [1.0, 2.0, -1.0],
            [2.0, 1.0, -1.0],
            [2.0, 1.0, -1.0],
            [1.0, 2.0, -1.0],
            [2.0, 1.0, -1.0],
        ],
        dtype=torch.float32,
    )
    flat_ok = torch.tensor(
        [
            [0.0, -1.0, 3.0],
            [-1.0, 0.0, 3.0],
            [0.0, -1.0, 3.0],
            [2.0, 0.0, -1.0],
            [0.0, 2.0, -1.0],
            [2.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
    )

    bad_loss = float(trainer._direction_flat_starvation_term(no_flat, targets, ctx_cat).item())
    good_loss = float(trainer._direction_flat_starvation_term(flat_ok, targets, ctx_cat).item())

    assert bad_loss > good_loss + 5.0
    assert good_loss < 1.0

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_FLAT_STARVATION_WEIGHT", 0.0)
    assert float(trainer._direction_flat_starvation_term(no_flat, targets, ctx_cat).item()) == 0.0


def test_entry_v10_validate_initializes_direction_utility_margin_accumulator() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")
    validate_start = text.index("def validate(")
    validate_init = text[validate_start:text.index("    with torch.no_grad():", validate_start)]

    assert "total_direction_utility_margin = 0.0" in validate_init
    assert "total_direction_utility_margin += " in text[text.index("    with torch.no_grad():", validate_start):]
    assert "total_direction_side_utility_conviction = 0.0" in validate_init
    assert "total_direction_side_utility_conviction += " in text[
        text.index("    with torch.no_grad():", validate_start):
    ]
    assert "total_direction_utility_trade_conviction = 0.0" in validate_init
    assert "total_direction_utility_trade_conviction += " in text[
        text.index("    with torch.no_grad():", validate_start):
    ]
    assert "total_direction_utility_triad_ce = 0.0" in validate_init
    assert "total_direction_utility_triad_ce += " in text[
        text.index("    with torch.no_grad():", validate_start):
    ]
    assert "total_direction_flat_starvation = 0.0" in validate_init
    assert "total_direction_flat_starvation += " in text[text.index("    with torch.no_grad():", validate_start):]


def test_entry_v10_direction_failure_evidence_records_model_native_loss_recipe() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")
    required_fields = (
        "direction_utility_margin_weight",
        "direction_side_utility_conviction_weight",
        "direction_utility_trade_conviction_weight",
        "direction_utility_triad_ce_weight",
        "hier_trade_global_prior_match_weight",
        "hier_slice_trade_prior_match_weight",
        "hier_flat_logit_margin_weight",
        "hier_slice_side_ce_weight",
        "hier_side_global_prior_match_weight",
        "direction_flat_starvation_weight",
    )

    for marker in (
        '"decision": "FAIL_DIRECTION_CLASS_BALANCE_GUARD"',
        '"decision": "FAIL_DIRECTION_SLICE_GUARD"',
    ):
        start = text.index(marker)
        block = text[start:text.index("raise RuntimeError", start)]
        for field in required_fields:
            assert f'"{field}"' in block

    retired_markers = (
        "direction_hierarchical_composition",
        "hier_compose_",
        "hier_public_",
        "hier_ctx_prior_adapter",
        "hier_ctx_direction_calibration",
        "public_trade_logit",
        "public_flat_logit",
        "public_side_logits",
        "margin_bridge",
        "residual_scale",
    )
    for marker in retired_markers:
        assert marker not in text


def test_entry_v10_direction_repair_fails_closed_without_calibration_fallback() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")

    assert "ENTRY_DIRECTION_CAL_FALLBACK" not in text
    assert "fallback_candidates" not in text
    assert "direction_calibration_guard_fallback_used" not in text
    assert "run_train_xau_direction_repair_guard_fallback" not in text
    assert "refusing to write a collapsed direction bundle" in text


def test_entry_v10_direction_aux_loss_uses_sample_weight_and_balance() -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    logits = torch.tensor(
        [
            [5.0, -2.0, -2.0],
            [5.0, -2.0, -2.0],
            [5.0, -2.0, -2.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 2], dtype=torch.long)
    sample_weight = torch.tensor([1.0, 1.0, 5.0], dtype=torch.float32)
    criterion = trainer.CostSensitiveCrossEntropyLoss(
        class_weights=None,
        cost_matrix=torch.zeros((3, 3), dtype=torch.float32),
        cost_scale=0.0,
        enabled=False,
        balance_alpha=0.20,
        balance_target="label",
        balance_class_weights=torch.tensor([1.0, 1.0, 4.0], dtype=torch.float32),
    )

    loss = trainer._direction_aux_ce_loss(logits, targets, criterion, sample_weight, None)
    weighted_ce = (criterion.ce(logits, targets) * sample_weight).mean()
    balance_term = trainer._direction_balance_term(torch.softmax(logits, dim=1), targets, criterion)

    assert torch.allclose(loss, weighted_ce + balance_term)
    assert float(loss.item()) > float(criterion.ce(logits, targets).mean().item())

    # With a logit-adjust offset the aux CE consumes adjusted logits while the
    # balance term stays on the raw probabilities.
    offset = torch.log(torch.tensor([0.2, 0.2, 0.6], dtype=torch.float32))
    loss_adj = trainer._direction_aux_ce_loss(
        logits, targets, criterion, sample_weight, offset
    )
    weighted_ce_adj = (criterion.ce(logits + offset, targets) * sample_weight).mean()
    assert torch.allclose(loss_adj, weighted_ce_adj + balance_term)


def test_entry_v10_train_and_validate_apply_mtf_aux_direction_repair() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")
    assert text.count('output_name="mtf_dir_logits"') == 2
    assert 'if "mtf_dir_logits" in out' not in text


def test_entry_v10_multi_tf_window_uses_entry_route_and_m5_clock() -> None:
    import numpy as np
    import pandas as pd

    from gx1.features.htf_features import (
        HTF_V4_MATRIX_CONTRACT,
        MULTI_TF_FEATURE_COUNT_V4,
        MULTI_TF_PER_BAR_FEATURES_V4,
        MULTI_TF_SHIFT,
        multi_tf_last_closed_label,
    )
    from gx1.contracts.entry_exit_feature_base_v1 import (
        ENTRY_DECISION_BAR_SECONDS,
        ENTRY_MTF_CONTEXT_TIMEFRAMES,
    )
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    target = pd.Timestamp("2026-07-08T18:00:00Z")
    cache = {}
    for value, timeframe in enumerate(("M5", "M15", "H1", "H4", "D1"), 1):
        closed_label = multi_tf_last_closed_label(
            target,
            timeframe,
            base_bar_duration=pd.Timedelta(minutes=5),
        )
        ts = np.array([closed_label.value], dtype=np.int64)
        feats = np.full(
            (1, MULTI_TF_FEATURE_COUNT_V4),
            float(value),
            dtype=np.float32,
        )
        frame = pd.DataFrame(
            feats,
            index=pd.DatetimeIndex(ts.astype("datetime64[ns]"), tz="UTC"),
            columns=list(MULTI_TF_PER_BAR_FEATURES_V4),
        )
        frame.attrs["ts_int64"] = ts
        frame.attrs["feats_np"] = feats
        frame.attrs["htf_feature_contract"] = HTF_V4_MATRIX_CONTRACT
        frame.attrs["causal_warmup_rows"] = 0
        cache[timeframe] = frame

    dataset = trainer.EntryV10CtxDataset.__new__(trainer.EntryV10CtxDataset)
    dataset._multi_tf_feats = cache
    dataset.per_tf_seq_lens = {tf: 1 for tf in cache}

    out = dataset._get_multi_tf_window(
        target,
        route_timeframes=ENTRY_MTF_CONTEXT_TIMEFRAMES,
        base_bar_seconds=ENTRY_DECISION_BAR_SECONDS,
        key_prefix="seq_",
    )
    assert tuple(out) == ("seq_m15", "seq_h1", "seq_h4", "seq_d1")
    for value, timeframe in enumerate(ENTRY_MTF_CONTEXT_TIMEFRAMES, 2):
        np.testing.assert_allclose(
            out[f"seq_{timeframe.lower()}"],
            np.full(
                (1, MULTI_TF_FEATURE_COUNT_V4),
                float(value),
                dtype=np.float32,
            ),
        )
    assert "seq_m5" not in out
    assert tuple(MULTI_TF_SHIFT) == ("M5", "M15", "H1", "H4", "D1")


def test_entry_v10_metadata_records_both_route_clocks() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")

    assert '"entry_target_availability_shift_minutes"' in text
    assert '"exit_target_availability_shift_minutes"' in text
    assert '"entry_route_timeframes"' in text
    assert '"exit_route_timeframes"' in text


def test_entry_v10_trainer_verifies_mtf_cache_source_sha() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")

    assert "MULTI_TF_CACHE_SOURCE_BINDING_MISMATCH" in text
    assert "m5_prebuilt_source_sha256" in text
    assert "cache_identity_sha256" in text
    assert "load_multi_tf_v4_cache" in text
    assert "MULTI_TF_DISK_CACHE_MANDATORY" in text


def test_entry_v10_side_validity_head_cannot_be_enabled_untrained() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")

    assert "ENTRY_SIDE_VALIDITY_HEAD_UNTRAINED" in text
    assert "exact architecture requires" in text
    assert "ENTRY_HIER_SIDE_VALIDITY_WEIGHT>0" in text


def test_entry_v10_xau_direction_repair_requires_xau_sources() -> None:
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    failures = trainer._xau_direction_repair_source_failures(
        {
            "train_parquet": "/data/untrusted_asset/v10_dataset_20260710_train.parquet",
            "val_parquet": "/data/generic/v10_dataset_val.parquet",
            "m5_prebuilt_path": "",
        }
    )

    text = "\n".join(failures)
    assert "stale pre-repair dataset marker" in text
    assert "m5_prebuilt_path missing" in text


def test_entry_v10_xau_direction_repair_requires_xau_manifest_provenance(tmp_path: Path) -> None:
    import pandas as pd

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    parquet = tmp_path / "xau_train.parquet"
    pd.DataFrame({"time": [pd.Timestamp("2026-07-08T12:00:00Z")]}).to_parquet(parquet, index=False)
    parquet.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "tape_root": "/data/untrusted_asset/m5_bid_ask",
            }
        ),
        encoding="utf-8",
    )

    failures = trainer._xau_direction_repair_manifest_failures({"train": parquet})

    assert any("immutable XAU_USD tape provenance invalid" in item for item in failures)


def test_entry_v10_outcome_target_contract_does_not_rewrite_structural_rows() -> None:
    import pandas as pd

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    frame = pd.DataFrame(
        {
            "y_direction": [1],
            "y_bad_path": [0.0],
            "y_trade": [1.0],
            "y_tradable": [1.0],
            "y_side": [1],
            "y_side_mask": [1.0],
            "mae_first_n_bps": [1.0],
            "mfe_first_n_bps": [0.0],
            "path_quality_bps": [-1.0],
            "y_position_size_target": [0.7],
            "mfe_long_first_n_bps": [8.0],
            "mae_long_first_n_bps": [2.0],
            "mfe_short_first_n_bps": [0.0],
            "mae_short_first_n_bps": [1.0],
            "bad_path_long_first_n": [0.0],
            "bad_path_short_first_n": [0.0],
            "y_long_final_pnl_at_direction_horizon_bps": [-5.0],
            "y_short_final_pnl_at_direction_horizon_bps": [20.0],
            "y_direction_target_mode_id": [1],
            "y_direction_long_score_bps": [-3.0],
            "y_direction_short_score_bps": [18.6],
            "y_long_path_utility_bps": [-3.0],
            "y_short_path_utility_bps": [18.6],
            "y_long_bad_path": [0.0],
            "y_short_bad_path": [0.0],
            "y_long_expected_mae_bps": [2.0],
            "y_short_expected_mae_bps": [1.0],
            "y_rising_channel_support_touch": [1.0],
            "y_falling_channel_resistance_touch": [0.0],
            "y_support_retest_continuation": [0.0],
            "y_resistance_retest_continuation": [0.0],
            "y_countertrend_short_trap": [0.0],
            "y_countertrend_long_trap": [0.0],
            "y_long_high_mae_low_mfe_early_failure": [0.0],
            "y_short_high_mae_low_mfe_early_failure": [0.0],
        }
    )

    failures = trainer._xau_direction_repair_target_failures("train", frame)

    assert failures == []


def test_entry_v10_outcome_target_contract_rejects_forced_utility_order() -> None:
    import pandas as pd

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    frame = pd.DataFrame(
        {
            "y_direction": [1],
            "y_bad_path": [0.0],
            "y_trade": [1.0],
            "y_tradable": [1.0],
            "y_side": [1],
            "y_side_mask": [1.0],
            "mae_first_n_bps": [1.0],
            "mfe_first_n_bps": [4.0],
            "path_quality_bps": [3.0],
            "y_position_size_target": [0.7],
            "mfe_long_first_n_bps": [2.0],
            "mae_long_first_n_bps": [3.0],
            "mfe_short_first_n_bps": [4.0],
            "mae_short_first_n_bps": [1.0],
            "bad_path_long_first_n": [1.0],
            "bad_path_short_first_n": [0.0],
            "y_long_final_pnl_at_direction_horizon_bps": [-5.0],
            "y_short_final_pnl_at_direction_horizon_bps": [20.0],
            "y_direction_target_mode_id": [1],
            "y_direction_long_score_bps": [-30.0],
            "y_direction_short_score_bps": [20.0],
            "y_long_path_utility_bps": [-30.0],
            "y_short_path_utility_bps": [20.0],
            "y_long_bad_path": [1.0],
            "y_short_bad_path": [0.0],
            "y_long_expected_mae_bps": [3.0],
            "y_short_expected_mae_bps": [1.0],
        }
    )

    failures = trainer._xau_direction_repair_target_failures("train", frame)

    assert any("long utility is not the declared future-outcome formula" in row for row in failures)


def test_entry_v10_xau_direction_repair_target_contract_uses_float32_path_semantics() -> None:
    import numpy as np
    import pandas as pd

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    mfe_short = np.float32(347.28643798828125)
    mae_short = np.float32(3.334178924560547)
    path_quality = np.float32(mfe_short - mae_short)
    frame = pd.DataFrame(
        {
            "y_direction": [1],
            "y_bad_path": [0.0],
            "y_trade": [1.0],
            "y_tradable": [1.0],
            "y_side": [1],
            "y_side_mask": [1.0],
            "mae_first_n_bps": [float(mae_short)],
            "mfe_first_n_bps": [float(mfe_short)],
            "path_quality_bps": [float(path_quality)],
            "y_position_size_target": [0.75],
            "mfe_long_first_n_bps": [-3.3330674171447754],
            "mae_long_first_n_bps": [355.1455078125],
            "mfe_short_first_n_bps": [float(mfe_short)],
            "mae_short_first_n_bps": [float(mae_short)],
            "bad_path_long_first_n": [1.0],
            "bad_path_short_first_n": [0.0],
            "y_long_final_pnl_at_direction_horizon_bps": [-358.47857666015625],
            "y_short_final_pnl_at_direction_horizon_bps": [float(path_quality)],
            "y_direction_target_mode_id": [1],
            "y_direction_long_score_bps": [-857.68212890625],
            "y_direction_short_score_bps": [547.65625],
            "y_long_path_utility_bps": [-857.68212890625],
            "y_short_path_utility_bps": [547.65625],
            "y_long_bad_path": [1.0],
            "y_short_bad_path": [0.0],
            "y_long_expected_mae_bps": [355.1455078125],
            "y_short_expected_mae_bps": [float(mae_short)],
            "y_rising_channel_support_touch": [0.0],
            "y_falling_channel_resistance_touch": [0.0],
            "y_support_retest_continuation": [0.0],
            "y_resistance_retest_continuation": [0.0],
            "y_countertrend_short_trap": [0.0],
            "y_countertrend_long_trap": [0.0],
            "y_long_high_mae_low_mfe_early_failure": [0.0],
            "y_short_high_mae_low_mfe_early_failure": [0.0],
        }
    )

    assert trainer._xau_direction_repair_target_failures("train", frame) == []


def test_entry_v10_train_and_validate_share_symmetric_aux_helpers() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")
    assert text.count("_direction_ce_sample_weight(") >= 3
    assert text.count("_aux_selector_mask(y_selector_long_mask, y_selector_short_mask)") >= 2
    assert text.count("_aux_clean_edge_target(y_clean_edge_long, y_clean_edge_bidir)") >= 2
    assert text.count("_aux_survival_target(y_survival_long, y_survival_bidir)") >= 2
    assert text.count("_clean_edge_rank_masks(") >= 3


def test_entry_v10_symmetric_aux_helpers_use_short_side(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_SYMMETRIC_NEGATIVES", True)

    long_selector = torch.tensor([1.0, 0.0, 0.0])
    short_selector = torch.tensor([0.0, 1.0, 0.0])
    assert trainer._aux_selector_mask(long_selector, short_selector).tolist() == [True, True, False]

    clean_long = torch.tensor([1.0, 0.0, 0.0])
    clean_bidir = torch.tensor([1.0, 1.0, 0.0])
    survival_long = torch.tensor([0.0, 1.0, 0.0])
    survival_bidir = torch.tensor([1.0, 1.0, 0.0])
    assert trainer._aux_clean_edge_target(clean_long, clean_bidir).tolist() == [1.0, 1.0, 0.0]
    assert trainer._aux_survival_target(survival_long, survival_bidir).tolist() == [1.0, 1.0, 0.0]


def test_entry_v10_selected_side_bad_path_penalty_is_long_short_swap_invariant() -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    probs = torch.tensor(
        [
            [0.70, 0.20, 0.10],
            [0.15, 0.65, 0.20],
            [0.50, 0.30, 0.20],
            [0.25, 0.55, 0.20],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    direction = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    bad_path = torch.tensor([1.0, 1.0, 0.0, 1.0])
    loss = trainer._selected_side_bad_path_probability_penalty(
        probs,
        direction,
        bad_path,
        0.24,
    )
    loss.backward()
    grad = probs.grad.detach().clone()

    swapped_probs = (
        probs.detach()[:, [1, 0, 2]].clone().requires_grad_(True)
    )
    swapped_direction = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    swapped_loss = trainer._selected_side_bad_path_probability_penalty(
        swapped_probs,
        swapped_direction,
        bad_path,
        0.24,
    )
    swapped_loss.backward()

    assert torch.allclose(loss, swapped_loss, atol=0.0, rtol=0.0)
    assert torch.allclose(
        grad[:, [1, 0, 2]],
        swapped_probs.grad,
        atol=0.0,
        rtol=0.0,
    )


def test_entry_v10_selected_side_bad_path_penalty_touches_only_selected_probability() -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    probs = torch.tensor(
        [
            [0.70, 0.20, 0.10],
            [0.15, 0.65, 0.20],
            [0.20, 0.25, 0.55],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    loss = trainer._selected_side_bad_path_probability_penalty(
        probs,
        torch.tensor([0, 1, 2]),
        torch.tensor([1.0, 1.0, 0.0]),
        0.40,
    )
    loss.backward()

    expected_grad = torch.tensor(
        [
            [0.20, 0.00, 0.00],
            [0.00, 0.20, 0.00],
            [0.00, 0.00, 0.00],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(loss, torch.tensor(0.27), atol=1e-7, rtol=0.0)
    assert torch.allclose(probs.grad, expected_grad, atol=0.0, rtol=0.0)


def test_entry_v10_selected_side_bad_path_penalty_rejects_flat_positive_even_when_disabled() -> None:
    import pytest
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    with pytest.raises(
        RuntimeError,
        match="ENTRY_SELECTED_SIDE_BAD_PATH_FLAT_TARGET_INVALID",
    ):
        trainer._selected_side_bad_path_probability_penalty(
            torch.tensor([[0.20, 0.20, 0.60]], dtype=torch.float32),
            torch.tensor([2]),
            torch.tensor([1.0]),
            0.0,
        )


def test_entry_v10_train_and_validate_share_selected_side_bad_path_penalty_and_stats() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")

    assert text.count("_selected_side_bad_path_probability_penalty(") >= 3
    assert "probs[bad_path_neg_mask, 0]" not in text
    assert text.count('"aux_bad_path_bce_loss_mean"') == 2
    assert text.count('"bad_path_prob_penalty_loss_mean"') == 2
    assert '"bad_path_prob_loss_mean"' not in text


def test_entry_v10_aux_pos_weights_match_active_bce_target_and_selector(monkeypatch) -> None:
    import pandas as pd
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    frame = pd.DataFrame(
        {
            "y_selector_long_mask": [1.0, 0.0, 1.0, 0.0],
            "y_selector_short_mask": [0.0, 1.0, 0.0, 0.0],
            "y_clean_edge_long": [1.0, 0.0, 0.0, 1.0],
            "y_clean_edge_bidir": [1.0, 1.0, 0.0, 0.0],
            "y_survival_long": [0.0, 0.0, 1.0, 1.0],
            "y_survival_bidir": [0.0, 1.0, 0.0, 0.0],
        }
    )

    target_specs = (
        (
            "clean_edge",
            "y_clean_edge_long",
            "y_clean_edge_bidir",
            trainer._aux_clean_edge_target,
        ),
        (
            "survival",
            "y_survival_long",
            "y_survival_bidir",
            trainer._aux_survival_target,
        ),
    )
    for symmetric in (False, True):
        monkeypatch.setattr(trainer, "ENTRY_SYMMETRIC_NEGATIVES", symmetric)
        selector = trainer._aux_selector_mask(
            torch.tensor(frame["y_selector_long_mask"].to_numpy()),
            torch.tensor(frame["y_selector_short_mask"].to_numpy()),
        )
        for target_name, long_column, bidir_column, target_helper in target_specs:
            active_target = target_helper(
                torch.tensor(frame[long_column].to_numpy()),
                torch.tensor(frame[bidir_column].to_numpy()),
            )
            expected_rate = float(active_target[selector].mean().item())
            measured_rate = trainer._active_aux_target_rate_from_frame(
                frame,
                split_name="train",
                target_name=target_name,
                long_column=long_column,
                bidir_column=bidir_column,
            )
            raw_weight, capped_weight = trainer._positive_class_weight_from_rate(
                measured_rate,
                10.0,
            )
            expected_raw_weight = (
                (1.0 - expected_rate) / expected_rate
                if expected_rate > 0.0
                else 1.0
            )

            assert measured_rate == expected_rate
            assert raw_weight == expected_raw_weight
            assert capped_weight == min(10.0, max(1.0, expected_raw_weight))


def _live_active_head_epoch_accumulator(trainer):
    import numpy as np

    rows = 32
    base = np.linspace(-1.0, 1.0, rows, dtype=np.float64)
    accumulator = trainer._new_active_head_epoch_accumulator()
    for head_name, component_names in trainer._ACTIVE_HEAD_TARGET_COMPONENTS.items():
        for component_name in component_names:
            width = int(trainer._ACTIVE_HEAD_COMPONENT_WIDTHS[component_name])
            prediction = np.stack(
                [base + 0.01 * column for column in range(width)],
                axis=1,
            )
            target = np.stack(
                [base[::-1] + 0.02 * column for column in range(width)],
                axis=1,
            )
            accumulator["heads"][head_name]["components"][component_name] = {
                "prediction": [prediction],
                "target": [target],
            }
        accumulator["heads"][head_name]["influence"] = [
            np.stack([0.10 * base, -0.10 * base, 0.05 * base], axis=1)
        ]
    return accumulator


def test_entry_v10_active_head_diagnostic_contract_covers_all_22_heads() -> None:
    from gx1.contracts.entry_model_native_readiness_v1 import (
        MODEL_NATIVE_ACTIVE_HEADS,
    )
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    assert len(MODEL_NATIVE_ACTIVE_HEADS) == 22
    assert tuple(trainer._ACTIVE_HEAD_FUSION_INPUTS) == MODEL_NATIVE_ACTIVE_HEADS
    assert tuple(trainer._ACTIVE_HEAD_TARGET_COMPONENTS) == MODEL_NATIVE_ACTIVE_HEADS
    assert trainer._active_head_contract_failures() == []

    metrics, failures = trainer._active_head_epoch_diagnostics(
        _live_active_head_epoch_accumulator(trainer)
    )

    assert failures == []
    assert metrics["active_head_health_ok"] is True
    assert tuple(metrics["active_head_diagnostics"]) == MODEL_NATIVE_ACTIVE_HEADS
    assert all(
        details["ok"] is True
        for details in metrics["active_head_diagnostics"].values()
    )


def test_entry_v10_every_active_head_dead_output_blocks_checkpoint_health() -> None:
    import numpy as np

    from gx1.contracts.entry_model_native_readiness_v1 import (
        MODEL_NATIVE_ACTIVE_HEADS,
    )
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    for head_name in MODEL_NATIVE_ACTIVE_HEADS:
        accumulator = _live_active_head_epoch_accumulator(trainer)
        component_name = trainer._ACTIVE_HEAD_TARGET_COMPONENTS[head_name][0]
        prediction = accumulator["heads"][head_name]["components"][component_name][
            "prediction"
        ][0]
        accumulator["heads"][head_name]["components"][component_name][
            "prediction"
        ] = [np.zeros_like(prediction)]

        metrics, failures = trainer._active_head_epoch_diagnostics(accumulator)

        assert metrics["active_head_health_ok"] is False
        assert any(
            "ENTRY_ACTIVE_HEAD_DIAGNOSTIC_OUTPUT_DEAD" in failure
            and f"head={head_name}" in failure
            for failure in failures
        )


def test_entry_v10_active_head_dead_target_or_fusion_influence_blocks_health() -> None:
    import numpy as np

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    target_dead = _live_active_head_epoch_accumulator(trainer)
    target_dead["heads"]["timing"]["components"]["timing_pred"]["target"] = [
        np.zeros((32, 12), dtype=np.float64)
    ]
    target_metrics, target_failures = trainer._active_head_epoch_diagnostics(
        target_dead
    )
    assert target_metrics["active_head_health_ok"] is False
    assert any(
        "ENTRY_ACTIVE_HEAD_DIAGNOSTIC_TARGET_DEAD" in failure
        and "head=timing" in failure
        for failure in target_failures
    )

    influence_dead = _live_active_head_epoch_accumulator(trainer)
    influence_dead["heads"]["offline_rl_expectile_value"]["influence"] = [
        np.zeros((32, 3), dtype=np.float64)
    ]
    influence_metrics, influence_failures = trainer._active_head_epoch_diagnostics(
        influence_dead
    )
    assert influence_metrics["active_head_health_ok"] is False
    assert any(
        "ENTRY_ACTIVE_HEAD_DIAGNOSTIC_INFLUENCE_DEAD" in failure
        and "head=offline_rl_expectile_value" in failure
        for failure in influence_failures
    )

    structural_flat_q = _live_active_head_epoch_accumulator(trainer)
    action_component = structural_flat_q["heads"]["offline_rl_action_value"][
        "components"
    ]["action_value"]
    action_component["prediction"][0][:, 6:9] = 0.0
    action_component["target"][0][:, 6:9] = 0.0
    structural_metrics, structural_failures = (
        trainer._active_head_epoch_diagnostics(structural_flat_q)
    )
    assert structural_failures == []
    assert structural_metrics["active_head_health_ok"] is True


def test_entry_v10_active_head_target_surfaces_match_exact_output_widths(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_SYMMETRIC_NEGATIVES", True)
    rows = 32
    base = torch.linspace(-1.0, 1.0, rows)
    out = {
        output_name: torch.stack(
            [base + 0.01 * column for column in range(width)],
            dim=1,
        )
        for output_name, width in trainer.EXACT_EVIDENCE_FUSION_OUTPUTS
    }
    out["raw_direction_logits"] = torch.stack(
        [base, -base, 0.5 * base],
        dim=1,
    )
    batch = {
        "y": torch.tensor([index % 3 for index in range(rows)]),
        "y_tradable": torch.tensor([float(index % 2) for index in range(rows)]),
        "y_selector_long_mask": torch.ones(rows),
        "y_selector_short_mask": torch.ones(rows),
        "y_clean_edge_long": torch.tensor(
            [float(index % 2) for index in range(rows)]
        ),
        "y_clean_edge_bidir": torch.tensor(
            [float((index + 1) % 2) for index in range(rows)]
        ),
        "y_survival_long": torch.tensor(
            [float(index % 2) for index in range(rows)]
        ),
        "y_survival_bidir": torch.tensor(
            [float((index // 2) % 2) for index in range(rows)]
        ),
        "path_quality_bps": 20.0 * base,
        "mfe_first_n_bps": 10.0 * base,
        "y_bad_path": torch.tensor(
            [float(index % 2) for index in range(rows)]
        ),
        "y_tf_agreement_score": (base + 1.0) / 2.0,
        "y_position_size_target": (base + 1.0) / 2.0,
        "y_trade": torch.tensor([float(index % 2) for index in range(rows)]),
        "y_side": torch.tensor([index % 2 for index in range(rows)]),
        "y_side_mask": torch.ones(rows),
        "y_long_path_utility_bps": 30.0 * base,
        "y_short_path_utility_bps": -25.0 * base,
        "y_long_bad_path": torch.tensor(
            [float(index % 2) for index in range(rows)]
        ),
        "y_short_bad_path": torch.tensor(
            [float((index + 1) % 2) for index in range(rows)]
        ),
        "y_long_expected_mae_bps": 5.0 * (base + 1.0),
        "y_short_expected_mae_bps": 6.0 * (1.0 - base),
        "y_rising_channel_support_touch": torch.tensor(
            [float(index % 2) for index in range(rows)]
        ),
        "y_falling_channel_resistance_touch": torch.tensor(
            [float((index + 1) % 2) for index in range(rows)]
        ),
        "y_countertrend_short_trap": torch.tensor(
            [float((index // 2) % 2) for index in range(rows)]
        ),
        "y_countertrend_long_trap": torch.tensor(
            [float((index // 3) % 2) for index in range(rows)]
        ),
        "y_short_high_mae_low_mfe_early_failure": torch.tensor(
            [float((index // 4) % 2) for index in range(rows)]
        ),
        "y_long_high_mae_low_mfe_early_failure": torch.tensor(
            [float((index // 5) % 2) for index in range(rows)]
        ),
    }
    for direction in trainer.DIP_DIRECTIONS:
        for horizon in trainer.DIP_HORIZONS:
            for target_name in trainer.DIP_TARGETS:
                column = (
                    f"y_dip_mfe_{direction}_K{horizon}"
                    if target_name.startswith("recovery")
                    else f"y_dip_mae_{direction}_K{horizon}"
                )
                batch[column] = base + float(horizon)
    for horizon in trainer.FORECAST_HORIZONS:
        batch[f"y_forecast_ret_K{horizon}"] = base + float(horizon)
    for direction in trainer.TIMING_DIRECTIONS:
        for horizon in trainer.TIMING_HORIZONS:
            for target_name in trainer.TIMING_TARGETS:
                batch[f"y_{target_name}_{direction}_K{horizon}"] = (
                    (base + 1.0) / 2.0
                )
    for direction in trainer.TAIL_RISK_DIRECTIONS:
        for horizon in trainer.TAIL_RISK_HORIZONS:
            batch[f"y_tail_mae_{direction}_K{horizon}"] = (
                base + float(horizon)
            )
    for horizon in trainer.VOL_FORECAST_HORIZONS:
        batch[f"y_vol_fwd_K{horizon}"] = base + float(horizon)
    for offset, column in enumerate(trainer._OFFLINE_RL_TARGET_COLS):
        batch[column] = (10.0 + float(offset)) * base + float(offset)

    surfaces = trainer._active_head_target_surfaces(
        out,
        batch,
        torch.device("cpu"),
        path_scale_bps=50.0,
        mfe_scale_bps=20.0,
    )

    assert tuple(surfaces) == tuple(trainer.MODEL_NATIVE_ACTIVE_HEADS)
    for head_name, components in surfaces.items():
        assert tuple(components) == trainer._ACTIVE_HEAD_TARGET_COMPONENTS[head_name]
        for component_name, (prediction, target, mask) in components.items():
            expected_width = trainer._ACTIVE_HEAD_COMPONENT_WIDTHS[component_name]
            assert tuple(prediction.shape) == (rows, expected_width)
            assert tuple(target.shape) == (rows, expected_width)
            assert tuple(mask.shape) == (rows,)

    class _FusionStub(torch.nn.Module):
        def _fuse_direction_evidence(self, pre_fusion_outputs):
            evidence = torch.cat(
                [
                    pre_fusion_outputs[output_name]
                    for output_name, _width in trainer.EXACT_EVIDENCE_FUSION_OUTPUTS
                ],
                dim=1,
            )
            scalar = evidence.sum(dim=1)
            return torch.stack([scalar, -0.5 * scalar, 0.25 * scalar], dim=1)

    fusion_stub = _FusionStub()
    out["raw_direction_logits"] = fusion_stub._fuse_direction_evidence(out)
    accumulator = trainer._new_active_head_epoch_accumulator()
    trainer._accumulate_active_head_epoch(
        accumulator,
        fusion_stub,
        out,
        batch,
        torch.device("cpu"),
        path_scale_bps=50.0,
        mfe_scale_bps=20.0,
    )
    metrics, failures = trainer._active_head_epoch_diagnostics(accumulator)

    assert failures == []
    assert metrics["active_head_health_ok"] is True


def test_entry_v10_checkpoint_admission_requires_all_active_head_health() -> None:
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    text = TRAINER_PATH.read_text(encoding="utf-8")
    assert 'val_stats.get("active_head_health_ok", False)' in text
    assert "[ENTRY_ACTIVE_HEAD_HEALTH_CHECKPOINT_BLOCKED]" in text
    assert "[ENTRY_ACTIVE_HEAD_HEALTH]" in text

    # Active-head health is mandatory in BOTH profiles: it is the liveness
    # proof, not an acceptance metric. Candidate additionally requires
    # auxiliary and cooperation health (see
    # tests/test_entry_profile_separated_checkpoint_admission.py for the
    # complete profile matrix).
    for profile in ("smoke", "candidate"):
        assert (
            trainer._checkpoint_admission_ok(
                profile=profile,
                aux_head_health_ok=True,
                active_head_health_ok=False,
                cooperation_gate_health_ok=True,
                class_support_ok=True,
            )
            is False
        )
        assert (
            trainer._checkpoint_admission_ok(
                profile=profile,
                aux_head_health_ok=True,
                active_head_health_ok=True,
                cooperation_gate_health_ok=True,
                class_support_ok=True,
            )
            is True
        )


def test_entry_v10_direction_ce_weight_includes_bad_path_and_short_side(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_SYMMETRIC_NEGATIVES", True)
    monkeypatch.setattr(trainer, "ENTRY_DEAD_LONG_CE_MULTIPLIER", 1.80)
    monkeypatch.setattr(trainer, "ENTRY_TEASER_LONG_CE_MULTIPLIER", 1.35)
    monkeypatch.setattr(trainer, "ENTRY_BAD_PATH_CE_MULTIPLIER", 1.50)
    monkeypatch.setattr(trainer, "ENTRY_HARD_NEG_LONG_CE_MULTIPLIER", 1.35)

    weight = trainer._direction_ce_sample_weight(
        y_bad_path=torch.tensor([0.0, 0.0, 1.0]),
        y_dead_negative_long=torch.tensor([1.0, 0.0, 0.0]),
        y_teaser_negative_long=torch.zeros(3),
        residual_hard_neg_long=torch.zeros(3),
        y_dead_negative_short=torch.tensor([0.0, 1.0, 0.0]),
        y_teaser_negative_short=torch.zeros(3),
        residual_hard_neg_short=torch.tensor([0.0, 0.0, 1.0]),
    )

    assert float(weight[0]) > 1.0
    assert float(weight[1]) > 1.0
    assert float(weight[2]) > 1.0


def test_entry_v10_clean_edge_rank_masks_use_bidir_short_negatives(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_SYMMETRIC_NEGATIVES", True)

    clean_pos, ranked_neg = trainer._clean_edge_rank_masks(
        y_clean_edge_long=torch.tensor([1.0, 0.0, 0.0]),
        y_clean_edge_bidir=torch.tensor([1.0, 1.0, 0.0]),
        y_dead_negative_long=torch.zeros(3),
        y_teaser_negative_long=torch.zeros(3),
        residual_hard_neg_long=torch.zeros(3),
        y_dead_negative_short=torch.tensor([0.0, 0.0, 1.0]),
        y_teaser_negative_short=torch.zeros(3),
        residual_hard_neg_short=torch.zeros(3),
    )

    assert clean_pos.tolist() == [True, True, False]
    assert ranked_neg.tolist() == [False, False, True]


def test_entry_v10_clean_edge_rank_masks_use_outcome_labels_without_teacher(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_SYMMETRIC_NEGATIVES", False)

    clean_pos, ranked_neg = trainer._clean_edge_rank_masks(
        y_clean_edge_long=torch.tensor([1.0, 0.0, 0.0]),
        y_clean_edge_bidir=torch.zeros(3),
        y_dead_negative_long=torch.zeros(3),
        y_teaser_negative_long=torch.tensor([0.0, 1.0, 0.0]),
        residual_hard_neg_long=torch.tensor([0.0, 0.0, 1.0]),
        y_dead_negative_short=torch.zeros(3),
        y_teaser_negative_short=torch.zeros(3),
        residual_hard_neg_short=torch.zeros(3),
    )

    assert clean_pos.tolist() == [True, False, False]
    assert ranked_neg.tolist() == [False, True, True]


def test_entry_v10_trainer_has_no_stale_teacher_label_plumbing() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")

    assert "y_teacher_bad_long" not in text
    assert "y_teacher_winner_long" not in text
    assert "y_v6_teacher_bad_long" not in text
    assert "y_v6_teacher_winner_long" not in text


def test_entry_v10_path_quality_rank_loss_penalizes_inverted_order(monkeypatch) -> None:
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_PATH_QUALITY_RANK_WEIGHT", 1.0)
    monkeypatch.setattr(trainer, "ENTRY_PATH_QUALITY_RANK_MARGIN", 0.20)
    monkeypatch.setattr(trainer, "ENTRY_PATH_QUALITY_RANK_QUANTILE", 0.25)

    quality = torch.arange(12, dtype=torch.float32)
    aligned_pred = (quality / 10.0).reshape(-1, 1)
    inverted_pred = torch.flip(aligned_pred, dims=[0])

    aligned = trainer._path_quality_rank_loss(aligned_pred, quality, torch.device("cpu"))
    inverted = trainer._path_quality_rank_loss(inverted_pred, quality, torch.device("cpu"))

    assert float(aligned.item()) == 0.0
    assert float(inverted.item()) > 0.5


def test_entry_v10_direction_ckpt_balance_guard_penalizes_class_collapse(monkeypatch) -> None:
    import numpy as np

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT", 0.50)
    monkeypatch.setattr(trainer, "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL", 0.35)
    monkeypatch.setattr(trainer, "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE", 0.05)

    targets = np.asarray([0] * 10 + [1] * 10 + [2] * 10)
    balanced = trainer._direction_ckpt_balance_stats(targets, targets, 1.0)
    collapsed = trainer._direction_ckpt_balance_stats(
        targets,
        np.asarray([0] * 10 + [1] * 20),
        20.0 / 30.0,
    )

    assert balanced["direction_class_balance_guard_ok"] is True
    assert balanced["direction_ckpt_score"] == 1.0
    assert collapsed["direction_class_balance_guard_ok"] is False
    assert collapsed["direction_pred_rate_flat"] == 0.0
    assert collapsed["direction_ckpt_score"] < collapsed["direction_ckpt_balance_penalty"]


def test_entry_v10_direction_ckpt_balance_guard_required_requires_active_thresholds(monkeypatch) -> None:
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_CKPT_CLASS_BALANCE_GUARD_WEIGHT", 0.50)
    monkeypatch.setattr(trainer, "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL", 0.35)
    monkeypatch.setattr(trainer, "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE", 0.05)

    assert trainer._direction_ckpt_balance_guard_required() is True

    monkeypatch.setattr(trainer, "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_TO_LABEL", 0.0)
    monkeypatch.setattr(trainer, "ENTRY_CKPT_CLASS_BALANCE_MIN_PRED_RATE", 0.0)

    assert trainer._direction_ckpt_balance_guard_required() is False


def test_entry_v10_direction_ckpt_slice_guard_required(monkeypatch) -> None:
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_CKPT_DIRECTION_SLICE_GUARD", False)
    assert trainer._direction_ckpt_slice_guard_required() is False

    monkeypatch.setattr(trainer, "ENTRY_CKPT_DIRECTION_SLICE_GUARD", True)
    assert trainer._direction_ckpt_slice_guard_required() is True


def test_entry_v10_train_refuses_to_write_bundle_when_best_class_balance_guard_failed() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")

    assert "[TRAIN_FAIL_DIRECTION_CLASS_BALANCE_GUARD]" in text
    assert "_direction_ckpt_balance_guard_required()" in text
    assert "FAIL_DIRECTION_CLASS_BALANCE_GUARD" in text
    assert "[ENTRY_DIR_CLASS_BALANCE_FAILURE_EVIDENCE]" in text
    assert "refusing to write a collapsed direction bundle" in text




def test_entry_v10_train_refuses_to_write_bundle_when_best_slice_guard_failed() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")

    assert "[TRAIN_FAIL_DIRECTION_SLICE_GUARD]" in text
    assert "_direction_ckpt_slice_guard_required()" in text
    assert "refusing to write a slice-failed direction bundle" in text


def _evidence_fusion_movement_states():
    import torch

    from gx1.contracts.entry_model_native_learned_component_movement_v1 import (
        PARAMETER_SHAPES,
    )

    initial = {
        key: torch.zeros(tuple(shape), dtype=torch.float32)
        for key, shape in PARAMETER_SHAPES.items()
    }
    selected = {
        key: torch.full(tuple(shape), 0.01 * (index + 1), dtype=torch.float32)
        for index, (key, shape) in enumerate(PARAMETER_SHAPES.items())
    }
    for row in range(3):
        selected["evidence_fusion_out.weight"][row].fill_(0.01 * (row + 1))
    return initial, selected


def test_entry_v10_train_model_uses_exact_evidence_fusion_contract() -> None:
    from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import (
        FUSION_MODE,
        HIDDEN_DIM,
        INPUTS,
        INPUT_DIM,
        OUTPUT_DIM,
        direction_evidence_fusion_metadata,
    )

    text = TRAINER_PATH.read_text(encoding="utf-8")
    train_ctor = text.split("model = EntryV10CtxHybridTransformer(", 1)[1].split(").to(device)", 1)[0]
    assert len(INPUTS) == 26
    assert sum(width for _, width in INPUTS) == INPUT_DIM == 96
    assert (HIDDEN_DIM, OUTPUT_DIM) == (128, 3)
    assert FUSION_MODE == "sole_learned_acyclic_96x128x3"
    assert direction_evidence_fusion_metadata()["sole_direction_path"] is True
    assert "_capture_evidence_fusion_initial_state(model)" in text
    for retired in (
        "residual_scale=",
        "hierarchical_composition_",
        "hierarchical_public_",
        "enable_hierarchical_ctx_",
    ):
        assert retired not in train_ctor


def test_entry_v10_evidence_fusion_movement_proof_has_exact_eight_parameter_schema() -> None:
    import pytest

    from gx1.contracts.entry_model_native_learned_component_movement_v1 import (
        COMPONENT_PARAMETERS,
        PARAMETER_SHAPES,
        require_learned_component_movement_metadata,
    )
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    initial, selected = _evidence_fusion_movement_states()
    proof = trainer._model_native_evidence_fusion_movement_proof(
        initial,
        selected,
        selected_checkpoint_epoch=4,
    )

    # Six evidence-fusion parameters plus the zero-initialized specialist_out
    # pair: movement there proves the eight-family branch left initialization.
    assert len(PARAMETER_SHAPES) == 8
    assert set(proof["parameter_deltas"]) == set(PARAMETER_SHAPES)
    assert all(row["changed"] is True for row in proof["parameter_deltas"].values())
    assert proof["component_changed"] == {name: True for name in COMPONENT_PARAMETERS}
    assert proof["output_rows_distinct"] is True
    assert proof["decision"] == "PASS"
    assert require_learned_component_movement_metadata(proof, context="TEST") == proof

    missing_parameter = {
        **proof,
        "parameter_deltas": dict(proof["parameter_deltas"]),
    }
    del missing_parameter["parameter_deltas"]["evidence_fusion_norm.bias"]
    with pytest.raises(RuntimeError, match="LEARNED_COMPONENT_MOVEMENT_PARAMETERS_INVALID"):
        require_learned_component_movement_metadata(
            missing_parameter,
            context="TEST",
        )


def test_entry_v10_evidence_fusion_movement_proof_requires_each_component_to_move() -> None:
    import pytest

    from gx1.contracts.entry_model_native_learned_component_movement_v1 import (
        COMPONENT_PARAMETERS,
    )
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    for component, keys in COMPONENT_PARAMETERS.items():
        initial, selected = _evidence_fusion_movement_states()
        for key in keys:
            selected[key] = initial[key].clone()
        with pytest.raises(
            RuntimeError,
            match=f"{component}:no_learned_parameter_movement",
        ):
            trainer._model_native_evidence_fusion_movement_proof(
                initial,
                selected,
                selected_checkpoint_epoch=4,
            )


def test_entry_v10_evidence_fusion_movement_proof_requires_distinct_output_rows() -> None:
    import pytest

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    initial, selected = _evidence_fusion_movement_states()
    selected["evidence_fusion_out.weight"][:] = selected[
        "evidence_fusion_out.weight"
    ][0]

    with pytest.raises(RuntimeError, match="class_rows_not_distinct"):
        trainer._model_native_evidence_fusion_movement_proof(
            initial,
            selected,
            selected_checkpoint_epoch=4,
        )


def test_entry_v10_model_exposes_only_the_learned_direction_fusion_path() -> None:
    model_path = TRAINER_PATH.with_name("entry_v10_ctx_hybrid_transformer.py")
    text = model_path.read_text(encoding="utf-8")

    assert "self.evidence_fusion_norm = nn.LayerNorm(EXACT_EVIDENCE_FUSION_INPUT_DIM)" in text
    assert "self.evidence_fusion_in = nn.Linear(" in text
    assert "self.evidence_fusion_out = nn.Linear(EXACT_EVIDENCE_FUSION_HIDDEN_DIM, 3)" in text
    assert "raw_direction_logits = self._fuse_direction_evidence(pre_fusion_outputs)" in text
    assert "direction_logits = raw_direction_logits" in text
    assert '"raw_direction_logits": raw_direction_logits' in text
    assert text.index("raw_direction_logits = self._fuse_direction_evidence") < text.index(
        "public_trade_flat_decision_logits = torch.stack"
    )
    for retired in (
        "hierarchical_direction_base_logits",
        "hierarchical_direction_residual_logits",
        "hierarchical_ctx_prior_adapter",
        "public_trade_logit",
        "public_flat_logit",
        "public_side_logits",
        "margin_bridge",
    ):
        assert retired not in text






def test_model_native_train_wrappers_require_exact_audited_positive_recipe() -> None:
    from gx1.contracts.entry_model_native_train_launch_v1 import (
        MODEL_NATIVE_RECIPE_ENV_KEYS,
        RECIPE_AUDIT_SCHEMA,
    )
    from gx1.contracts.entry_model_native_training_objective_v1 import (
        REQUIRED_POSITIVE_LOSS_WEIGHTS,
    )

    repo = Path(__file__).resolve().parents[1]
    wrapper = (
        repo / "scripts" / "run_entry_model_native_seq513_train.sh"
    ).read_text(encoding="utf-8")

    assert "--recipe-audit-json" in wrapper
    assert "gx1.contracts.entry_model_native_train_launch_v1" in wrapper
    assert "ENTRY_HIER_LEGACY_CE_MULT" not in wrapper
    assert "--profile" in wrapper

    assert set(REQUIRED_POSITIVE_LOSS_WEIGHTS).issubset(MODEL_NATIVE_RECIPE_ENV_KEYS)
    assert "ENTRY_HIER_LEGACY_CE_MULT" not in MODEL_NATIVE_RECIPE_ENV_KEYS
    assert RECIPE_AUDIT_SCHEMA == "entry_model_native_seq513_train_recipe_audit_v4"


def test_entry_v10_standalone_eval_matches_training_objective() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")

    assert '"validation_objective_matches_train": True' in text
    assert "direction_only_ce_plus_residual_side_bias" not in text
    assert '"hierarchical_loss_metrics_included": False' not in text


def test_entry_v10_has_one_deterministic_low_memory_execution_path() -> None:
    source = TRAINER_PATH.read_text(encoding="utf-8")

    assert "gx1.utils.fast_train" not in source
    assert "GX1_FAST_TRAIN" not in source
    assert 'parser.add_argument("--fast"' not in source
    assert "torch.compile" not in source
    assert "torch.autocast" not in source
    assert "num_workers must equal 0" in source
    assert "torch.use_deterministic_algorithms(True)" in source


def test_entry_v10_grad_accum_cli_drives_steps_and_rescales_final_remainder(
    monkeypatch,
) -> None:
    import pytest
    import torch

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.weight.grad = torch.tensor([[0.5]])

    monkeypatch.setattr(trainer, "_GRAD_CLIP_NORM", 1_000.0)
    assert trainer._step_partial_gradient_accumulation(
        model=model,
        optimizer=optimizer,
        configured_steps=4,
        observed_steps=2,
    )
    assert float(model.weight.detach().item()) == pytest.approx(0.9)
    assert model.weight.grad is None

    with pytest.raises(RuntimeError, match="GRAD_ACCUM_REMAINDER_INVALID"):
        trainer._step_partial_gradient_accumulation(
            model=model,
            optimizer=optimizer,
            configured_steps=4,
            observed_steps=4,
        )

    source = TRAINER_PATH.read_text(encoding="utf-8")
    assert 'parser.add_argument("--grad-accum-steps", type=int, required=True)' in source
    assert "grad_accum_steps=int(grad_accum_steps)" in source


def test_entry_v10_logit_adjust_tau_zero_is_bit_compatible(monkeypatch) -> None:
    """tau=0 is the exact-compatibility switch: no offset is built, the live
    logits tensor is passed through unchanged (same object, not just equal),
    so the CE graph is bit-identical to the pre-adjustment trainer."""
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_LOGIT_ADJUST_TAU", 0.0)
    offset = trainer._direction_log_prior_offset(
        train_long_rate=0.2,
        train_short_rate=0.2,
        train_flat_rate=0.6,
        device=torch.device("cpu"),
    )
    assert offset is None

    logits = torch.randn(5, 3, requires_grad=True)
    adjusted = trainer._direction_logit_adjusted_ce_logits(logits, offset)
    assert adjusted is logits

    targets = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    raw_ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    adjusted_ce = torch.nn.functional.cross_entropy(adjusted, targets, reduction="none")
    assert torch.equal(raw_ce, adjusted_ce)


def test_entry_v10_logit_adjust_offset_orders_by_direction_contract_indices(
    monkeypatch,
) -> None:
    import math

    import pytest

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.models.entry_v10.direction_decision_contract import (
        MODEL_DIRECTION_FLAT_INDEX,
        MODEL_DIRECTION_LONG_INDEX,
        MODEL_DIRECTION_SHORT_INDEX,
    )

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_LOGIT_ADJUST_TAU", 1.0)
    offset = trainer._direction_log_prior_offset(
        train_long_rate=0.2,
        train_short_rate=0.3,
        train_flat_rate=0.5,
        device=torch.device("cpu"),
    )
    assert offset is not None and offset.shape == (3,)
    assert float(offset[MODEL_DIRECTION_LONG_INDEX]) == pytest.approx(math.log(0.2))
    assert float(offset[MODEL_DIRECTION_SHORT_INDEX]) == pytest.approx(math.log(0.3))
    assert float(offset[MODEL_DIRECTION_FLAT_INDEX]) == pytest.approx(math.log(0.5))

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_LOGIT_ADJUST_TAU", 0.5)
    half = trainer._direction_log_prior_offset(
        train_long_rate=0.2,
        train_short_rate=0.3,
        train_flat_rate=0.5,
        device=torch.device("cpu"),
    )
    assert torch.allclose(half * 2.0, offset)


def test_entry_v10_logit_adjust_tau_one_shifts_ce_toward_the_prior(
    monkeypatch,
) -> None:
    """Hand example, priors (LONG,SHORT,FLAT)=(0.2,0.2,0.6): at zero logits
    softmax(log p) = p exactly, so the adjusted CE equals -log(prior of the
    target). Majority FLAT gets cheaper than the raw uniform CE log(3);
    minority LONG/SHORT get costlier — the adjustment handicaps the prior
    inside the loss instead of leaving CE to prefer collapse."""
    import math

    import pytest

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from gx1.models.entry_v10.direction_decision_contract import (
        MODEL_DIRECTION_FLAT_INDEX,
        MODEL_DIRECTION_LONG_INDEX,
        MODEL_DIRECTION_SHORT_INDEX,
    )

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_LOGIT_ADJUST_TAU", 1.0)
    offset = trainer._direction_log_prior_offset(
        train_long_rate=0.2,
        train_short_rate=0.2,
        train_flat_rate=0.6,
        device=torch.device("cpu"),
    )
    logits = torch.zeros(3, 3)
    targets = torch.tensor(
        [
            MODEL_DIRECTION_LONG_INDEX,
            MODEL_DIRECTION_SHORT_INDEX,
            MODEL_DIRECTION_FLAT_INDEX,
        ],
        dtype=torch.long,
    )
    raw_ce = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    adjusted_ce = torch.nn.functional.cross_entropy(
        trainer._direction_logit_adjusted_ce_logits(logits, offset),
        targets,
        reduction="none",
    )

    uniform = math.log(3.0)
    assert float(raw_ce[0]) == pytest.approx(uniform)
    # softmax(log p) == p: adjusted CE is exactly -log(prior_target).
    assert float(adjusted_ce[0]) == pytest.approx(-math.log(0.2))
    assert float(adjusted_ce[1]) == pytest.approx(-math.log(0.2))
    assert float(adjusted_ce[2]) == pytest.approx(-math.log(0.6))
    # Directional: minority targets cost more than raw, majority costs less.
    assert float(adjusted_ce[0]) > float(raw_ce[0])
    assert float(adjusted_ce[1]) > float(raw_ce[1])
    assert float(adjusted_ce[2]) < float(raw_ce[2])


def test_entry_v10_logit_adjust_zero_rate_fails_closed(monkeypatch) -> None:
    import pytest as _pytest

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_LOGIT_ADJUST_TAU", 1.0)
    with _pytest.raises(
        RuntimeError, match="ENTRY_DIRECTION_LOGIT_ADJUST_PRIOR_INVALID"
    ):
        trainer._direction_log_prior_offset(
            train_long_rate=0.4,
            train_short_rate=0.0,
            train_flat_rate=0.6,
            device=torch.device("cpu"),
        )


def test_entry_v10_direction_class_weights_neutral_under_adjust_sqrt_when_off(
    monkeypatch,
) -> None:
    import numpy as np
    import pytest

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    long_rate, short_rate = 0.204, 0.184  # V28-era TRAIN rates (documented)
    raw_long = (1.0 - long_rate) / long_rate
    raw_short = (1.0 - short_rate) / short_rate

    # tau > 0: adjustment replaces reweighting — all weights exactly 1.0.
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_LOGIT_ADJUST_TAU", 1.0)
    result = trainer._direction_class_weights(
        train_long_rate=long_rate, train_short_rate=short_rate
    )
    assert result[0] == pytest.approx(raw_long)
    assert result[1] == pytest.approx(raw_short)
    assert result[2:] == (1.0, 1.0, 1.0)

    # tau == 0: the exact 2026-08-08 sqrt-softened construction is unchanged.
    monkeypatch.setattr(trainer, "ENTRY_DIRECTION_LOGIT_ADJUST_TAU", 0.0)
    result_off = trainer._direction_class_weights(
        train_long_rate=long_rate, train_short_rate=short_rate
    )
    cap = float(trainer.ENTRY_DIRECTION_CLASS_WEIGHT_CAP)
    floor = float(trainer.ENTRY_FLAT_CLASS_WEIGHT_FLOOR)
    assert result_off[2] == pytest.approx(
        min(cap, max(1.0, float(np.sqrt(max(raw_long, 1.0)))))
    )
    assert result_off[3] == pytest.approx(
        min(cap, max(1.0, float(np.sqrt(max(raw_short, 1.0)))))
    )
    assert result_off[4] == pytest.approx(max(floor, 1.0))
    # The corrected algebra: sqrt weights do NOT equalize w_k * r_k.
    assert result_off[2] * long_rate != pytest.approx(result_off[3] * short_rate)


def test_entry_v10_logit_adjust_is_training_loss_only_and_serving_logits_raw() -> None:
    """Rule 6 / rule 3: the adjustment must never touch the model's emitted
    logits or the battery probabilities. Proven from source: the model module
    has no tau dependence, the helper never mutates its input, and every
    battery softmax consumes the raw logits."""
    import inspect

    from gx1.models.entry_v10 import entry_v10_ctx_hybrid_transformer as model_module
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    model_source = Path(model_module.__file__).read_text(encoding="utf-8")
    assert "logit_adjust" not in model_source.lower()
    forward_params = inspect.signature(
        model_module.EntryV10CtxHybridTransformer.forward
    ).parameters
    assert not any("tau" in name or "prior" in name for name in forward_params)

    logits = torch.randn(4, 3)
    logits_before = logits.clone()
    offset = torch.log(torch.tensor([0.2, 0.2, 0.6]))
    adjusted = trainer._direction_logit_adjusted_ce_logits(logits, offset)
    assert adjusted is not logits
    assert torch.equal(logits, logits_before)
    assert torch.allclose(adjusted, logits_before + offset)

    text = TRAINER_PATH.read_text(encoding="utf-8")
    # Both main direction CE call sites (train_epoch + validate) are adjusted;
    # the mtf aux CE is adjusted inside _direction_aux_ce_loss.
    assert (
        text.count(
            "_direction_logit_adjusted_ce_logits(logits, direction_log_prior_offset)"
        )
        == 2
    )
    assert "_direction_logit_adjusted_ce_logits(aux_logits, log_prior_offset)" in text
    # Battery/metric probabilities stay on raw logits: no softmax over the
    # adjusted logits anywhere.
    assert text.count("probs = torch.softmax(logits, dim=1)") >= 2
    assert "torch.softmax(_direction_logit_adjusted_ce_logits" not in text
