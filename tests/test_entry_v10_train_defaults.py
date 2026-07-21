from __future__ import annotations

import ast
import json
from pathlib import Path


TRAINER_PATH = (
    Path(__file__).resolve().parents[1]
    / "gx1"
    / "models"
    / "entry_v10"
    / "entry_v10_ctx_train_v3.py"
)


def _trainer_ast() -> ast.Module:
    return ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))


def _canonical_env_defaults(module: ast.Module) -> dict[str, str]:
    for node in module.body:
        if not (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_CANONICAL_ENTRY_TRAIN_ENV_DEFAULTS"
            and isinstance(node.value, ast.Dict)
        ):
            continue
        defaults: dict[str, str] = {}
        for key, value in zip(node.value.keys, node.value.values):
            if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                defaults[str(key.value)] = str(value.value)
        return defaults
    raise AssertionError("_CANONICAL_ENTRY_TRAIN_ENV_DEFAULTS not found")


def _env_str_defaults(module: ast.Module) -> dict[str, str]:
    defaults: dict[str, str] = {}
    for node in ast.walk(module):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_env_str"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[1], ast.Constant)
        ):
            continue
        defaults[str(node.args[0].value)] = str(node.args[1].value)
    return defaults


def test_entry_v10_env_defaults_match_canonical_guard_contract() -> None:
    module = _trainer_ast()
    canonical = _canonical_env_defaults(module)
    env_defaults = _env_str_defaults(module)

    mismatches = {
        key: {"env_str_default": env_defaults[key], "canonical_default": canonical[key]}
        for key in sorted(canonical.keys() & env_defaults.keys())
        if env_defaults[key] != canonical[key]
    }

    assert mismatches == {}


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
    import itertools
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

    first_batch = list(itertools.islice(iter(sampler), 12))
    batch_slices = set(ctx_cat[first_batch, 0].tolist())

    assert len(sampler) == 48
    assert sampler.audited_slice_count == 2
    assert batch_slices == {0, 1}


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

    loss = trainer._direction_aux_ce_loss(logits, targets, criterion, sample_weight)
    weighted_ce = (criterion.ce(logits, targets) * sample_weight).mean()
    balance_term = trainer._direction_balance_term(torch.softmax(logits, dim=1), targets, criterion)

    assert torch.allclose(loss, weighted_ce + balance_term)
    assert float(loss.item()) > float(criterion.ce(logits, targets).mean().item())


def test_entry_v10_train_and_validate_apply_mtf_aux_direction_repair() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")
    assert text.count('output_name="mtf_dir_logits"') == 2
    assert 'if "mtf_dir_logits" in out' not in text


def test_entry_v10_multi_tf_window_uses_m5_close_availability_for_closed_bar() -> None:
    import numpy as np
    import pandas as pd

    from gx1.features.htf_features import (
        HTF_V2_MATRIX_CONTRACT,
        MULTI_TF_FEATURE_COUNT_V2,
    )
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    target = pd.Timestamp("2026-07-08T18:00:00Z")
    ts = np.array(
        [
            (target - pd.Timedelta(minutes=5)).value,
            target.value,
        ],
        dtype=np.int64,
    )
    feats = np.stack(
        [
            np.full(MULTI_TF_FEATURE_COUNT_V2, 1.0, dtype=np.float32),
            np.full(MULTI_TF_FEATURE_COUNT_V2, 2.0, dtype=np.float32),
        ]
    )
    frame = pd.DataFrame(
        feats,
        index=pd.DatetimeIndex(ts.astype("datetime64[ns]"), tz="UTC"),
    )
    frame.attrs["ts_int64"] = ts
    frame.attrs["feats_np"] = feats
    frame.attrs["htf_feature_contract"] = HTF_V2_MATRIX_CONTRACT
    frame.attrs["causal_warmup_rows"] = 0

    dataset = trainer.EntryV10CtxDataset.__new__(trainer.EntryV10CtxDataset)
    dataset._multi_tf_feats = {"M5": frame}
    dataset._multi_tf_shift = {"M5": pd.Timedelta(minutes=5)}
    dataset._multi_tf_target_availability_shift = pd.Timedelta(minutes=5)
    dataset.multi_tf_seq_len = 1
    dataset.per_tf_seq_lens = {"M5": 1}

    out = dataset._get_multi_tf_window(target)

    np.testing.assert_allclose(
        out["seq_m5"],
        np.full((1, MULTI_TF_FEATURE_COUNT_V2), 2.0, dtype=np.float32),
    )


def test_entry_v10_metadata_records_multi_tf_target_availability_shift() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")

    assert '"target_availability_shift_minutes"' in text
    assert '"closed_bar_target_availability"' in text


def test_entry_v10_trainer_verifies_mtf_cache_source_sha() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")

    assert "MULTI_TF_CACHE_SOURCE_SHA_MISMATCH" in text
    assert "m5_prebuilt_source_sha256" in text
    assert "MULTI_TF_CACHE_FEATURE_CONTRACT_MISMATCH" in text
    assert "MULTI_TF_CACHE_SHIFT_CONTRACT_MISMATCH" in text


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


def test_entry_v10_evidence_fusion_movement_proof_has_exact_six_parameter_schema() -> None:
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

    assert len(PARAMETER_SHAPES) == 6
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
    smoke = (repo / "scripts" / "run_entry_model_native_seq513_smoke_train.sh").read_text(encoding="utf-8")
    candidate = (repo / "scripts" / "run_entry_model_native_seq513_candidate_train.sh").read_text(encoding="utf-8")

    for text in (smoke, candidate):
        assert "--recipe-audit-json" in text
        assert "gx1.contracts.entry_model_native_train_launch_v1" in text
        assert "ENTRY_HIER_LEGACY_CE_MULT" not in text

    assert set(REQUIRED_POSITIVE_LOSS_WEIGHTS).issubset(MODEL_NATIVE_RECIPE_ENV_KEYS)
    assert "ENTRY_HIER_LEGACY_CE_MULT" not in MODEL_NATIVE_RECIPE_ENV_KEYS
    assert RECIPE_AUDIT_SCHEMA == "entry_model_native_seq513_train_recipe_audit_v1"


def test_entry_v10_standalone_eval_matches_training_objective() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")

    assert '"validation_objective_matches_train": True' in text
    assert "direction_only_ce_plus_residual_side_bias" not in text
    assert '"hierarchical_loss_metrics_included": False' not in text
