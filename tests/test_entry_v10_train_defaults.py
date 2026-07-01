from __future__ import annotations

import ast
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


def test_entry_v10_bad_path_aux_default_is_parked() -> None:
    env_defaults = _env_str_defaults(_trainer_ast())
    assert env_defaults["ENTRY_AUX_BAD_PATH_WEIGHT"] == "0.0"
    assert env_defaults["ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT"] == "0.0"
    assert env_defaults["ENTRY_PATH_QUALITY_RANK_WEIGHT"] == "0.0"


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
        y_teacher_winner_long=torch.zeros(3),
        y_teacher_bad_long=torch.zeros(3),
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


def test_entry_v10_train_refuses_to_write_bundle_when_best_class_balance_guard_failed() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")

    assert "[TRAIN_FAIL_DIRECTION_CLASS_BALANCE_GUARD]" in text
    assert "_direction_ckpt_balance_guard_required()" in text
    assert "refusing to write a collapsed direction bundle" in text


def test_entry_foundation_train_wrappers_enable_path_quality_rank_recipe() -> None:
    repo = Path(__file__).resolve().parents[1]
    smoke = (repo / "scripts" / "run_entry_foundation_seq146_smoke_train.sh").read_text(encoding="utf-8")
    candidate = (repo / "scripts" / "run_entry_foundation_seq146_candidate_train.sh").read_text(encoding="utf-8")

    for text in (smoke, candidate):
        assert "ENTRY_PATH_QUALITY_RANK_WEIGHT" in text
        assert "ENTRY_PATH_QUALITY_RANK_MARGIN" in text
        assert "ENTRY_PATH_QUALITY_RANK_QUANTILE" in text
