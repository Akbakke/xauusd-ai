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


def test_entry_foundation_train_wrappers_enable_path_quality_rank_recipe() -> None:
    repo = Path(__file__).resolve().parents[1]
    smoke = (repo / "scripts" / "run_entry_foundation_seq146_smoke_train.sh").read_text(encoding="utf-8")
    candidate = (repo / "scripts" / "run_entry_foundation_seq146_candidate_train.sh").read_text(encoding="utf-8")

    for text in (smoke, candidate):
        assert "ENTRY_PATH_QUALITY_RANK_WEIGHT" in text
        assert "ENTRY_PATH_QUALITY_RANK_MARGIN" in text
        assert "ENTRY_PATH_QUALITY_RANK_QUANTILE" in text
