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
    assert covered["direction_slice_failure_count"] == 0
    assert trainer._direction_slice_ckpt_score(0.40, collapsed) < trainer._direction_slice_ckpt_score(0.40, covered)


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
    assert text.count("_direction_aux_ce_loss(out[\"mtf_dir_logits\"], y, criterion, ce_sample_weight)") >= 2


def test_entry_v10_multi_tf_window_uses_m5_close_availability_for_closed_bar() -> None:
    import numpy as np
    import pandas as pd

    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    target = pd.Timestamp("2026-07-08T18:00:00Z")
    ts = np.array(
        [
            (target - pd.Timedelta(minutes=5)).value,
            target.value,
        ],
        dtype=np.int64,
    )
    feats = np.array([[1.0], [2.0]], dtype=np.float32)
    frame = pd.DataFrame(index=pd.DatetimeIndex(ts.astype("datetime64[ns]"), tz="UTC"))
    frame.attrs["ts_int64"] = ts
    frame.attrs["feats_np"] = feats

    dataset = trainer.EntryV10CtxDataset.__new__(trainer.EntryV10CtxDataset)
    dataset._multi_tf_feats = {"M5": frame}
    dataset._multi_tf_shift = {"M5": pd.Timedelta(minutes=5)}
    dataset._multi_tf_target_availability_shift = pd.Timedelta(minutes=5)
    dataset.multi_tf_seq_len = 1
    dataset.per_tf_seq_lens = {"M5": 1}

    out = dataset._get_multi_tf_window(target)

    np.testing.assert_allclose(out["seq_m5"], [[2.0]])


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
    assert "enable_side_validity_head=true requires" in text


def test_entry_v10_xau_direction_repair_requires_xau_sources() -> None:
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    failures = trainer._xau_direction_repair_source_failures(
        {
            "train_parquet": "/data/foreign_fx/v10_dataset_20260710_train.parquet",
            "val_parquet": "/data/generic/v10_dataset_val.parquet",
            "m5_prebuilt_path": "",
        }
    )

    text = "\n".join(failures)
    assert "XAU-specific" in text
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
                "neutral_xgb_bridge": True,
                "xgb_bridge_source": "neutral_uniform_proba",
                "tape_root": "/home/andre2/GX1_DATA/data/oanda/canonical/foreign_fx_m5_bid_ask__CANONICAL",
            }
        ),
        encoding="utf-8",
    )

    failures = trainer._xau_direction_repair_manifest_failures({"train": parquet})

    assert any("XAUUSD tape_root" in item for item in failures)


def test_entry_v10_xau_direction_repair_target_contract_rejects_wrong_side_rows() -> None:
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
            "y_position_size_target": [1.0],
            "mfe_long_first_n_bps": [8.0],
            "mae_long_first_n_bps": [2.0],
            "mfe_short_first_n_bps": [0.0],
            "mae_short_first_n_bps": [1.0],
            "y_long_path_utility_bps": [5.0],
            "y_short_path_utility_bps": [6.0],
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
    text = "\n".join(failures)

    assert "still labeled SHORT" in text
    assert "still teach SHORT" in text
    assert "SHORT utility >= LONG utility" in text


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
            "y_long_path_utility_bps": [-358.47857666015625],
            "y_short_path_utility_bps": [float(path_quality)],
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


def test_entry_v10_standalone_eval_declares_direction_only_loss_scope() -> None:
    text = TRAINER_PATH.read_text(encoding="utf-8")

    assert '"test_loss_scope": "direction_only_ce_plus_residual_side_bias"' in text
    assert '"validation_objective_matches_train": False' in text
    assert '"hierarchical_loss_metrics_included": False' in text
