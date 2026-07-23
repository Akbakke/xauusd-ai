from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_model_native_readiness_v1 import MODEL_NATIVE_ACTIVE_HEADS
from gx1.contracts.entry_model_native_training_objective_v1 import (
    REQUIRED_POSITIVE_LOSS_WEIGHTS,
)
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.models.entry_v10.direction_decision_contract import (
    model_direction_decision_contract_metadata,
)
from tests.entry_v10_trainer_dataset_support import (
    aux_head_target_contract,
    install_multi_tf_stub,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields


TRAINER_PATH = Path("gx1/models/entry_v10/entry_v10_ctx_train_v3.py")


def _valid_active_target_frame(rows: int = 3) -> pd.DataFrame:
    values: dict[str, np.ndarray] = {
        name: np.zeros(rows, dtype=np.float64)
        for name in trainer._MODEL_NATIVE_ACTIVE_TARGET_COLS
    }
    values["y_direction"] = np.resize(np.array([0.0, 1.0, 2.0]), rows)
    values["y_side"] = np.resize(np.array([0.0, 1.0, 0.0]), rows)
    for name in trainer._MODEL_NATIVE_UNIT_INTERVAL_TARGET_COLS:
        values[name] = np.full(rows, 0.5, dtype=np.float64)
    return pd.DataFrame(values)


@pytest.mark.parametrize("missing_target", trainer._MODEL_NATIVE_ACTIVE_TARGET_COLS)
def test_every_missing_model_native_active_target_fails(missing_target: str) -> None:
    frame = _valid_active_target_frame().drop(columns=[missing_target])

    failures = trainer._model_native_active_target_failures("train", frame)

    assert failures
    assert missing_target in " | ".join(failures)


def test_model_native_active_target_contract_rejects_nonfinite() -> None:
    frame = _valid_active_target_frame()
    frame.loc[0, "y_forecast_ret_K1"] = np.nan

    failures = trainer._model_native_active_target_failures("train", frame)

    assert any("y_forecast_ret_K1 contains non-finite" in item for item in failures)


def test_model_native_active_target_contract_accepts_signed_spread_aware_mfe() -> None:
    frame = _valid_active_target_frame()
    frame.loc[0, "mfe_first_n_bps"] = -4.8103

    failures = trainer._model_native_active_target_failures("train", frame)

    assert failures == []


@pytest.mark.parametrize(
    "mae_target",
    ("y_long_expected_mae_bps", "y_short_expected_mae_bps"),
)
def test_model_native_active_target_contract_rejects_negative_mae_magnitude(
    mae_target: str,
) -> None:
    frame = _valid_active_target_frame()
    frame.loc[0, mae_target] = -0.01

    failures = trainer._model_native_active_target_failures("train", frame)

    assert any(f"{mae_target} contains negative values" in item for item in failures)


def test_aux_path_regression_preserves_signed_forward_outcome_targets() -> None:
    values = torch.tensor([-10.0, 20.0, -30.0])
    positive_mask = torch.tensor([True, True, False])

    scaled = trainer._signed_scaled_aux_regression_target(
        values,
        positive_mask,
        20.0,
    )

    torch.testing.assert_close(scaled, torch.tensor([-0.5, 1.0]))


def test_model_native_architecture_has_no_head_enable_config_surface() -> None:
    import inspect

    run_train_parameters = inspect.signature(trainer.run_train).parameters
    forbidden = [name for name in run_train_parameters if name.startswith("enable_")]
    assert forbidden == []
    source = TRAINER_PATH.read_text(encoding="utf-8")
    assert "BooleanOptionalAction" not in source
    assert "--prelim-no-aux-heads" not in source
    assert "--allow-constant-labels" not in source
    assert "--smoke-date-from" not in source
    assert "def set_seed(" not in source
    assert "def set_thread_limits(" not in source
    assert "train_entry_v10_ctx_depth_ladder.py" not in source


def _valid_loss_weights() -> dict[str, float]:
    return {
        name: 1.0
        for name in REQUIRED_POSITIVE_LOSS_WEIGHTS
    }


@pytest.mark.parametrize(
    "weight_name",
    REQUIRED_POSITIVE_LOSS_WEIGHTS,
)
def test_every_missing_model_native_active_loss_weight_fails(
    weight_name: str,
) -> None:
    weights = _valid_loss_weights()
    del weights[weight_name]

    failures = trainer._model_native_active_loss_weight_failures(weights)

    assert f"{weight_name}=missing" in failures


@pytest.mark.parametrize(
    "bad_value,expected",
    [
        (0.0, "expected >0"),
        (-0.1, "expected >0"),
        (float("nan"), "non-finite"),
        (float("inf"), "non-finite"),
        ("not-a-number", "non-numeric"),
    ],
)
def test_model_native_active_loss_weights_reject_soft_pass_throughs(
    bad_value: object,
    expected: str,
) -> None:
    weights: dict[str, object] = _valid_loss_weights()
    weight_name = REQUIRED_POSITIVE_LOSS_WEIGHTS[0]
    weights[weight_name] = bad_value

    failures = trainer._model_native_active_loss_weight_failures(weights)

    assert any(weight_name in item and expected in item for item in failures)


def test_model_native_active_loss_weight_contract_accepts_only_positive_surface() -> None:
    assert trainer._model_native_active_loss_weight_failures(_valid_loss_weights()) == []


def test_model_native_active_head_names_are_the_exact_enabled_surface() -> None:
    heads = trainer._build_active_head_names()

    assert heads == list(MODEL_NATIVE_ACTIVE_HEADS)


def _valid_output_heads(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(batch_size, width)
        for name, width in trainer._MODEL_NATIVE_ACTIVE_OUTPUT_WIDTHS.items()
    }


@pytest.mark.parametrize("missing_head", trainer._MODEL_NATIVE_ACTIVE_OUTPUT_WIDTHS)
def test_every_missing_model_native_output_head_fails(missing_head: str) -> None:
    outputs = _valid_output_heads()
    del outputs[missing_head]

    failures = trainer._model_native_active_output_head_failures(outputs)

    assert any(missing_head in item for item in failures)


def _valid_dip_forecast_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(batch_size)
        for name in trainer._DIP_FORECAST_TARGET_COLS
    }


def test_dip_forecast_loss_unconditionally_rejects_missing_head_and_target() -> None:
    outputs = _valid_output_heads()
    batch = _valid_dip_forecast_batch()

    missing_head = dict(outputs)
    del missing_head["timing_pred"]
    with pytest.raises(RuntimeError, match="ACTIVE_HEAD_MISSING.*timing_pred"):
        trainer.dip_forecast_loss(
            missing_head,
            batch,
            torch.device("cpu"),
        )

    missing_target = dict(batch)
    del missing_target["y_vol_fwd_K96"]
    with pytest.raises(RuntimeError, match="ACTIVE_HEAD_TARGET_MISSING.*y_vol_fwd_K96"):
        trainer.dip_forecast_loss(
            outputs,
            missing_target,
            torch.device("cpu"),
        )


def test_direction_decision_contract_export_is_canonical_and_split_brain_safe() -> None:
    canonical = model_direction_decision_contract_metadata()
    lock = {"direction_decision_contract": canonical}
    meta = {"direction_decision_contract": canonical}

    assert trainer._direction_decision_contract_export_failures(lock, meta) == []

    broken_meta = {"direction_decision_contract": {**canonical, "selection_mode": "broken"}}
    failures = trainer._direction_decision_contract_export_failures(lock, broken_meta)
    assert any("bundle_metadata" in item for item in failures)
    assert any("split-brain" in item for item in failures)


def _write_model_native_dataset(path: Path, *, missing_target: str | None = None) -> None:
    rows = 3
    seq_len = 2
    seq = [
        [[float(row + step + col) for col in range(MODEL_NATIVE_SIGNAL_DIM)] for step in range(seq_len)]
        for row in range(rows)
    ]
    snap = [
        [float(row + col) for col in range(MODEL_NATIVE_SIGNAL_DIM)]
        for row in range(rows)
    ]
    columns: dict[str, pa.Array] = {
        "time": pa.array([f"2026-01-0{row + 1}T00:00:00Z" for row in range(rows)]),
        "seq": pa.array(seq, type=pa.list_(pa.list_(pa.float32()))),
        "snap": pa.array(snap, type=pa.list_(pa.float32())),
        "ctx_cont": pa.array(
            [[float(row + col) for col in range(MODEL_NATIVE_CTX_CONT_DIM)] for row in range(rows)],
            type=pa.list_(pa.float32()),
        ),
        "ctx_cat": pa.array(
            [[int((row + col) % 3) for col in range(MODEL_NATIVE_CTX_CAT_DIM)] for row in range(rows)],
            type=pa.list_(pa.int64()),
        ),
        "mae_first_n_bps": pa.array([0.0, 0.0, 0.0]),
        "y_early_move": pa.array([0.0, 0.0, 0.0]),
        "y_quality_score": pa.array([0.0, 0.0, 0.0]),
    }
    frame = _valid_active_target_frame(rows)
    for name in trainer._MODEL_NATIVE_ACTIVE_TARGET_COLS:
        if name == missing_target:
            continue
        values = frame[name].to_numpy()
        columns[name] = pa.array(values, type=pa.float64())
    pq.write_table(pa.table(columns), path)

    contract = model_native_signal_contract_metadata(
        canonical_model_native_selected_fields(
            remainder_prefix="session_regime.target_contract_fixture"
        )
    )
    manifest = {
        "extra": {
            "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
            "model_native_signal_contract": contract,
            "aux_head_target_contract": aux_head_target_contract(),
            "signal_bridge": {
                "fields": contract["fields"],
                "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
            },
        }
    }
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_model_native_dataset_fails_before_training_when_active_target_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet = tmp_path / "xau_model_native_train.parquet"
    _write_model_native_dataset(parquet, missing_target="y_vol_fwd_K96")
    m5_path = install_multi_tf_stub(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="MODEL_NATIVE_ACTIVE_TARGET_CONTRACT_INVALID.*y_vol_fwd_K96"):
        trainer.EntryV10CtxDataset(
            parquet,
            seq_len=2,
            m5_prebuilt_path=m5_path,
            multi_tf_seq_len=2,
        )


def test_model_native_dataset_getitem_reads_exact_targets_without_hold_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet = tmp_path / "xau_model_native_train.parquet"
    _write_model_native_dataset(parquet)
    m5_path = install_multi_tf_stub(tmp_path, monkeypatch)

    dataset = trainer.EntryV10CtxDataset(
        parquet,
        seq_len=2,
        m5_prebuilt_path=m5_path,
        multi_tf_seq_len=2,
    )
    sample = dataset[0]

    for name in trainer._MODEL_NATIVE_ACTIVE_TARGET_COLS:
        if name != "y_direction":
            assert name in sample
    assert "y_hold_horizon_target" not in sample


def test_model_native_getitem_branch_has_no_target_default_or_alias() -> None:
    source = TRAINER_PATH.read_text(encoding="utf-8")
    strict_branch = source.split("for target_name in _MODEL_NATIVE_ACTIVE_TARGET_COLS:", 1)[1].split(
        "            mtf = self._get_multi_tf_window", 1
    )[0]

    assert "row[target_name]" in strict_branch
    assert "row.get(" not in strict_branch
    assert "y_hold_horizon_target" not in strict_branch
    assert source.count('"direction_decision_contract": direction_decision_contract') == 2


def test_position_size_is_mandatory_and_has_no_trainer_disable_api() -> None:
    import inspect

    assert "enable_position_size_head" not in inspect.signature(trainer.run_train).parameters
    assert "require_all_heads" not in inspect.signature(trainer.dip_forecast_loss).parameters
    assert "strict_model_native_heads" not in inspect.signature(trainer.train_epoch).parameters
    assert "strict_model_native_heads" not in inspect.signature(trainer.validate).parameters
    source = TRAINER_PATH.read_text(encoding="utf-8")
    assert "strict_model_native_heads" not in source
    assert "require_all_heads" not in source
    assert "if position_size_logit is not None" not in source
    assert "if tf_agreement_logit is not None" not in source
    heads = trainer._build_active_head_names()
    assert "position_size" in heads


def test_every_fused_evidence_head_is_required_during_train_and_validation() -> None:
    source = TRAINER_PATH.read_text(encoding="utf-8")

    for retired_soft_branch in (
        'path_pred = out.get("path_quality")',
        'mfe_pred = out.get("mfe_first_n")',
        'tradable_logit = out.get("tradable_logit")',
        'bad_path_logit = out.get("bad_path_logit")',
        'clean_edge_logit = out.get("clean_edge_logit")',
        'survival_logit = out.get("survival_logit")',
        'path_log_var = out.get("path_quality_log_var")',
        "if path_log_var is not None",
        "if trade_logit is not None",
        "if side_logits is not None",
        "else probs[:, 0]",
    ):
        assert retired_soft_branch not in source

    with pytest.raises(RuntimeError, match="ACTIVE_HEAD_MISSING.*specialist_gate"):
        trainer._specialist_gate_regularization({}, torch.device("cpu"))


def test_entry_trainer_has_no_stale_warm_start_artifact_lane() -> None:
    source = TRAINER_PATH.read_text(encoding="utf-8")
    assert "--init-from-state-dict" not in source
    assert "init_from_state_dict" not in source
    assert '"lane_contract"' not in source
    assert "entry_admission_policy" not in source
    assert "OVERLAP_LONG_REPLACES" not in source
    assert not Path("gx1/scripts/warm_start_v10_v2_from_v1.py").exists()
