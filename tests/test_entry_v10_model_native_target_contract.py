from __future__ import annotations

import inspect
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
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_exit_production_architecture_v1 import (
    PRODUCTION_MTF_PER_TF_WINDOW_BARS,
)
from gx1.contracts.entry_model_native_readiness_v1 import MODEL_NATIVE_ACTIVE_HEADS
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
    """Rule 16: spread-aware MFE and path quality stay SIGNED; MAE is adverse.

    The retired ``_signed_aux_regression_target`` masking helper is gone —
    the auxiliary heads now consume their forward outcomes raw.  The domain
    contract is therefore proved on the owner tuples and on the loss source:
    no MFE column may be declared non-negative, every MAE/vol column must be,
    and the loss owner may not clip, abs() or nan_to_num a target.
    """
    nonnegative = set(trainer._MODEL_NATIVE_NONNEGATIVE_TARGET_COLS)
    signed = [
        name
        for name in trainer._DIP_TARGET_COLS + trainer._FORECAST_TARGET_COLS
        if "_mfe_" in name or name.startswith("y_forecast_ret_")
    ]
    assert signed
    assert nonnegative.isdisjoint(signed)
    adverse = [
        name
        for name in trainer._MODEL_NATIVE_ACTIVE_TARGET_COLS
        if "_mae_" in name or name.endswith("_expected_mae_bps")
    ]
    assert adverse
    assert set(adverse) <= nonnegative

    loss_source = inspect.getsource(trainer.dip_forecast_task_losses)
    for forbidden in ("clamp(", "abs()", "torch.abs", "nan_to_num", "clip("):
        assert forbidden not in loss_source

    # A signed MFE row is admitted; a negative MAE row is not.
    frame = _valid_active_target_frame()
    frame.loc[0, "y_dip_mfe_long_K12"] = -12.5
    assert trainer._model_native_active_target_failures("train", frame) == []
    frame.loc[0, "y_dip_mae_long_K12"] = -0.01
    assert any(
        "y_dip_mae_long_K12 contains negative values" in item
        for item in trainer._model_native_active_target_failures("train", frame)
    )


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


def test_model_native_active_head_names_include_unified_exit_surface() -> None:
    heads = trainer._build_active_head_names()

    assert heads == [*MODEL_NATIVE_ACTIVE_HEADS, "unified_exit"]


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


def test_dip_forecast_task_losses_unconditionally_reject_missing_head_and_target() -> None:
    outputs = _valid_output_heads()
    batch = _valid_dip_forecast_batch()

    missing_head = dict(outputs)
    del missing_head["timing_pred"]
    with pytest.raises(RuntimeError, match="ACTIVE_HEAD_MISSING.*timing_pred"):
        trainer.dip_forecast_task_losses(
            missing_head,
            batch,
            torch.device("cpu"),
        )

    missing_target = dict(batch)
    del missing_target["y_vol_fwd_K96"]
    with pytest.raises(RuntimeError, match="ACTIVE_HEAD_TARGET_MISSING.*y_vol_fwd_K96"):
        trainer.dip_forecast_task_losses(
            outputs,
            missing_target,
            torch.device("cpu"),
        )


def test_forward_bps_heads_use_raw_native_units() -> None:
    batch = {
        name: torch.full((2,), 40.0)
        for name in trainer._DIP_FORECAST_TARGET_COLS
    }
    for name in trainer._TIMING_TARGET_COLS:
        batch[name] = torch.full((2,), 0.5)
    outputs = {
        "dip_pred": torch.full((2, 18), 40.0),
        "forecast_pred": torch.full((2, 4), 40.0),
        "timing_pred": torch.full((2, 12), 0.5),
        "tail_risk_pred": torch.full((2, 6), 40.0),
        "vol_forecast_pred": torch.full((2, 3), 40.0),
    }

    losses = trainer.dip_forecast_task_losses(
        outputs,
        batch,
        torch.device("cpu"),
    )

    assert set(losses) == {
        "dip_bps",
        "forecast_return_bps",
        "dip_timing_fraction",
        "tail_risk_bps",
        "forward_volatility_bps",
    }
    assert all(float(loss.item()) == pytest.approx(0.0) for loss in losses.values())


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
    seq_len = MODEL_NATIVE_SEQ_LEN
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
            seq_len=MODEL_NATIVE_SEQ_LEN,
            m5_prebuilt_path=m5_path,
            per_tf_seq_lens=dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS),
            multi_tf_closed_bar=True,
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
        seq_len=MODEL_NATIVE_SEQ_LEN,
        m5_prebuilt_path=m5_path,
        per_tf_seq_lens=dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS),
        multi_tf_closed_bar=True,
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
    assert "require_all_heads" not in inspect.signature(trainer.dip_forecast_task_losses).parameters
    assert "strict_model_native_heads" not in inspect.signature(trainer.train_epoch).parameters
    assert "strict_model_native_heads" not in inspect.signature(trainer.validate).parameters
    source = TRAINER_PATH.read_text(encoding="utf-8")
    assert "strict_model_native_heads" not in source
    assert "require_all_heads" not in source
    assert "if position_size_logit is not None" not in source
    heads = trainer._build_active_head_names()
    assert "position_size" in heads
    assert "tf_agreement" not in heads
    # Sizing has no direction authority: the position-size output is never an
    # Entry action-value component (rule 5).
    assert "position_size" in trainer._ACTIVE_HEAD_ACTION_AUTHORITY_NONE
    assert "position_size_logit" not in dict(
        trainer._ACTIVE_HEAD_OUTPUT_COMPONENTS
    )["entry_action_q"]


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

    assert not hasattr(trainer, "_specialist_gate_regularization")


def test_cooperation_gate_epoch_health_requires_every_gate_family_live() -> None:
    accumulator = trainer._new_cooperation_gate_epoch_accumulator()
    uniform = {
        name: torch.full((4, width), 1.0 / float(width), dtype=torch.float32)
        for name, width in trainer._MODEL_NATIVE_COOPERATION_GATE_WIDTHS.items()
    }
    trainer._accumulate_cooperation_gate_epoch(accumulator, uniform)
    stats = trainer._finalize_cooperation_gate_epoch(accumulator)
    feature_accumulator = trainer._new_feature_tf_gate_epoch_accumulator()
    feature_gate = torch.stack(
        [
            torch.full(
                trainer._MODEL_NATIVE_FEATURE_TF_GATE_SHAPE,
                0.95 + 0.03 * row,
                dtype=torch.float32,
            )
            for row in range(4)
        ]
    )
    trainer._accumulate_feature_tf_gate_epoch(
        feature_accumulator,
        {"family_tf_feature_gate": feature_gate},
    )
    stats.update(
        trainer._finalize_feature_tf_gate_epoch(feature_accumulator)
    )

    assert trainer._cooperation_gate_health_failures(stats) == []
    assert stats["specialist_gate_rows"] == 4
    assert stats["specialist_gate_min_mean"] == pytest.approx(1.0 / 8.0)
    assert stats["tf_gate_min_mean"] == pytest.approx(
        1.0 / trainer._MODEL_NATIVE_COOPERATION_GATE_WIDTHS["tf_gate"]
    )
    assert stats["family_tf_cooperation_gate_min_mean"] == pytest.approx(
        1.0
        / trainer._MODEL_NATIVE_COOPERATION_GATE_WIDTHS[
            "family_tf_cooperation_gate"
        ]
    )


def test_unified_exit_gate_health_covers_all_five_timeframes_and_features() -> None:
    cooperation = trainer._new_cooperation_gate_epoch_accumulator(
        trainer._UNIFIED_EXIT_COOPERATION_GATE_WIDTHS
    )
    features = trainer._new_feature_tf_gate_epoch_accumulator(
        trainer._UNIFIED_EXIT_FEATURE_TF_GATE_SHAPE
    )
    rows = 4
    exit_out = {
        f"exit_{name}": torch.full(
            (rows, width),
            1.0 / float(width),
            dtype=torch.float32,
        )
        for name, width in trainer._UNIFIED_EXIT_COOPERATION_GATE_WIDTHS.items()
    }
    exit_out["exit_family_tf_feature_gate"] = torch.stack(
        [
            torch.full(
                trainer._UNIFIED_EXIT_FEATURE_TF_GATE_SHAPE,
                0.95 + 0.03 * row,
                dtype=torch.float32,
            )
            for row in range(rows)
        ]
    )
    gate_view = trainer._unified_exit_gate_view(exit_out)
    trainer._accumulate_cooperation_gate_epoch(
        cooperation,
        gate_view,
        gate_widths=trainer._UNIFIED_EXIT_COOPERATION_GATE_WIDTHS,
    )
    trainer._accumulate_feature_tf_gate_epoch(
        features,
        gate_view,
        gate_shape=trainer._UNIFIED_EXIT_FEATURE_TF_GATE_SHAPE,
    )
    stats, failures = trainer._finalize_unified_exit_gate_epoch(
        cooperation,
        features,
    )

    assert failures == []
    assert stats["exit_cooperation_gate_health_ok"] is True
    assert stats["exit_tf_gate_rows"] == rows
    assert len(stats["exit_tf_gate_mean_weight"]) == 5
    assert len(stats["exit_family_tf_cooperation_gate_mean_weight"]) == 40
    assert len(stats["exit_family_tf_feature_gate_std_weight"]) == (
        5 * trainer.MULTI_TF_FEATURE_COUNT_V4
    )
    # Ownership proof: the gate-view owner lives in the trainer and its
    # verdict is bound into checkpoint admission.
    assert trainer._unified_exit_gate_view.__module__ == trainer.__name__
    source = TRAINER_PATH.read_text(encoding="utf-8")
    assert "_unified_exit_gate_view(" in source
    assert "exit_cooperation_gate_health_ok=(" in source


def test_technical_preflight_only_journals_static_open_exit_feature_gates() -> None:
    static_failure = "family_tf_feature_gate constant/dead indices=[7, 9]"
    blocking, diagnostic = trainer._technical_preflight_exit_gate_disposition(
        [static_failure]
    )

    assert blocking == []
    assert diagnostic is not None
    assert diagnostic["decision"] == "WARN_STATIC_BUT_OPEN_GATE"
    assert diagnostic["candidate_gate_health_remains_strict"] is True
    assert diagnostic["failures"] == [static_failure]

    blocking, diagnostic = trainer._technical_preflight_exit_gate_disposition(
        [static_failure, "family_tf_feature_gate saturated indices=[7]"]
    )
    assert blocking == ["family_tf_feature_gate saturated indices=[7]"]
    assert diagnostic is not None
    assert diagnostic["blocking_failures_after_disposition"] == blocking


def test_cooperation_gate_epoch_health_uses_empirical_liveness_not_target_share() -> None:
    accumulator = trainer._new_cooperation_gate_epoch_accumulator()
    out = {
        name: torch.full((3, width), 1.0 / float(width), dtype=torch.float32)
        for name, width in trainer._MODEL_NATIVE_COOPERATION_GATE_WIDTHS.items()
    }
    starved = torch.full((3, 8), (1.0 - 0.001) / 7.0, dtype=torch.float32)
    starved[:, 0] = 0.001
    out["specialist_gate"] = starved
    trainer._accumulate_cooperation_gate_epoch(accumulator, out)
    stats = trainer._finalize_cooperation_gate_epoch(accumulator)
    feature_accumulator = trainer._new_feature_tf_gate_epoch_accumulator()
    feature_gate = torch.stack(
        [
            torch.full(
                trainer._MODEL_NATIVE_FEATURE_TF_GATE_SHAPE,
                0.95 + 0.04 * row,
                dtype=torch.float32,
            )
            for row in range(3)
        ]
    )
    trainer._accumulate_feature_tf_gate_epoch(
        feature_accumulator,
        {"family_tf_feature_gate": feature_gate},
    )
    stats.update(
        trainer._finalize_feature_tf_gate_epoch(feature_accumulator)
    )

    # A small but genuinely observed routing share is live.  Admission must not
    # encode a hand-written target distribution for learned gate weights.
    assert trainer._cooperation_gate_health_failures(stats) == []

    exactly_starved = dict(stats)
    exactly_starved["specialist_gate_mean_weight"] = [
        0.0,
        *([1.0 / 7.0] * 7),
    ]
    exactly_starved["specialist_gate_min_mean"] = 0.0
    exactly_starved_failures = trainer._cooperation_gate_health_failures(
        exactly_starved
    )
    assert any(
        "specialist_gate min mean=0.000000" in failure
        for failure in exactly_starved_failures
    )
    # Exact starvation must block candidate checkpoint admission.
    # At smoke it stays a logged diagnostic by user vedtak 2026-07-25; the
    # complete profile matrix lives in
    # tests/test_entry_profile_separated_checkpoint_admission.py.
    assert (
        trainer._checkpoint_admission_ok(
            profile="candidate",
            active_head_health_ok=True,
            cooperation_gate_health_ok=False,
            exit_cooperation_gate_health_ok=True,
        )
        is False
    )


def test_entry_trainer_has_no_stale_warm_start_artifact_lane() -> None:
    source = TRAINER_PATH.read_text(encoding="utf-8")
    assert "--init-from-state-dict" not in source
    assert "init_from_state_dict" not in source
    assert '"lane_contract"' not in source
    assert "entry_admission_policy" not in source
    assert "OVERLAP_LONG_REPLACES" not in source
    assert not Path("gx1/scripts/warm_start_v10_v2_from_v1.py").exists()
