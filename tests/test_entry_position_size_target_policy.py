from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.contracts.entry_direction_target_policy_v1 import (
    fit_entry_direction_target_policy,
)
from gx1.contracts.entry_position_size_target_policy_v1 import (
    entry_position_size_target_policy_contract,
    entry_position_size_targets_from_policy,
    fit_entry_position_size_target_policy,
    require_entry_position_size_target_manifest_binding,
    require_entry_position_size_target_policy,
)
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    _masked_position_size_mse,
)


def _fit_frame() -> pd.DataFrame:
    time = pd.date_range("2020-01-01T00:00:00Z", periods=1500, freq="5min")
    phase = np.arange(len(time), dtype=np.float64)
    mid = (
        1800.0
        + 1.8 * np.sin(phase / 7.0)
        + 0.9 * np.sin(phase / 29.0)
        + phase * 0.0002
    )
    return pd.DataFrame(
        {"time": time, "bid_close": mid - 0.05, "ask_close": mid + 0.05}
    )


def _fit(
    frame: pd.DataFrame,
    artifact_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    direction = fit_entry_direction_target_policy(
        closed_m5=frame,
        train_start=frame["time"].iloc[0],
        train_end=frame["time"].iloc[1100] + pd.Timedelta(minutes=5),
        source_parquet_sha256="1" * 64,
        tape_provenance_sha256="2" * 64,
    )
    sizing = fit_entry_position_size_target_policy(
        closed_m5=frame,
        entry_direction_target_policy=direction,
        source_parquet_sha256="1" * 64,
        tape_provenance_sha256="2" * 64,
        ecdf_artifact_path=artifact_dir / "position_size_ecdf.npy",
    )
    return direction, sizing


def test_position_size_policy_is_exact_train_only_and_hash_bound(
    tmp_path: Path,
) -> None:
    frame = _fit_frame()
    direction, baseline = _fit(frame, tmp_path)
    assert _fit(frame.copy(), tmp_path)[1] == baseline

    mutated = frame.copy()
    mutated.loc[1200:, "bid_close"] += 10.0
    mutated.loc[1200:, "ask_close"] += 10.0
    assert _fit(mutated, tmp_path)[1] == baseline
    assert baseline["fit_split"] == "train"
    assert baseline["val_test_rows_used_for_fit"] == 0
    assert baseline["entry_direction_target_policy_sha256"] == direction[
        "policy_sha256"
    ]
    assert baseline["fit_population_rows"] == (
        baseline["fit_long_rows"] + baseline["fit_short_rows"]
    )
    assert baseline["ecdf_artifact_rows"] == baseline["fit_population_rows"]
    assert Path(baseline["ecdf_artifact_path"]).is_file()


def test_position_size_policy_applies_frozen_ecdf_only_on_selected_trade_rows(
    tmp_path: Path,
) -> None:
    _, policy = _fit(_fit_frame(), tmp_path)
    result = entry_position_size_targets_from_policy(
        policy=policy,
        mfe_first_n_bps=[1.0, 5.0, 15.0, 0.0],
        mae_first_n_bps=[8.0, 5.0, 1.0, 0.0],
        selected_side=[0, 1, 0, -1],
        trade_mask=[1, 1, 1, 0],
    )
    assert result["mask"].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert result["target"][0] <= result["target"][1] <= result["target"][2]
    assert result["target"][3] == 0.0
    assert np.all((result["target"] >= 0.0) & (result["target"] <= 1.0))

    with pytest.raises(RuntimeError, match="SIDE_MASK_MISMATCH"):
        entry_position_size_targets_from_policy(
            policy=policy,
            mfe_first_n_bps=[1.0],
            mae_first_n_bps=[0.0],
            selected_side=[0],
            trade_mask=[0],
        )


def test_position_size_policy_rejects_hash_and_ecdf_tampering(
    tmp_path: Path,
) -> None:
    _, policy = _fit(_fit_frame(), tmp_path)
    broken_hash = deepcopy(policy)
    broken_hash["fit_scope"] = "VAL"
    with pytest.raises(RuntimeError, match="HASH_INVALID"):
        require_entry_position_size_target_policy(broken_hash)

    artifact_path = Path(policy["ecdf_artifact_path"])
    artifact_path.write_bytes(artifact_path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="ECDF_ARTIFACT_INVALID"):
        require_entry_position_size_target_policy(policy)


def test_position_size_manifest_binding_checks_projection_and_train_window(
    tmp_path: Path,
) -> None:
    direction, policy = _fit(_fit_frame(), tmp_path)
    extra = entry_position_size_target_policy_contract(policy)
    bound = require_entry_position_size_target_manifest_binding(
        extra,
        expected_source_parquet_sha256=direction["source_parquet_sha256"],
        expected_tape_provenance_sha256=direction["tape_provenance_sha256"],
        expected_direction_policy_sha256=direction["policy_sha256"],
        expected_train_start=direction["train_start_utc"],
        expected_train_end=direction["train_end_utc"],
    )
    assert bound == policy

    tampered = deepcopy(extra)
    tampered["position_size_target_unmasked_training_allowed"] = True
    with pytest.raises(RuntimeError, match="MANIFEST_PROJECTION_MISMATCH"):
        require_entry_position_size_target_manifest_binding(tampered)
    with pytest.raises(RuntimeError, match="TRAIN_END_MISMATCH"):
        require_entry_position_size_target_manifest_binding(
            extra,
            expected_train_end=pd.Timestamp(direction["train_end_utc"])
            + pd.Timedelta(minutes=5),
        )


def test_pretest_target_audit_does_not_conflate_m5_and_m1_tape_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gx1.scripts import audit_entry_foundation_targets_v1 as target_audit

    split_manifest = tmp_path / "pretest_split.manifest.json"
    split_manifest.write_text(
        json.dumps(
            {
                "splits": {"train": {"start": "2021-06-01", "end": "2025-06-01"}},
                "extra": {
                    "pretest_only": True,
                    "source_frame": {"parquet_sha256": "1" * 64},
                    "xau_tape_provenance": {"authority": "direct_m5"},
                    "unified_exit_lifecycle": {"m1_source_sha256": "2" * 64},
                    "entry_causal_m1_position_size_target_policy": {},
                },
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def _capture(extra: object, **kwargs: object) -> dict[str, object]:
        del extra
        captured.update(kwargs)
        return captured

    monkeypatch.setattr(
        target_audit,
        "require_causal_m1_position_size_target_manifest_binding",
        _capture,
    )
    result = target_audit._entry_position_size_policy_from_split_manifest(
        split_manifest,
        direction_policy={"policy_sha256": "3" * 64},
    )
    assert result["expected_tape_provenance_sha256"] is None
    assert result["expected_m1_source_sha256"] == "2" * 64
    assert result["expected_source_parquet_sha256"] == "1" * 64


def test_position_size_training_loss_has_zero_gradient_outside_policy_mask() -> None:
    logits = torch.tensor([[0.0], [0.0], [0.0]], requires_grad=True)
    target = torch.tensor([0.1, 0.9, 0.0])
    mask = torch.tensor([1.0, 1.0, 0.0])
    loss = _masked_position_size_mse(logits, target, mask)
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad[0]) != 0.0
    assert float(logits.grad[1]) != 0.0
    assert float(logits.grad[2]) == 0.0

    all_flat_logits = torch.tensor([[1.0], [-1.0]], requires_grad=True)
    flat_loss = _masked_position_size_mse(
        all_flat_logits,
        torch.zeros(2),
        torch.zeros(2),
    )
    flat_loss.backward()
    assert float(flat_loss) == 0.0
    assert torch.equal(all_flat_logits.grad, torch.zeros_like(all_flat_logits))

    with pytest.raises(RuntimeError, match="MASK_INVALID"):
        _masked_position_size_mse(
            torch.zeros((2, 1)),
            torch.zeros(2),
            torch.tensor([1.0, 0.25]),
        )


def test_builder_has_no_static_position_size_sigmoid_or_atr_multiplier() -> None:
    from gx1.scripts import build_entry_v10_ctx_training_dataset_v3 as builder
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer

    source = Path(builder.__file__).read_text(encoding="utf-8")
    assert "signed_edge_atr" not in source
    assert "atr * 2.0" not in source
    assert "out[mask == 0.0] = 0.5" not in source
    assert "entry_position_size_targets_from_policy" in source
    contract_source = Path(
        require_entry_position_size_target_policy.__code__.co_filename
    ).read_text(encoding="utf-8")
    assert "sorted_train_path_evidence_bps" not in contract_source
    assert "ecdf_artifact_sha256" in contract_source
    trainer_source = Path(trainer.__file__).read_text(encoding="utf-8")
    assert trainer_source.count("_masked_position_size_mse(") == 3
    assert trainer_source.count(
        'target_names=("y_position_size_target", "y_position_size_mask")'
    ) == 2
    assert '_active_head_batch_target(\n                        batch, "y_position_size_mask"' in trainer_source
