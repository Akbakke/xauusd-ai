from __future__ import annotations

import pytest

from gx1.features.entry_specialist_feature_groups_v1 import (
    model_native_context_temporal_alias_policy,
)
from gx1.scripts import attended_model_native_hardware_smoke_v1 as smoke


def test_hardware_smoke_builds_exact_shape_contract_without_reading_market_data() -> None:
    normalization, samples = smoke._synthetic_normalization()
    batch = smoke._batch(samples)

    assert normalization["lineage"]["dataset_run_id"] == (
        "ATTENDED_HARDWARE_SMOKE_NO_DATA_AUTHORITY_V1"
    )
    assert normalization["lineage"]["train_parquet_path"].startswith(
        "/attended-hardware-smoke/"
    )
    # 2026-09-20 (D-3/A1/B9 wave, same rebuild generation as the D-1/D-2
    # step that moved 238 -> 244 / 176 -> 182): the signal dim moves
    # 244 -> 240 and the per-TF lane width 182 -> 178 (-3 squeeze
    # exact-derivatives, -1 unsigned level break age per lane).  The literals
    # are the drift guard, not the source — the owners are
    # entry_model_native_signal_v1.MODEL_NATIVE_SIGNAL_DIM and
    # htf_features.MULTI_TF_FEATURE_COUNT_V4.
    assert tuple(batch["seq_x"].shape) == (8, 96, 241)
    assert tuple(batch["snap_x"].shape) == (8, 241)
    assert tuple(batch["ctx_cont"].shape) == (8, 71)
    assert tuple(batch["ctx_cat"].shape) == (8, 1)
    assert tuple(batch["seq_m15"].shape) == (8, 64, 190)
    assert tuple(batch["seq_h1"].shape) == (8, 96, 190)
    assert tuple(batch["seq_h4"].shape) == (8, 96, 190)
    assert tuple(batch["seq_d1"].shape) == (8, 252, 190)
    assert (batch["seq_x"][:, -1, :] == batch["snap_x"]).all()
    for alias in model_native_context_temporal_alias_policy(smoke._signal_names())["aliases"]:
        assert (
            batch["snap_x"][:, int(alias["signal_index"])]
            == batch["ctx_cont"][:, int(alias["ctx_cont_index"])]
        ).all()


def test_hardware_smoke_parser_refuses_non_cuda_or_missing_marker() -> None:
    parser = smoke.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--device", "cuda", "--specialist-audit-json", "/x"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--attended-hardware-smoke",
                "--device",
                "cpu",
                "--specialist-audit-json",
                "/x",
            ]
        )
