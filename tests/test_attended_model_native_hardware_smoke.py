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
    assert tuple(batch["seq_x"].shape) == (8, 96, 238)
    assert tuple(batch["snap_x"].shape) == (8, 238)
    assert tuple(batch["ctx_cont"].shape) == (8, 71)
    assert tuple(batch["ctx_cat"].shape) == (8, 1)
    assert tuple(batch["seq_m15"].shape) == (8, 64, 176)
    assert tuple(batch["seq_h1"].shape) == (8, 96, 176)
    assert tuple(batch["seq_h4"].shape) == (8, 96, 176)
    assert tuple(batch["seq_d1"].shape) == (8, 252, 176)
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
