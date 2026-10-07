from __future__ import annotations

import pytest

from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_MTF_CONTEXT_TIMEFRAMES
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    model_native_context_temporal_alias_policy,
)
from gx1.features.htf_features import MULTI_TF_FEATURE_COUNT_V4
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
    batch_size = smoke.HARDWARE_SMOKE_BATCH_SIZE
    assert tuple(batch["seq_x"].shape) == (batch_size, 96, MODEL_NATIVE_SIGNAL_DIM)
    assert tuple(batch["snap_x"].shape) == (batch_size, MODEL_NATIVE_SIGNAL_DIM)
    assert tuple(batch["ctx_cont"].shape) == (batch_size, MODEL_NATIVE_CTX_CONT_DIM)
    assert tuple(batch["ctx_cat"].shape) == (batch_size, MODEL_NATIVE_CTX_CAT_DIM)
    for timeframe in ENTRY_MTF_CONTEXT_TIMEFRAMES:
        assert tuple(batch[f"seq_{timeframe.lower()}"].shape) == (
            batch_size, smoke._PER_TF_SEQ_LENS[timeframe], MULTI_TF_FEATURE_COUNT_V4
        )
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
