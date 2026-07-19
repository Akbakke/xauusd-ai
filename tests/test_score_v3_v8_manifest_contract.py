from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gx1.exits.contracts.exit_io_v8_regime_m1l512 import (
    EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT,
    EXIT_IO_V8_REGIME_M1L512_FEATURE_NAMES_HASH,
    EXIT_IO_V8_REGIME_M1L512_IO_VERSION,
)
from gx1.scripts.score_v3_v8_on_per_bar_v1 import (
    _model_contract_manifest_fields,
)


def test_scored_manifest_uses_loaded_v8_model_contract(tmp_path) -> None:
    (tmp_path / "transformer_config.json").write_text(
        json.dumps(
            {
                "exit_ml_io_version": EXIT_IO_V8_REGIME_M1L512_IO_VERSION,
                "input_dim": EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT,
                "window_len": 512,
            }
        ),
        encoding="utf-8",
    )
    model = SimpleNamespace(
        input_dim=EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT,
        window_len=512,
    )

    fields = _model_contract_manifest_fields(tmp_path, model)

    assert fields == {
        "v3_exit_io_version": EXIT_IO_V8_REGIME_M1L512_IO_VERSION,
        "v3_v8_input_dim": EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT,
        "v3_v8_window_len": 512,
        "v3_feature_names_hash": EXIT_IO_V8_REGIME_M1L512_FEATURE_NAMES_HASH,
    }


def test_scored_manifest_rejects_model_config_dimension_drift(tmp_path) -> None:
    (tmp_path / "transformer_config.json").write_text(
        json.dumps(
            {
                "exit_ml_io_version": EXIT_IO_V8_REGIME_M1L512_IO_VERSION,
                "input_dim": EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT,
                "window_len": 512,
            }
        ),
        encoding="utf-8",
    )
    model = SimpleNamespace(
        input_dim=EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT - 1,
        window_len=512,
    )

    with pytest.raises(RuntimeError, match="does not match"):
        _model_contract_manifest_fields(tmp_path, model)
