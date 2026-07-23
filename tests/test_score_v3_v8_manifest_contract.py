from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gx1.exits.contracts.exit_io_v8_regime_m1l512 import (
    EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT,
    EXIT_IO_V8_REGIME_M1L512_FEATURE_NAMES_HASH,
    EXIT_IO_V8_REGIME_M1L512_IO_VERSION,
)
from gx1.scripts.score_v3_v8_on_per_bar_v1 import (
    PER_BAR_COL_BY_V6,
    TRADE_STATE_FEATURE_NAMES_V6,
    _model_contract_manifest_fields,
    build_overlay_for_trade,
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


def _overlay_frame() -> pd.DataFrame:
    rows = []
    for bar_idx, base in ((2, 200.0), (1, 100.0)):
        row = {
            "bar_idx_v1": bar_idx,
            "bar_ts_ns_v1": bar_idx * 60_000_000_000,
            "bars_in_trade_v1": bar_idx,
            "entry_fill_ts_ns_v1": 0,
        }
        for feature_idx, column in enumerate(PER_BAR_COL_BY_V6.values()):
            row[column] = base + feature_idx
        rows.append(row)
    return pd.DataFrame(rows)


def test_trade_overlay_is_exact_sorted_and_never_zero_fills() -> None:
    frame = _overlay_frame()

    overlay, sorted_rows = build_overlay_for_trade(frame)

    assert sorted_rows["bar_idx_v1"].tolist() == [1, 2]
    assert overlay.shape == (2, len(TRADE_STATE_FEATURE_NAMES_V6))
    assert overlay[0].tolist() == pytest.approx(
        [
            float(sorted_rows.iloc[0][PER_BAR_COL_BY_V6[name]])
            for name in TRADE_STATE_FEATURE_NAMES_V6
        ]
    )
    assert overlay[1].tolist() == pytest.approx(
        [
            float(sorted_rows.iloc[1][PER_BAR_COL_BY_V6[name]])
            for name in TRADE_STATE_FEATURE_NAMES_V6
        ]
    )

    missing = frame.drop(columns=[next(iter(PER_BAR_COL_BY_V6.values()))])
    with pytest.raises(RuntimeError, match="OVERLAY_FIELDS_MISSING"):
        build_overlay_for_trade(missing)

    nonfinite = frame.copy()
    nonfinite.loc[0, next(iter(PER_BAR_COL_BY_V6.values()))] = np.nan
    with pytest.raises(RuntimeError, match="OVERLAY_NONFINITE"):
        build_overlay_for_trade(nonfinite)
