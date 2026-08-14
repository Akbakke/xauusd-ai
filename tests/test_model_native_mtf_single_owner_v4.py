from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
)
from gx1.features.htf_features import (
    HTF_V4_MATRIX_CONTRACT,
    MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4,
    MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4,
    MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4,
    MULTI_TF_PER_BAR_FEATURES_V4,
    attach_model_native_mtf_scalars_v4,
    model_native_mtf_owner_marker_v4,
    project_model_native_mtf_scalars_v4,
    require_model_native_mtf_scalar_owner_v4,
)


_TF_FIXTURES = {
    "M5": ("5min", 5_000, 0.0),
    "M15": ("15min", 2_000, 10_000.0),
    "H1": ("1h", 600, 20_000.0),
    "H4": ("4h", 150, 30_000.0),
    "D1": ("1D", 30, 40_000.0),
}


def _verified_v4_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for timeframe, (freq, rows, offset) in _TF_FIXTURES.items():
        index = pd.date_range(
            "2025-12-20T00:00:00Z",
            periods=rows,
            freq=freq,
        )
        core = np.zeros(
            (rows, len(MULTI_TF_PER_BAR_FEATURES_V4)),
            dtype=np.float32,
        )
        frame = pd.DataFrame(
            core,
            index=index,
            columns=MULTI_TF_PER_BAR_FEATURES_V4,
            copy=False,
        )
        frame.attrs["ts_int64"] = index.asi8.astype(np.int64, copy=True)
        frame.attrs["feats_np"] = frame.to_numpy(dtype=np.float32, copy=False)
        frame.attrs["causal_warmup_rows"] = 0
        frame.attrs["htf_feature_contract"] = HTF_V4_MATRIX_CONTRACT

        scalar_fields = MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4[
            timeframe
        ]
        scalar = np.empty((rows, len(scalar_fields)), dtype=np.float32)
        for column, name in enumerate(scalar_fields):
            scalar[:, column] = (
                offset
                + column * 1_000.0
                + np.arange(rows, dtype=np.float32)
            )
        frame.attrs["model_native_mtf_scalar_fields_v4"] = scalar_fields
        frame.attrs["model_native_mtf_scalars_np_v4"] = scalar
        frame.attrs["model_native_mtf_scalar_warmup_rows_v4"] = 0
        frame.attrs["model_native_mtf_scalar_contract_v4"] = (
            MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4
        )
        frames[timeframe] = frame
    return frames


def test_model_native_mtf_fields_are_raw_continuous_and_ordered_once() -> None:
    continuous = tuple(
        name
        for name in MODEL_NATIVE_CTX_CONT_FIELDS
        if name in MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4
    )
    expected_continuous = MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4

    assert continuous == expected_continuous
    assert MODEL_NATIVE_CTX_CAT_FIELDS == ("session_id",)
    assert len(MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4) == 25
    assert len(set(MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4)) == 25


def test_m5_and_m1_projection_share_the_same_closed_v4_state() -> None:
    frames = _verified_v4_frames()
    require_model_native_mtf_scalar_owner_v4(frames)
    entry_times = pd.date_range(
        "2026-01-01T12:35:00Z",
        periods=5,
        freq="5min",
    )
    exit_times = pd.date_range(
        "2026-01-01T12:55:00Z",
        periods=5,
        freq="min",
    )

    entry = project_model_native_mtf_scalars_v4(
        frames,
        entry_times.asi8,
        decision_bar_duration=pd.Timedelta(minutes=5),
    )
    exit_ = project_model_native_mtf_scalars_v4(
        frames,
        exit_times.asi8,
        decision_bar_duration=pd.Timedelta(minutes=1),
    )

    for name in MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4:
        assert np.isfinite(entry[name]).all(), name
        assert np.isfinite(exit_[name]).all(), name
        assert float(entry[name][-1]) == float(exit_[name][-1]), name

    # V30 package 2 (2026-08-13): the SCALAR route carries M5 on both clocks —
    # the momentum-G3 `m5_rsi14_canon_v2` ctx scalar lives on the M5 frame, and
    # its projection onto the M5 decision clock is the identity (cutoff =
    # t + 5min - MULTI_TF_SHIFT["M5"] = t).  The windowed per-TF SEQUENCE route
    # is a different contract and still M15/H1/H4/D1 for Entry
    # (ENTRY_MTF_CONTEXT_TIMEFRAMES, entry_exit_feature_base_v1).
    assert model_native_mtf_owner_marker_v4(
        decision_bar_duration=pd.Timedelta(minutes=5)
    )["route_timeframes"] == ["M5", "M15", "H1", "H4", "D1"]
    assert model_native_mtf_owner_marker_v4(
        decision_bar_duration=pd.Timedelta(minutes=1)
    )["route_timeframes"] == ["M5", "M15", "H1", "H4", "D1"]
    from gx1.contracts.entry_exit_feature_base_v1 import (
        ENTRY_MTF_CONTEXT_TIMEFRAMES,
    )

    assert ENTRY_MTF_CONTEXT_TIMEFRAMES == ("M15", "H1", "H4", "D1")


def test_v4_attachment_rejects_any_preexisting_mtf_field_owner() -> None:
    rows = 70
    index = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=rows,
        freq="5min",
    )
    close = 2_000.0 + np.arange(rows, dtype=np.float64) * 0.01
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.full(rows, 10.0),
            "_v1h1_atr": np.ones(rows),
        },
        index=index,
    )

    with pytest.raises(RuntimeError, match="DUPLICATE_MTF_OWNER"):
        attach_model_native_mtf_scalars_v4(
            frame,
            multi_tf=_verified_v4_frames(),
            decision_bar_duration=pd.Timedelta(minutes=5),
        )
