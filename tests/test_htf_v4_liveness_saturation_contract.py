"""Strict V4 liveness: no constant-field or saturation exception exists."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import gx1.features.htf_features as htf


def _frames() -> dict[str, pd.DataFrame]:
    source_index = pd.date_range(
        "2026-01-01T00:00:00Z", periods=8 * 288, freq="5min"
    )
    expected_indices = htf.build_multi_tf_v4_closed_timestamp_indices(source_index)
    width = htf.MULTI_TF_FEATURE_COUNT_V4
    frames: dict[str, pd.DataFrame] = {}
    for offset, timeframe in enumerate(htf.MULTI_TF_RESAMPLE_RULES):
        index = expected_indices[timeframe]
        rows = np.arange(len(index), dtype=np.float32).reshape(-1, 1)
        columns = np.arange(width, dtype=np.float32).reshape(1, -1)
        values = rows * (columns + 1.0) + np.float32(offset + 1)
        frame = pd.DataFrame(
            values,
            index=index,
            columns=htf.MULTI_TF_PER_BAR_FEATURES_V4,
            copy=False,
        )
        frame.attrs["feats_np"] = frame.to_numpy(dtype=np.float32, copy=False)
        frame.attrs["ts_int64"] = index.asi8.astype(np.int64, copy=True)
        frame.attrs["causal_warmup_rows"] = 0
        frame.attrs["htf_feature_contract"] = htf.HTF_V4_MATRIX_CONTRACT
        frames[timeframe] = frame
    return frames


def _set_constant(
    frames: dict[str, pd.DataFrame], tf: str, field: str, value: float
) -> None:
    index = htf.MULTI_TF_PER_BAR_FEATURES_V4.index(field)
    frames[tf].attrs["feats_np"][:, index] = np.float32(value)


def _reseal(payload: dict) -> dict:
    identity = {
        key: item for key, item in payload.items() if key != "contract_sha256"
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return payload


def test_variable_surface_passes_without_saturation_schema() -> None:
    payload = htf.build_multi_tf_v4_liveness_contract(_frames())
    assert payload["decision"] == "PASS"
    assert payload["failures"] == []
    for row in payload["timeframes"].values():
        assert "saturated_presence_masks" not in row
        assert row["constant_fields"] == []
    htf.require_multi_tf_v4_liveness_contract(payload)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_every_constant_numeric_field_is_red(value: float) -> None:
    frames = _frames()
    field = "geomline_above_active"
    _set_constant(frames, "D1", field, value)
    payload = htf.build_multi_tf_v4_liveness_contract(frames)
    assert payload["decision"] == "FAIL"
    assert field in payload["timeframes"]["D1"]["constant_fields"]
    with pytest.raises(RuntimeError, match="DECISION_INVALID"):
        htf.require_multi_tf_v4_liveness_contract(payload)


def test_validator_rejects_forged_constant_stats() -> None:
    payload = htf.build_multi_tf_v4_liveness_contract(_frames())
    stats = payload["timeframes"]["D1"]["fields"]["geomline_above_active"]
    stats["unique_count"] = 1
    stats["std"] = 0.0
    _reseal(payload)
    with pytest.raises(RuntimeError, match="STATS_INVALID"):
        htf.require_multi_tf_v4_liveness_contract(payload)


def test_validator_rejects_retired_saturation_key() -> None:
    payload = htf.build_multi_tf_v4_liveness_contract(_frames())
    payload["timeframes"]["D1"]["saturated_presence_masks"] = []
    _reseal(payload)
    with pytest.raises(RuntimeError, match="TF_KEYS_INVALID"):
        htf.require_multi_tf_v4_liveness_contract(payload)
