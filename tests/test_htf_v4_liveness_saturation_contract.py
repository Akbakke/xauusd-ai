"""Strict V4 value liveness with explicit source-defined observation masks.

``geomline_above_active_count`` is used purely as a representative emitted
column to force constancy on; it replaced ``geomline_above_active``, retired
2026-08-15 for being the ">= 1" indicator of that very count.
"""
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
        values[:, htf.MULTI_TF_PER_BAR_FEATURES_V4.index("close_range_observed")] = np.arange(len(index)) % 2
        values[:, htf.MULTI_TF_PER_BAR_FEATURES_V4.index("session_change_flag")] = 0.0 if timeframe == "D1" else np.arange(len(values)) % 2
        values[:, htf.MULTI_TF_PER_BAR_FEATURES_V4.index("close_distance_below_high_range_fraction")] = np.where(
            values[:, htf.MULTI_TF_PER_BAR_FEATURES_V4.index("close_range_observed")] == 1.0,
            (np.arange(len(values)) + 1.0) / (len(values) + 1.0), 0.0)
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
    field = "geomline_above_active_count"
    _set_constant(frames, "D1", field, value)
    payload = htf.build_multi_tf_v4_liveness_contract(frames)
    assert payload["decision"] == "FAIL"
    assert field in payload["timeframes"]["D1"]["constant_fields"]
    with pytest.raises(RuntimeError, match="DECISION_INVALID"):
        htf.require_multi_tf_v4_liveness_contract(payload)


def test_validator_rejects_forged_constant_stats() -> None:
    payload = htf.build_multi_tf_v4_liveness_contract(_frames())
    stats = payload["timeframes"]["D1"]["fields"]["geomline_above_active_count"]
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


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_exact_observation_mask_can_be_constant_without_claiming_market_variation(value: float) -> None:
    frames = _frames()
    _set_constant(frames, "D1", "close_range_observed", value)
    payload = htf.build_multi_tf_v4_liveness_contract(frames)
    assert payload["decision"] == "PASS"
    stats = payload["timeframes"]["D1"]["fields"]["close_range_observed"]
    assert stats["unique_count"] == 1 and stats["std"] == 0.0
    assert stats["minimum"] == stats["maximum"] == value
    htf.require_multi_tf_v4_liveness_contract(payload)


def test_observation_mask_rejects_nonbinary_values_and_forged_constant_stats() -> None:
    frames = _frames()
    _set_constant(frames, "D1", "close_range_observed", 0.5)
    payload = htf.build_multi_tf_v4_liveness_contract(frames)
    assert payload["decision"] == "FAIL"
    assert any("invalid_observation_mask" in failure for failure in payload["failures"])
    frames = _frames()
    _set_constant(frames, "D1", "close_range_observed", 1.0)
    payload = htf.build_multi_tf_v4_liveness_contract(frames)
    payload["timeframes"]["D1"]["fields"]["close_range_observed"]["std"] = 0.25
    _reseal(payload)
    with pytest.raises(RuntimeError, match="OBSERVATION_MASK_STATS_INVALID"):
        htf.require_multi_tf_v4_liveness_contract(payload)


def test_d1_session_event_is_source_constant_zero_not_a_missing_or_dense_exemption():
    frames = _frames()
    payload = htf.build_multi_tf_v4_liveness_contract(frames)
    assert payload["decision"] == "PASS"
    stats = payload["timeframes"]["D1"]["fields"]["session_change_flag"]
    assert stats["minimum"] == stats["maximum"] == stats["mean"] == stats["std"] == 0.0
    assert stats["unique_count"] == 1
    htf.require_multi_tf_v4_liveness_contract(payload)
    _set_constant(frames, "D1", "session_change_flag", 1.0)
    bad = htf.build_multi_tf_v4_liveness_contract(frames)
    assert bad["decision"] == "FAIL"
    assert any("invalid_source_constant_binary_event" in failure for failure in bad["failures"])
