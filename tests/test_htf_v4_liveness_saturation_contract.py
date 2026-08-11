"""Contract tests for the presence-mask saturation rule in the V4 liveness owner.

The rule (htf_features.HTF_V4_PRESENCE_MASK_SATURATION_CONTRACT, measured
2026-08-11 on real declared D1 tape) admits a constant column ONLY as a
saturated occupancy mask: exact value 1.0, every sibling attribute
non-constant on the same TF, paired touch event firing. Constant 0.0, dead
siblings, silent events and every non-mask constant field remain RED.

All frames here are synthetic (rule 2c): they prove the code contract only.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import gx1.features.htf_features as htf

_MASK_ABOVE = "geomline_above_active"
_MASK_BELOW = "geomline_below_active"
_EVENT_BELOW = "geomline_touch_below"
_SIBLING_BELOW = "geomline_below_touch_count"


def _frames() -> dict[str, pd.DataFrame]:
    source_index = pd.date_range(
        "2026-01-01T00:00:00Z", periods=8 * 288, freq="5min"
    )
    expected_indices = htf.build_multi_tf_v4_closed_timestamp_indices(
        source_index
    )
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
        frame.attrs["feats_np"] = frame.to_numpy(
            dtype=np.float32, copy=False
        )
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


def test_saturated_masks_pass_and_are_recorded_then_revalidated() -> None:
    frames = _frames()
    _set_constant(frames, "D1", _MASK_ABOVE, 1.0)
    _set_constant(frames, "D1", _MASK_BELOW, 1.0)
    payload = htf.build_multi_tf_v4_liveness_contract(frames)
    assert payload["decision"] == "PASS"
    assert payload["failures"] == []
    d1 = payload["timeframes"]["D1"]
    assert d1["constant_fields"] == []
    assert d1["saturated_presence_masks"] == [_MASK_ABOVE, _MASK_BELOW]
    for timeframe in ("M5", "M15", "H1", "H4"):
        assert payload["timeframes"][timeframe]["saturated_presence_masks"] == []
    assert (
        payload["schema_version"]
        == htf.HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION
    )
    # The strict validator re-proves the claim from the payload's own stats.
    htf.require_multi_tf_v4_liveness_contract(payload)


def test_constant_zero_mask_stays_red() -> None:
    frames = _frames()
    _set_constant(frames, "D1", _MASK_BELOW, 0.0)
    payload = htf.build_multi_tf_v4_liveness_contract(frames)
    assert payload["decision"] == "FAIL"
    assert _MASK_BELOW in payload["timeframes"]["D1"]["constant_fields"]
    assert payload["timeframes"]["D1"]["saturated_presence_masks"] == []


def test_saturated_mask_with_dead_sibling_stays_red() -> None:
    frames = _frames()
    _set_constant(frames, "D1", _MASK_BELOW, 1.0)
    _set_constant(frames, "D1", _SIBLING_BELOW, 3.0)
    payload = htf.build_multi_tf_v4_liveness_contract(frames)
    assert payload["decision"] == "FAIL"
    constant = payload["timeframes"]["D1"]["constant_fields"]
    assert _MASK_BELOW in constant
    assert _SIBLING_BELOW in constant


def test_saturated_mask_with_silent_event_stays_red() -> None:
    frames = _frames()
    _set_constant(frames, "D1", _MASK_BELOW, 1.0)
    _set_constant(frames, "D1", _EVENT_BELOW, 0.0)
    payload = htf.build_multi_tf_v4_liveness_contract(frames)
    assert payload["decision"] == "FAIL"
    assert _MASK_BELOW in payload["timeframes"]["D1"]["constant_fields"]


def test_non_mask_constant_field_stays_red() -> None:
    frames = _frames()
    _set_constant(frames, "H4", "geomchan_active", 1.0)
    payload = htf.build_multi_tf_v4_liveness_contract(frames)
    assert payload["decision"] == "FAIL"
    assert (
        "geomchan_active"
        in payload["timeframes"]["H4"]["constant_fields"]
    )


def test_validator_rejects_forged_saturation_claim() -> None:
    frames = _frames()
    payload = htf.build_multi_tf_v4_liveness_contract(frames)
    assert payload["decision"] == "PASS"
    payload["timeframes"]["D1"]["saturated_presence_masks"] = [_MASK_ABOVE]
    _reseal(payload)
    with pytest.raises(RuntimeError, match="SATURATION_CLAIM_INVALID"):
        htf.require_multi_tf_v4_liveness_contract(payload)


def test_validator_rejects_unknown_mask_in_claim() -> None:
    frames = _frames()
    _set_constant(frames, "D1", _MASK_ABOVE, 1.0)
    _set_constant(frames, "D1", _MASK_BELOW, 1.0)
    payload = htf.build_multi_tf_v4_liveness_contract(frames)
    payload["timeframes"]["D1"]["saturated_presence_masks"] = [
        _MASK_ABOVE,
        _MASK_BELOW,
        "geomchan_active",
    ]
    _reseal(payload)
    with pytest.raises(RuntimeError, match="SATURATION_CLAIM_INVALID"):
        htf.require_multi_tf_v4_liveness_contract(payload)


def test_validator_accepts_immutable_v2_payload_exactly() -> None:
    frames = _frames()
    payload = htf.build_multi_tf_v4_liveness_contract(frames)
    assert payload["decision"] == "PASS"
    payload["schema_version"] = htf.HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION_V2
    for row in payload["timeframes"].values():
        del row["saturated_presence_masks"]
    _reseal(payload)
    htf.require_multi_tf_v4_liveness_contract(payload)


def test_validator_rejects_v2_payload_with_saturation_key() -> None:
    frames = _frames()
    payload = htf.build_multi_tf_v4_liveness_contract(frames)
    payload["schema_version"] = htf.HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION_V2
    _reseal(payload)
    with pytest.raises(RuntimeError, match="TF_KEYS_INVALID"):
        htf.require_multi_tf_v4_liveness_contract(payload)
