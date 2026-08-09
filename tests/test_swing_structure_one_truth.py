from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CTX_CONT_MICRO_FIELDS,
    MODEL_NATIVE_CTX_CONT_SESSION_FIELDS,
    MODEL_NATIVE_CTX_CONT_SOURCE_PREFIX_FIELDS,
    MODEL_NATIVE_CTX_CONT_SWING_FIELDS,
    MODEL_NATIVE_CTX_CONT_V1_PREFIX_FIELDS,
    MODEL_NATIVE_CTX_CONT_V2_EXTENSION_FIELDS,
    MODEL_NATIVE_CTX_CONT_V3_EXTENSION_FIELDS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_PREBUILT_CTX_CONT_FIELDS,
)
from gx1.features.micro_structure_v1 import (
    MICRO_FEATURE_NAMES_V1,
    compute_micro_structure_features,
)
from gx1.features.regime_v4_features import REGIME_V4_SOURCE_COLS
from gx1.features.swing_structure_v1 import (
    SWING_FEATURE_NAMES_V1,
    compute_swing_structure_features,
)
from gx1.features.volume_features import VOLUME_FEATURE_NAMES
from gx1.contracts.entry_model_native_state_v2 import (
    bucket_against_train_reference,
)
from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    _model_native_artifact_owner_fields,
)


ROOT = Path(__file__).resolve().parents[1]


def _ohlc() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close = np.array([100, 102, 104, 102, 101, 103, 105, 103, 102], dtype=np.float64)
    high = close + np.array([1, 1, 2, 1, 1, 1, 2, 1, 1], dtype=np.float64)
    low = close - 1.0
    return high, low, close


def test_swing_structure_is_causal_and_exact() -> None:
    high, low, close = _ohlc()
    observed = compute_swing_structure_features(high, low, close)
    assert tuple(observed) == SWING_FEATURE_NAMES_V1
    assert all(values.shape == close.shape for values in observed.values())
    assert all(np.isfinite(values).all() for values in observed.values())

    changed_high = high.copy()
    changed_low = low.copy()
    changed_close = close.copy()
    changed_high[-1] += 20.0
    changed_low[-1] -= 20.0
    changed_close[-1] += 5.0
    changed = compute_swing_structure_features(
        changed_high,
        changed_low,
        changed_close,
    )
    for name in SWING_FEATURE_NAMES_V1:
        np.testing.assert_array_equal(observed[name][:-1], changed[name][:-1])


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda h, _l, _c: h.__setitem__(2, np.nan), "NONFINITE"),
        (lambda h, _l, _c: h.__setitem__(2, 0.0), "NONPOSITIVE"),
        (
            lambda h, low_values, _c: low_values.__setitem__(2, h[2] + 1.0),
            "GEOMETRY",
        ),
    ],
)
def test_swing_structure_rejects_invalid_market_evidence(mutator, match: str) -> None:
    high, low, close = _ohlc()
    mutator(high, low, close)
    with pytest.raises(RuntimeError, match=match):
        compute_swing_structure_features(high, low, close)


def test_swing_structure_rejects_empty_or_invalid_parameters() -> None:
    with pytest.raises(RuntimeError, match="LENGTH"):
        compute_swing_structure_features([], [], [])
    high, low, close = _ohlc()
    with pytest.raises(RuntimeError, match="LOOKBACK"):
        compute_swing_structure_features(high, low, close, lookback=0)
    with pytest.raises(RuntimeError, match="ATR_PERIOD"):
        compute_swing_structure_features(high, low, close, atr_period=0)


def test_micro_structure_is_causal_exact_and_strict() -> None:
    high, low, close = _ohlc()
    observed = compute_micro_structure_features(high, low, close)
    assert tuple(observed) == MICRO_FEATURE_NAMES_V1
    assert all(np.isfinite(values).all() for values in observed.values())
    assert observed["micro_momentum_3"][:3].tolist() == [0.0, 0.0, 0.0]
    assert observed["micro_momentum_5"][:5].tolist() == [0.0] * 5
    # bps of the current close (repo ret_* convention), not raw USD diffs.
    np.testing.assert_allclose(
        observed["micro_momentum_3"][3:],
        (close[3:] - close[:-3]) / close[3:] * 1e4,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        observed["micro_acceleration"][2:],
        np.diff(np.diff(close)) / close[2:] * 1e4,
        rtol=1e-6,
    )
    # A zero-range bar has no close-location evidence: neutral 0.5, never the
    # fabricated "closed at high" 0.0.
    flat = compute_micro_structure_features(close.copy(), close.copy(), close.copy())
    assert flat["wick_ratio"].tolist() == [0.5] * len(close)

    changed_high = high.copy()
    changed_low = low.copy()
    changed_close = close.copy()
    changed_high[-1] += 20.0
    changed_low[-1] -= 20.0
    changed_close[-1] += 5.0
    changed = compute_micro_structure_features(
        changed_high,
        changed_low,
        changed_close,
    )
    for name in MICRO_FEATURE_NAMES_V1:
        np.testing.assert_array_equal(observed[name][:-1], changed[name][:-1])

    high[2] = np.nan
    with pytest.raises(RuntimeError, match="NONFINITE"):
        compute_micro_structure_features(high, low, close)


def test_entry_contract_is_the_only_context_subgroup_owner() -> None:
    assert MODEL_NATIVE_PREBUILT_CTX_CONT_FIELDS == (
        MODEL_NATIVE_CTX_CONT_SOURCE_PREFIX_FIELDS
        + MODEL_NATIVE_CTX_CONT_MICRO_FIELDS
        + MODEL_NATIVE_CTX_CONT_SWING_FIELDS
    )
    assert MODEL_NATIVE_CTX_CONT_V1_PREFIX_FIELDS == (
        MODEL_NATIVE_PREBUILT_CTX_CONT_FIELDS + MODEL_NATIVE_CTX_CONT_SESSION_FIELDS
    )
    assert len(MODEL_NATIVE_PREBUILT_CTX_CONT_FIELDS) == 16
    assert len(MODEL_NATIVE_CTX_CAT_FIELDS) == 5


def test_active_context_has_no_future_or_soft_pass_through() -> None:
    htf_owner = (ROOT / "gx1/features/htf_features.py").read_text(encoding="utf-8")
    augment_owner = (ROOT / "gx1/execution/v12_ctx_augment_live.py").read_text(
        encoding="utf-8"
    )
    enriched_owner = (
        ROOT / "gx1/scripts/build_entry_exit_m1_enriched_frame_v1.py"
    ).read_text(encoding="utf-8")
    builder = (
        ROOT / "gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py"
    ).read_text(encoding="utf-8")

    for source in (htf_owner, augment_owner, enriched_owner):
        assert "shift(-" not in source
        assert "ctx-cont-dim" not in source
        assert "cv3-cross-source" not in source
    assert 'suffixes=("", "_tape")' not in builder
    assert 'if "is_ASIA" not in df.columns' not in builder
    assert "src_supplied" not in builder
    assert "fall back to canonical_v2" not in builder


def test_builder_artifact_field_owners_are_exact_and_disjoint() -> None:
    cv2_owned, source_owned = _model_native_artifact_owner_fields(
        MODEL_NATIVE_BASE_FIELDS
    )
    assert set(cv2_owned) == (
        (set(MODEL_NATIVE_BASE_FIELDS) - set(VOLUME_FEATURE_NAMES))
        | set(MODEL_NATIVE_CTX_CONT_V2_EXTENSION_FIELDS)
    )
    assert set(source_owned) == (
        set(MODEL_NATIVE_CTX_CONT_V3_EXTENSION_FIELDS)
        | (set(REGIME_V4_SOURCE_COLS) - {"D1_dist_from_ema200_atr"})
        | {"volume"}
    )
    assert set(cv2_owned).isdisjoint(source_owned)


def test_prebuilt_rank_bucket_uses_explicit_reference_without_missing_fallback() -> None:
    reference = np.array([1.0, 2.0, 3.0])
    observed = bucket_against_train_reference(
        np.array([1.0, 2.0, 3.0]),
        reference,
    )
    assert observed.dtype == np.int64
    assert observed.tolist() == [1, 3, 4]
    with pytest.raises(RuntimeError, match="NONFINITE"):
        bucket_against_train_reference(np.array([1.0, np.nan]), reference)
