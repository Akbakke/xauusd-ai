from __future__ import annotations

import numpy as np
import pandas as pd

from gx1.audit.entry_transformer_feature_audit import DERIVED_CANDIDATE_NAMES
from gx1.audit.entry_transformer_feature_diagnostics import (
    _batch_feature_matrix,
    _mtf_current_matrix,
    _predict_proba,
    _selected_indices,
)
from gx1.contracts.signal_bridge_v3 import (
    ORDERED_CTX_CAT_NAMES_V3,
    ORDERED_CTX_CONT_NAMES_V3,
    ORDERED_SEQ_FIELDS_V3,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V2, MULTI_TF_SHIFT


def test_selected_indices_is_deterministic_sorted_and_optional() -> None:
    assert _selected_indices(10, 0, 7) is None
    assert _selected_indices(10, 10, 7) is None

    a = _selected_indices(100, 10, 7)
    b = _selected_indices(100, 10, 7)
    assert a is not None and b is not None
    np.testing.assert_array_equal(a, b)
    assert np.all(a[:-1] < a[1:])


def test_mtf_current_matrix_uses_closed_bar_shift() -> None:
    target = pd.Timestamp("2026-06-26T12:00:00Z")
    cutoff = target.value - int(MULTI_TF_SHIFT["M5"].value)
    ts = np.array([cutoff - 60_000_000_000, cutoff + 60_000_000_000], dtype=np.int64)
    feats = np.zeros((2, len(MULTI_TF_PER_BAR_FEATURES_V2)), dtype=np.float32)
    feats[0, :] = 1.0
    feats[1, :] = 2.0
    df = pd.DataFrame(index=pd.DatetimeIndex(ts.astype("datetime64[ns]"), tz="UTC"))
    df.attrs["ts_int64"] = ts
    df.attrs["feats_np"] = feats

    out, specs = _mtf_current_matrix(np.array([target.value], dtype=np.int64), {"M5": df})

    assert out.shape == (1, len(MULTI_TF_PER_BAR_FEATURES_V2))
    np.testing.assert_allclose(out[0], 1.0)
    assert specs[0].name.startswith("mtf_m5__")
    assert all(s.active_contract for s in specs)


def test_batch_feature_matrix_covers_active_and_derived_surfaces() -> None:
    seq = np.zeros((2, 96, len(ORDERED_SEQ_FIELDS_V3)), dtype=np.float32)
    snap = np.zeros((2, len(ORDERED_SEQ_FIELDS_V3)), dtype=np.float32)
    ctx = np.zeros((2, len(ORDERED_CTX_CONT_NAMES_V3)), dtype=np.float32)
    cat = np.zeros((2, len(ORDERED_CTX_CAT_NAMES_V3)), dtype=np.int64)
    pdf = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-06-26T12:00:00Z", "2026-06-26T12:05:00Z"], utc=True),
            "seq": list(seq),
            "snap": list(snap),
            "ctx_cont": list(ctx),
            "ctx_cat": list(cat),
            "y_direction": [0, 2],
            "label_horizon_bars": [3, 3],
            "y_forecast_ret_K1": [1.0, -1.0],
        }
    )

    x, y, specs, targets = _batch_feature_matrix(
        pdf,
        include_seq_summary=True,
        include_derived=True,
        mtf_cache=None,
    )

    expected_cols = (
        len(ORDERED_SEQ_FIELDS_V3)
        + 4 * len(ORDERED_SEQ_FIELDS_V3)
        + len(ORDERED_CTX_CONT_NAMES_V3)
        + len(ORDERED_CTX_CAT_NAMES_V3)
        + len(DERIVED_CANDIDATE_NAMES)
    )
    assert x.shape == (2, expected_cols)
    np.testing.assert_array_equal(y, [0, 2])
    assert len(specs) == expected_cols
    assert "label_horizon_bars" in targets
    assert "y_forecast_ret_K1" in targets
    assert any(s.source == "derived_candidate" and not s.active_contract for s in specs)
    assert any(s.source == "seq_mean12" and s.active_contract for s in specs)


def test_predict_proba_normalizes_rows() -> None:
    class BadProbModel:
        def predict_proba(self, _x):
            return np.array([[0.2, 0.2, 0.2], [0.0, 0.0, 0.0]], dtype=np.float64)

    out = _predict_proba(BadProbModel(), np.zeros((2, 1), dtype=np.float32))

    np.testing.assert_allclose(out.sum(axis=1), 1.0)
    np.testing.assert_allclose(out[0], [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(out[1], [1 / 3, 1 / 3, 1 / 3])
