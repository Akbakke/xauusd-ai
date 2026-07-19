from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.features.tf_agreement_score import (
    TF_AGREEMENT_SOURCE_FIELDS,
    compute_tf_agreement_score,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "D1_dist_from_ema200_atr": [1.0, -1.0, 0.0],
            "H4_trend_sign_cat": [2, 0, 1],
            "H1_range_compression_ratio": [1.1, 0.9, 1.0],
            "M15_range_compression_ratio": [1.1, 0.9, 1.0],
            "micro_momentum_3": [1.0, -1.0, 0.0],
        }
    )


def test_tf_agreement_score_uses_exact_multitimeframe_signs() -> None:
    score = compute_tf_agreement_score(_frame())

    np.testing.assert_array_equal(score.to_numpy(), np.ones(3, dtype=np.float32))

    conflicted = _frame()
    conflicted.loc[0, "H4_trend_sign_cat"] = 0
    conflicted.loc[0, "micro_momentum_3"] = -1.0
    observed = compute_tf_agreement_score(conflicted)
    assert observed.iloc[0] == 0.5
    np.testing.assert_array_equal(observed.iloc[1:].to_numpy(), np.ones(2, dtype=np.float32))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("D1_dist_from_ema200_atr", np.nan, "TF_AGREEMENT_SOURCE_NONFINITE"),
        ("micro_momentum_3", np.inf, "TF_AGREEMENT_SOURCE_NONFINITE"),
        ("H4_trend_sign_cat", 1.5, "TF_AGREEMENT_H4_CATEGORY_NONINTEGER"),
        ("H4_trend_sign_cat", 3, "TF_AGREEMENT_H4_CATEGORY_INVALID"),
        ("H1_range_compression_ratio", "bad", "TF_AGREEMENT_SOURCE_NOT_NUMERIC"),
    ],
)
def test_tf_agreement_score_rejects_invalid_sources(
    field: str,
    value: object,
    message: str,
) -> None:
    frame = _frame()
    if isinstance(value, str) or (field == "H4_trend_sign_cat" and isinstance(value, float)):
        frame[field] = frame[field].astype(object)
    frame.loc[1, field] = value
    with pytest.raises(RuntimeError, match=message):
        compute_tf_agreement_score(frame)


def test_tf_agreement_score_rejects_missing_duplicate_and_empty_sources() -> None:
    for field in TF_AGREEMENT_SOURCE_FIELDS:
        with pytest.raises(RuntimeError, match=f"TF_AGREEMENT_SOURCE_MISSING: {field}"):
            compute_tf_agreement_score(_frame().drop(columns=[field]))

    duplicate = _frame()
    duplicate.insert(1, TF_AGREEMENT_SOURCE_FIELDS[0], 0.0, allow_duplicates=True)
    with pytest.raises(RuntimeError, match="TF_AGREEMENT_SOURCE_DUPLICATE"):
        compute_tf_agreement_score(duplicate)

    with pytest.raises(RuntimeError, match="TF_AGREEMENT_SOURCE_EMPTY"):
        compute_tf_agreement_score(_frame().iloc[:0])
