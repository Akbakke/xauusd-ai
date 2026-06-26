import numpy as np
import pandas as pd
import pytest

from gx1.scripts.rescore_forward_outcome_v10v2_full import (
    _add_entry_iql_base_columns,
    _assert_v10_base_matches_v2,
)


def _minimal_v10_v2_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "p_long_v2": [0.70, 0.20],
            "p_short_v2": [0.20, 0.45],
            "p_flat_v2": [0.10, 0.35],
            "p_hat_v2": [0.70, 0.45],
            "tradable_prob_v2": [0.80, 0.25],
            "mfe_first_n_pred_v2": [12.5, -3.0],
            "path_quality_pred_v2": [0.40, -0.20],
            "bad_path_prob_v2": [0.10, 0.55],
            "direction_logit_long_v2": [3.0, -1.0],
            "direction_logit_short_v2": [1.0, 2.0],
            "direction_logit_flat_v2": [0.0, 1.0],
            "path_quality_std_v2": [0.0, 0.0],
            "v10_dip_0_v2": [0.33, 0.66],
        }
    )


def test_add_entry_iql_base_columns_overwrites_contract_names():
    out = _add_entry_iql_base_columns(_minimal_v10_v2_frame())

    for base_col, v2_col in [
        ("p_long", "p_long_v2"),
        ("p_short", "p_short_v2"),
        ("p_flat", "p_flat_v2"),
        ("p_hat", "p_hat_v2"),
        ("tradable_prob", "tradable_prob_v2"),
        ("mfe_first_n_pred", "mfe_first_n_pred_v2"),
        ("path_quality_pred", "path_quality_pred_v2"),
        ("bad_path_prob", "bad_path_prob_v2"),
        ("direction_logit_long", "direction_logit_long_v2"),
        ("direction_logit_short", "direction_logit_short_v2"),
        ("direction_logit_flat", "direction_logit_flat_v2"),
        ("path_quality_std", "path_quality_std_v2"),
        ("v10_dip_0", "v10_dip_0_v2"),
    ]:
        np.testing.assert_allclose(out[base_col], out[v2_col])

    np.testing.assert_allclose(out["margin"], [0.50, 0.10], atol=1e-7)
    np.testing.assert_allclose(out["uncertainty_score"], [0.30, 0.55], atol=1e-7)

    probs = out[["p_long_v2", "p_short_v2", "p_flat_v2"]].to_numpy(dtype=float)
    expected_entropy = -np.sum(probs * np.log(probs), axis=1)
    np.testing.assert_allclose(out["entropy_v1"], expected_entropy, atol=1e-7)

    _assert_v10_base_matches_v2(out)


def test_assert_v10_base_matches_v2_rejects_stale_base_column():
    out = _add_entry_iql_base_columns(_minimal_v10_v2_frame())
    out.loc[0, "p_long"] = 0.01

    with pytest.raises(AssertionError, match="p_long"):
        _assert_v10_base_matches_v2(out)
