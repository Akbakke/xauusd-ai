from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import materialize_entry_val_decision_journal_v1 as journal


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "side": ["LONG", "SHORT", "FLAT"],
            "trade_side": [-1, 1, 2],
            "y_long_final_pnl_at_direction_horizon_bps": [4.0, 3.0, 2.0],
            "y_short_final_pnl_at_direction_horizon_bps": [-2.0, -1.0, -3.0],
            "mfe_long_first_n_bps": [5.0, -0.5, 1.0],
            "mfe_short_first_n_bps": [-2.5, 6.0, -1.0],
            "mae_long_first_n_bps": [1.0, 2.0, 3.0],
            "mae_short_first_n_bps": [4.0, 5.0, 6.0],
            "y_long_expected_mae_bps": [1.0, 2.0, 3.0],
            "y_short_expected_mae_bps": [4.0, 5.0, 6.0],
            "research_policy_gross_spread_inclusive_pnl_bps": [4.0, -1.0, 0.0],
        }
    )


def test_val_journal_uses_existing_side_and_keeps_flat_non_trade() -> None:
    result = journal._select_side_outcomes(_rows())

    assert result.loc[0, "actual_final_executable_pnl_bps"] == 4.0
    assert result.loc[1, "actual_final_executable_pnl_bps"] == -1.0
    assert result.loc[1, "actual_mfe_bps"] == 6.0
    assert np.isnan(result.loc[2, "actual_final_executable_pnl_bps"])
    summary = journal._summary(result)
    assert summary["decision_rows"] == 3
    assert summary["trade_rows"] == 2
    assert summary["flat_rows"] == 1
    assert summary["mae_before_mfe"]["status"] == (
        "NOT_AVAILABLE_FROM_CURRENT_LABEL_SURFACE"
    )


def test_val_journal_rejects_policy_outcome_rewrite() -> None:
    rows = _rows()
    rows.loc[1, "research_policy_gross_spread_inclusive_pnl_bps"] = 7.0
    with pytest.raises(journal.ValDecisionJournalError, match="POLICY_PNL_LABEL_MISMATCH"):
        journal._select_side_outcomes(rows)
