"""Tests for materialize_investigate_trail_stop_deep_dive_v1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import materialize_investigate_trail_stop_deep_dive_v1 as gate


def _trade_df(pnls, mfes):
    return pd.DataFrame(
        {
            "candidate_uid_v1": ["X"] * len(pnls),
            "bars_held_v1": list(range(len(pnls))),
            "running_pnl_at_close_bps_v1": pnls,
            "running_mfe_bps_v1": mfes,
        }
    )


def test_decompose_never_fired_when_mfe_too_low() -> None:
    df = _trade_df([0.0, 1.0, -1.0, 2.0], [0.0, 1.0, 1.0, 2.0])  # max mfe 2 < 5 bps min
    out = gate._decompose_trade(df, "F1")
    assert out["firing_status_v1"] == "NEVER_FIRED"
    assert out["realized_pnl_v1"] == 2.0
    assert out["trail_stop_pnl_v1"] == 2.0


def test_decompose_fires_when_giveback_exceeds_25_pct() -> None:
    # MFE peaks at 100 (bar 1), then PNL drops to 70 (bar 2): giveback = 30/100 = 0.30 >= 0.25
    df = _trade_df([0.0, 100.0, 70.0, 50.0], [0.0, 100.0, 100.0, 100.0])
    out = gate._decompose_trade(df, "F1")
    assert out["firing_status_v1"] == "FIRED"
    assert out["fire_bar_index_v1"] == 2
    assert out["trail_stop_pnl_v1"] == 70.0
    assert out["bars_to_fire_v1"] == 2


def test_decompose_fired_at_or_near_peak_when_no_post_fire_recovery() -> None:
    df = _trade_df([0.0, 100.0, 70.0, 50.0], [0.0, 100.0, 100.0, 100.0])
    out = gate._decompose_trade(df, "F1")
    # After firing at bar 2 (PNL 70), the rest is 50 -> no higher PNL afterwards.
    # peak_pnl_at_close = 100 (bar 1); trail-stop PNL 70 < peak 100 -> classified as
    # FIRED_AFTER_PEAK_PNL_REGRET_LATE_EXIT.
    assert out["failure_mode_v1"] == "FIRED_AFTER_PEAK_PNL_REGRET_LATE_EXIT"


def test_decompose_fired_before_peak_when_post_fire_higher_pnl() -> None:
    # MFE peaks at 100 (bar 1), PNL drops to 70 (firing bar 2), then PNL recovers to 150 (bar 3)
    df = _trade_df(
        [0.0, 100.0, 70.0, 150.0, 80.0],
        [0.0, 100.0, 100.0, 150.0, 150.0],
    )
    out = gate._decompose_trade(df, "F1")
    assert out["firing_status_v1"] == "FIRED"
    assert out["fire_bar_index_v1"] == 2
    assert out["trail_stop_pnl_v1"] == 70.0
    # Post-fire max PNL = 150 > 70 -> regret early exit.
    assert out["failure_mode_v1"] == "FIRED_BEFORE_PEAK_PNL_REGRET_EARLY_EXIT"


def test_decompose_never_fired_trade_lost_at_realized() -> None:
    df = _trade_df([0.0, -10.0, -50.0, -100.0], [0.0, 0.5, 1.0, 1.0])  # mfe < 5, never fires
    out = gate._decompose_trade(df, "F1")
    assert out["firing_status_v1"] == "NEVER_FIRED"
    assert out["failure_mode_v1"] == "NEVER_FIRED_TRADE_LOST_AT_REALIZED"


def test_decompose_never_fired_trade_won_at_realized() -> None:
    df = _trade_df([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0])
    # MFE never exceeds 5; PNL ends positive.
    out = gate._decompose_trade(df, "F1")
    assert out["firing_status_v1"] == "NEVER_FIRED"
    assert out["failure_mode_v1"] == "NEVER_FIRED_TRADE_WON_AT_REALIZED"


def test_summarize_failure_modes_aggregates_correctly() -> None:
    rows = [
        {
            "fold_id_v1": "F1",
            "candidate_uid_v1": f"T{i}",
            "trail_stop_pnl_v1": pnl,
            "realized_pnl_v1": pnl,
            "peak_pnl_at_close_bps_v1": pnl,
            "delta_trail_stop_minus_realized_v1": 0.0,
            "delta_peak_pnl_minus_trail_stop_v1": 0.0,
            "giveback_at_fire_bps_v1": 10.0,
            "mfe_at_fire_bps_v1": 50.0,
            "bars_to_fire_v1": 5,
            "failure_mode_v1": mode,
        }
        for i, (mode, pnl) in enumerate(
            [
                ("FIRED_BEFORE_PEAK_PNL_REGRET_EARLY_EXIT", 10.0),
                ("FIRED_BEFORE_PEAK_PNL_REGRET_EARLY_EXIT", 20.0),
                ("FIRED_AT_OR_NEAR_PEAK_PNL_OK", 50.0),
            ]
        )
    ]
    out = gate._summarize_failure_modes(rows)
    assert out["trade_count_v1"] == 3
    assert out["failure_mode_counts_v1"]["FIRED_BEFORE_PEAK_PNL_REGRET_EARLY_EXIT"] == 2
    assert (
        out["failure_mode_counts_v1"]["FIRED_AT_OR_NEAR_PEAK_PNL_OK"] == 1
    )
    assert out["pnl_by_mode_v1"]["FIRED_BEFORE_PEAK_PNL_REGRET_EARLY_EXIT"]["mean_pnl_v1"] == pytest.approx(15.0)


def test_recommend_hybrid_picks_dominant_failure_mode() -> None:
    summaries = {
        "F1": {
            "failure_mode_pcts_v1": {
                "FIRED_BEFORE_PEAK_PNL_REGRET_EARLY_EXIT": 70.0,
                "FIRED_AT_OR_NEAR_PEAK_PNL_OK": 20.0,
                "FIRED_AFTER_PEAK_PNL_REGRET_LATE_EXIT": 10.0,
            }
        },
        "F2": {
            "failure_mode_pcts_v1": {
                "FIRED_BEFORE_PEAK_PNL_REGRET_EARLY_EXIT": 75.0,
                "FIRED_AT_OR_NEAR_PEAK_PNL_OK": 25.0,
            }
        },
    }
    out = gate._recommend_hybrid(summaries)
    assert out["primary_failure_mode_v1"] == "FIRED_BEFORE_PEAK_PNL_REGRET_EARLY_EXIT"
    # Mean pct of dominant mode = (70 + 75) / 2 = 72.5
    assert out["failure_pct_means_v1"]["FIRED_BEFORE_PEAK_PNL_REGRET_EARLY_EXIT"] == pytest.approx(72.5)
    assert any("POSTPONES" in note for note in out["design_notes_v1"])


def test_validate_final_status_rejects_unknown_status() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("MADE_UP", "ACCEPT_TRAIL_STOP_AS_RESEARCH_BASELINE_V1")


def test_validate_final_status_rejects_unknown_next_action() -> None:
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status("INVESTIGATE_TRAIL_STOP_LOCKED_V1", "TRAIN_NOW")


def test_validate_no_deprecated_revival_passes_on_self() -> None:
    from pathlib import Path

    gate.validate_no_deprecated_revival(Path(gate.__file__))
