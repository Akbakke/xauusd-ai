from __future__ import annotations

from gx1.execution.v12_daily_trade_review import render_trade_detail, trade_summary_row


def test_daily_trade_review_renders_smart_trendline_rail_fields(tmp_path) -> None:
    trade = {
        "trade_id": "T-SMART-RAIL",
        "entry_snapshot": {
            "entry_time": "2026-07-08T18:00:00Z",
            "side": "short",
            "entry_price": 2360.0,
            "entry_spread_bps": 1.2,
            "entry_score": {
                "decision_ts": "2026-07-08T17:55:00Z",
                "v10_p_long": 0.61,
                "v10_p_short": 0.24,
                "v10_path_quality_pred": 1.0,
                "v10_mfe_pred_at_entry": 6.0,
                "v10_tradable_prob": 0.8,
                "v10_bad_path_prob": 0.1,
                "smart_p_long": 0.61,
                "smart_p_short": 0.24,
                "smart_p_flat": 0.15,
                "smart_p_trade": 0.83,
                "smart_p_long_given_trade": 0.72,
                "smart_p_short_given_trade": 0.28,
                "smart_expected_utility_long_bps": 18.5,
                "smart_expected_utility_short_bps": -7.0,
                "smart_long_bad_path_prob": 0.08,
                "smart_short_bad_path_prob": 0.41,
                "smart_geometry_rising_support_rail_long_pressure": 0.81,
                "smart_geometry_rising_support_rail_short_trap_pressure": 0.77,
                "smart_geometry_falling_resistance_rail_short_pressure": 0.02,
                "smart_geometry_falling_resistance_rail_long_trap_pressure": 0.03,
                "smart_trendline_rail_long_minus_short": 0.42,
                "smart_mtf_trend_evidence": 0.69,
            },
        },
        "exit_summary": {
            "exit_time": "2026-07-08T18:20:00Z",
            "exit_price": 2365.0,
            "exit_reason": "SL",
            "realized_pnl_bps": -21.0,
            "max_mfe_bps": 0.5,
            "max_mae_bps": -24.0,
            "intratrade_drawdown_bps": 24.5,
        },
        "v12_bar_decisions": [],
    }

    summary = trade_summary_row(trade)
    out_path = tmp_path / "trade.md"
    render_trade_detail(trade, summary, out_path)
    rendered = out_path.read_text(encoding="utf-8")

    assert summary["smart_p_trade"] == 0.83
    assert summary["smart_trendline_rail_long_minus_short"] == 0.42
    assert "### SMART entry outputs" in rendered
    assert "trendline rail L-S" in rendered
    assert "+0.420" in rendered
