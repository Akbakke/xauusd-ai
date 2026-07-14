import pandas as pd
from pathlib import Path

from gx1.execution.v12_pipeline import (
    _entry_decision_latency_fields,
    _entry_signal_fields_from_candidate,
    _latest_closed_m5_start,
)
from gx1.scripts.audit_smart_direction_contribution_v1 import (
    _expected_utility_contract_failures,
    _selected,
)
from gx1.scripts.audit_smart_direction_live_like_pockets_v1 import (
    _assert_selection_score_mode,
    _decision,
    _side_from_predictions,
    _selected_from_predictions,
)


def test_entry_signal_fields_expose_margin_for_runner_sizing():
    fields = _entry_signal_fields_from_candidate(
        {
            "margin": 0.42,
            "p_hat": 0.72,
            "uncertainty_score": 0.28,
            "entropy_v1": 0.81,
        }
    )

    assert fields["margin"] == 0.42
    assert fields["margin_top1_top2"] == 0.42
    assert fields["p_hat"] == 0.72
    assert fields["uncertainty_score"] == 0.28
    assert fields["entropy_v1"] == 0.81


def test_v12_pipeline_default_loader_calls_smart_serving_gate() -> None:
    text = (
        Path(__file__).resolve().parents[1]
        / "gx1"
        / "execution"
        / "v12_pipeline.py"
    ).read_text(encoding="utf-8")

    assert "assert_smart_serving_gate()" in text


def test_entry_signal_fields_fallback_to_margin_top1_top2():
    fields = _entry_signal_fields_from_candidate({"margin_top1_top2": 0.35})

    assert fields["margin"] == 0.35
    assert fields["margin_top1_top2"] == 0.35


def test_entry_decision_latency_uses_t_plus_5_availability():
    fields = _entry_decision_latency_fields(
        pd.Timestamp("2026-07-09T12:18:00Z"),
        pd.Timestamp("2026-07-09T12:05:00Z"),
        latency_cap_sec=90.0,
    )

    assert fields["decision_ts"] == "2026-07-09 12:05:00+00:00"
    assert fields["decision_available_ts"] == "2026-07-09 12:10:00+00:00"
    assert fields["entry_signal_latency_sec"] == 480.0
    assert fields["entry_signal_latency_min"] == 8.0
    assert fields["entry_signal_stale"] is True


def test_entry_decision_latency_allows_fresh_signal_inside_cap():
    fields = _entry_decision_latency_fields(
        pd.Timestamp("2026-07-09T12:10:00Z"),
        pd.Timestamp("2026-07-09T12:05:00Z"),
        latency_cap_sec=90.0,
    )

    assert fields["entry_signal_latency_sec"] == 0.0
    assert fields["entry_signal_stale"] is False


def test_latest_closed_m5_start_excludes_current_forming_bar():
    assert _latest_closed_m5_start(pd.Timestamp("2026-07-09T12:07:00Z")) == pd.Timestamp(
        "2026-07-09T12:00:00Z"
    )
    assert _latest_closed_m5_start(pd.Timestamp("2026-07-09T12:10:00Z")) == pd.Timestamp(
        "2026-07-09T12:05:00Z"
    )


def test_smart_direction_pocket_gate_fails_rising_support_short_bias():
    template = {
        "rows": 0,
        "selected_rows": 0,
        "selected_side_short_rate": 0.0,
        "selected_side_long_rate": 0.0,
        "selected_mean_proxy_pnl_bps": 1.0,
    }
    summaries = {
        "intraday_bull": dict(template),
        "intraday_bull__htf_bull": dict(template),
        "intraday_bull__htf_bear": dict(template),
        "intraday_bear": dict(template),
        "intraday_bear__htf_bear": dict(template),
        "intraday_bear__htf_bull": dict(template),
        "rising_channel_support_touch": {
            **template,
            "rows": 30,
            "selected_rows": 30,
            "selected_side_short_rate": 0.50,
            "selected_mean_proxy_pnl_bps": 5.0,
        },
        "falling_channel_resistance_touch": dict(template),
        "support_retest_continuation": dict(template),
        "resistance_retest_continuation": dict(template),
        "rising_channel_support_continuation": dict(template),
        "falling_channel_resistance_continuation": dict(template),
        "countertrend_short_trap": dict(template),
        "countertrend_long_trap": dict(template),
        "short_high_mae_low_mfe_early_failure": dict(template),
        "long_high_mae_low_mfe_early_failure": dict(template),
    }

    decision, failures = _decision(0.35, 30, summaries)

    assert decision == "FAIL"
    assert any("rising_channel_support_touch selected SHORT rate" in x for x in failures)


def test_smart_direction_pocket_gate_fails_countertrend_short_trap() -> None:
    template = {
        "rows": 0,
        "selected_rows": 0,
        "selected_side_short_rate": 0.0,
        "selected_side_long_rate": 0.0,
        "selected_mean_proxy_pnl_bps": 1.0,
    }
    summaries = {
        "intraday_bull": dict(template),
        "intraday_bull__htf_bull": dict(template),
        "intraday_bull__htf_bear": dict(template),
        "intraday_bear": dict(template),
        "intraday_bear__htf_bear": dict(template),
        "intraday_bear__htf_bull": dict(template),
        "rising_channel_support_touch": dict(template),
        "falling_channel_resistance_touch": dict(template),
        "support_retest_continuation": dict(template),
        "resistance_retest_continuation": dict(template),
        "rising_channel_support_continuation": dict(template),
        "falling_channel_resistance_continuation": dict(template),
        "countertrend_short_trap": {
            **template,
            "rows": 30,
            "selected_rows": 30,
            "selected_side_short_rate": 0.60,
            "selected_mean_proxy_pnl_bps": -4.0,
        },
        "countertrend_long_trap": dict(template),
        "short_high_mae_low_mfe_early_failure": dict(template),
        "long_high_mae_low_mfe_early_failure": dict(template),
    }

    decision, failures = _decision(0.35, 30, summaries)

    assert decision == "FAIL"
    assert any("countertrend_short_trap selected SHORT rate" in x for x in failures)
    assert any("countertrend_short_trap selected mean proxy pnl" in x for x in failures)


def test_smart_direction_pocket_gate_fails_low_support_wrong_side() -> None:
    template = {
        "rows": 0,
        "selected_rows": 0,
        "selected_side_short_count": 0,
        "selected_side_long_count": 0,
        "selected_side_short_rate": 0.0,
        "selected_side_long_rate": 0.0,
        "selected_mean_proxy_pnl_bps": 1.0,
    }
    summaries = {
        "intraday_bull": dict(template),
        "intraday_bull__htf_bull": dict(template),
        "intraday_bull__htf_bear": dict(template),
        "intraday_bear": dict(template),
        "intraday_bear__htf_bear": dict(template),
        "intraday_bear__htf_bull": dict(template),
        "rising_channel_support_touch": {
            **template,
            "rows": 30,
            "selected_rows": 12,
            "selected_side_short_count": 1,
            "selected_side_short_rate": 1.0 / 12.0,
        },
        "falling_channel_resistance_touch": dict(template),
        "support_retest_continuation": dict(template),
        "resistance_retest_continuation": dict(template),
        "rising_channel_support_continuation": dict(template),
        "falling_channel_resistance_continuation": dict(template),
        "countertrend_short_trap": dict(template),
        "countertrend_long_trap": dict(template),
        "short_high_mae_low_mfe_early_failure": dict(template),
        "long_high_mae_low_mfe_early_failure": dict(template),
    }

    decision, failures = _decision(0.35, 30, summaries)

    assert decision == "FAIL"
    assert any("rising_channel_support_touch selected SHORT count" in x for x in failures)


def test_direction_contribution_selection_uses_expected_utility_threshold() -> None:
    frame = pd.DataFrame(
        {
            "edge_score": [0.01, 0.90],
            "selection_score": [4.0, -1.0],
            "selection_score_threshold": [0.0, 0.0],
        }
    )

    selected = _selected(frame, edge_threshold=0.145)

    assert selected.tolist() == [True, False]


def test_direction_audits_default_expected_utility_threshold_to_zero() -> None:
    frame = pd.DataFrame(
        {
            "edge_score": [0.99, 0.01],
            "selection_score_mode": ["expected_utility", "expected_utility"],
            "selection_score": [-0.1, 0.2],
        }
    )

    assert _selected(frame, edge_threshold=0.145).tolist() == [False, True]
    assert _selected_from_predictions(frame, edge_threshold=0.145).tolist() == [False, True]


def test_direction_pocket_audit_can_override_expected_utility_threshold() -> None:
    frame = pd.DataFrame(
        {
            "edge_score": [0.99, 0.01],
            "selection_score_mode": ["expected_utility", "expected_utility"],
            "selection_score": [4.0, 11.0],
            "selection_score_threshold": [0.0, 0.0],
        }
    )

    selected = _selected_from_predictions(
        frame,
        edge_threshold=0.145,
        selection_score_threshold_override=10.0,
    )

    assert selected.tolist() == [False, True]


def test_direction_pocket_audit_requires_expected_utility_prediction_surface() -> None:
    legacy = pd.DataFrame({"edge_score": [0.5], "trade_side": [1]})
    expected = pd.DataFrame(
        {
            "selection_score_mode": ["expected_utility"],
            "selection_score": [1.0],
            "selection_score_threshold": [0.0],
            "trade_side": [0],
            "expected_utility_long_bps": [7.0],
            "expected_utility_short_bps": [-2.0],
            "expected_utility_side": [0],
        }
    )

    assert _assert_selection_score_mode(expected, "expected_utility") == []
    assert any("selection_score_mode mismatch" in item for item in _assert_selection_score_mode(legacy, "expected_utility"))


def test_direction_pocket_audit_rejects_stale_selected_action_for_expected_utility() -> None:
    frame = pd.DataFrame(
        {
            "selection_score_mode": ["expected_utility", "expected_utility"],
            "selection_score": [-1.0, 2.0],
            "selection_score_threshold": [0.0, 0.0],
            "trade_side": [0, 1],
            "expected_utility_long_bps": [4.0, -3.0],
            "expected_utility_short_bps": [-1.0, 5.0],
            "expected_utility_side": [0, 1],
            "selected": [True, True],
            "action": ["TAKE_SHORT_NOW", "TAKE_LONG_NOW"],
        }
    )

    assert _selected_from_predictions(frame, edge_threshold=0.145).tolist() == [False, True]
    failures = _assert_selection_score_mode(frame, "expected_utility")
    assert any("selected column mismatches" in item for item in failures)
    assert any("action column mismatches" in item for item in failures)


def test_direction_pocket_audit_rejects_stale_trade_side_for_expected_utility() -> None:
    frame = pd.DataFrame(
        {
            "selection_score_mode": ["expected_utility"],
            "selection_score": [3.0],
            "selection_score_threshold": [0.0],
            "trade_side": [1],
            "expected_utility_long_bps": [8.0],
            "expected_utility_short_bps": [-4.0],
            "expected_utility_side": [0],
        }
    )

    assert _side_from_predictions(frame).tolist() == [0]
    failures = _assert_selection_score_mode(frame, "expected_utility")
    assert any("trade_side mismatches" in item for item in failures)


def test_direction_contribution_audit_rejects_stale_expected_utility_action() -> None:
    frame = pd.DataFrame(
        {
            "selection_score_mode": ["expected_utility"],
            "selection_score": [2.0],
            "selection_score_threshold": [0.0],
            "trade_side": [1],
            "expected_utility_long_bps": [9.0],
            "expected_utility_short_bps": [-1.0],
            "expected_utility_side": [0],
            "selected": [True],
            "action": ["TAKE_SHORT_NOW"],
        }
    )

    failures = _expected_utility_contract_failures(frame)

    assert any("trade_side mismatches" in item for item in failures)
    assert any("action side mismatches" in item for item in failures)
