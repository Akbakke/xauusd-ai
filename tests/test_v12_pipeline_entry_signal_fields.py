import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace

from gx1.scripts.audit_model_native_direction_pockets_v1 import (
    MODEL_DIRECTION_SELECTION_MODE,
    _assert_selection_score_mode,
    _decision,
    _model_direction_contract_failures,
    _side_from_predictions,
    _selected_from_predictions,
)
from tests.model_native_serve_gate_support import passing_direction_repair_pockets


def _canonical_model_direction_predictions() -> pd.DataFrame:
    probabilities = np.asarray(
        [
            [0.80, 0.10, 0.10],
            [0.10, 0.75, 0.15],
            [0.10, 0.10, 0.80],
        ],
        dtype=np.float64,
    )
    direction_logits = np.log(probabilities)
    public_logits = np.column_stack(
        [np.max(direction_logits[:, :2], axis=1), direction_logits[:, 2]]
    )
    return pd.DataFrame(
        {
            "selection_score_mode": [MODEL_DIRECTION_SELECTION_MODE] * 3,
            "pred_direction": [0, 1, 2],
            "direction_logits": direction_logits.tolist(),
            "public_trade_flat_decision_logits": public_logits.tolist(),
            "p_long": probabilities[:, 0],
            "p_short": probabilities[:, 1],
            "p_flat": probabilities[:, 2],
            "public_trade_probability": [0.88888889, 0.83333333, 0.11111111],
            "public_flat_probability": [0.11111111, 0.16666667, 0.88888889],
            "public_trade_flat_margin": [2.07944154, 1.60943791, -2.07944154],
            "public_trade_flat_hard_decision": [0, 0, 1],
        }
    )


def _passing_direction_pocket_summaries() -> dict[str, dict[str, object]]:
    summaries = passing_direction_repair_pockets()
    short_template = dict(summaries["rising_channel_support_touch"])
    long_template = dict(summaries["falling_channel_resistance_touch"])
    for name in (
        "intraday_bull",
        "intraday_bull__htf_bull",
        "intraday_bull__htf_bear",
    ):
        summaries[name] = dict(short_template)
    for name in (
        "intraday_bear",
        "intraday_bear__htf_bear",
        "intraday_bear__htf_bull",
    ):
        summaries[name] = dict(long_template)
    return summaries


def test_entry_decision_latency_uses_t_plus_5_availability():
    from gx1.execution.v12_pipeline import _entry_decision_latency_fields

    fields = _entry_decision_latency_fields(
        pd.Timestamp("2026-07-09T12:18:00Z"),
        pd.Timestamp("2026-07-09T12:05:00Z"),
    )

    assert fields["decision_ts"] == "2026-07-09 12:05:00+00:00"
    assert fields["decision_available_ts"] == "2026-07-09 12:10:00+00:00"
    assert fields["entry_signal_latency_sec"] == 480.0
    assert fields["entry_signal_latency_min"] == 8.0
    assert fields["entry_signal_stale"] is True


def test_entry_decision_latency_allows_fresh_signal_inside_cap():
    from gx1.execution.v12_pipeline import _entry_decision_latency_fields

    fields = _entry_decision_latency_fields(
        pd.Timestamp("2026-07-09T12:10:00Z"),
        pd.Timestamp("2026-07-09T12:05:00Z"),
    )

    assert fields["entry_signal_latency_sec"] == 0.0
    assert fields["entry_signal_stale"] is False


def test_entry_validation_failure_does_not_consume_fresh_m5_bucket(
    monkeypatch,
):
    from gx1.execution.v12_pipeline import (
        ENTRY_SEQ_LEN,
        EntryDecisionUnavailable,
        V12Pipeline,
    )

    class _SmartEntry:
        def __init__(self) -> None:
            self.calls = 0

        def predict_live_bar(self, _loader, _decision_m5, **_kwargs):
            self.calls += 1
            return {}

    smart = _SmartEntry()
    monkeypatch.setattr(
        "gx1.execution.v12_pipeline._require_unified_model",
        lambda *_args, **_kwargs: None,
    )
    pipeline = V12Pipeline(
        prebuilt_loader=object(),
        smart_entry=smart,
    )
    decision_m5 = pd.Timestamp("2026-07-09T12:05:00Z")
    pipeline._last_augmented = pd.DataFrame(
        {"atr_bps": [np.nan] * ENTRY_SEQ_LEN},
        index=pd.date_range(
            end=decision_m5,
            periods=ENTRY_SEQ_LEN,
            freq="5min",
        ),
    )
    pipeline._refresh_entry_canonical = lambda _now: None
    pipeline._last_entry_prebuilt_snapshot = SimpleNamespace(
        pair_generation_id="1" * 64,
        pair_manifest_sha256="2" * 64,
    )

    for _ in range(2):
        with pytest.raises(EntryDecisionUnavailable) as exc_info:
            pipeline.make_entry_decision(
                pd.Timestamp("2026-07-09T12:10:00Z"),
                bid=3300.0,
                ask=3300.2,
            )
        assert exc_info.value.reason == "model_native_atr_invalid"
        assert pipeline._last_smart_bucket is None

    assert smart.calls == 2


def test_invalid_model_direction_is_structured_unavailable_not_flat(
    monkeypatch,
) -> None:
    from gx1.execution.v12_pipeline import (
        ENTRY_SEQ_LEN,
        EntryDecisionUnavailable,
        V12Pipeline,
    )

    class _InvalidDirectionEntry:
        def __init__(self) -> None:
            self.calls = 0

        def predict_live_bar(self, _loader, decision_m5, **_kwargs):
            return {"time": decision_m5}

        def decide(self, _head, *, atr_bps):
            self.calls += 1
            assert atr_bps == 12.0
            raise RuntimeError("direction logits/probability parity mismatch")

    smart = _InvalidDirectionEntry()
    monkeypatch.setattr(
        "gx1.execution.v12_pipeline._require_unified_model",
        lambda *_args, **_kwargs: None,
    )
    pipeline = V12Pipeline(
        prebuilt_loader=object(),
        smart_entry=smart,
    )
    decision_m5 = pd.Timestamp("2026-07-09T12:05:00Z")
    pipeline._last_augmented = pd.DataFrame(
        {"atr_bps": [12.0] * ENTRY_SEQ_LEN},
        index=pd.date_range(end=decision_m5, periods=ENTRY_SEQ_LEN, freq="5min"),
    )
    pipeline._refresh_entry_canonical = lambda _now: None
    pipeline._last_entry_prebuilt_snapshot = SimpleNamespace(
        pair_generation_id="1" * 64,
        pair_manifest_sha256="2" * 64,
    )

    for _ in range(2):
        with pytest.raises(EntryDecisionUnavailable) as exc_info:
            pipeline.make_entry_decision(
                pd.Timestamp("2026-07-09T12:10:00Z"),
                bid=3300.0,
                ask=3300.2,
            )
        assert exc_info.value.reason == "model_native_direction_decision_invalid"
        assert exc_info.value.evidence["error_type"] == "RuntimeError"
        assert "parity mismatch" in exc_info.value.evidence["error"]
        assert "action" not in exc_info.value.evidence
        assert "model_direction" not in exc_info.value.evidence
        assert pipeline._last_smart_bucket is None

    assert smart.calls == 2


def test_entry_decision_latency_does_not_floor_away_subminute_staleness():
    from gx1.execution.v12_pipeline import _entry_decision_latency_fields

    fields = _entry_decision_latency_fields(
        pd.Timestamp("2026-07-09T12:06:31Z"),
        pd.Timestamp("2026-07-09T12:00:00Z"),
    )

    assert fields["entry_signal_latency_sec"] == 91.0
    assert fields["entry_signal_latency_cap_sec"] == 90.0
    assert fields["entry_signal_stale"] is True


def test_latest_closed_m5_start_excludes_current_forming_bar():
    from gx1.execution.v12_pipeline import _latest_closed_m5_start

    assert _latest_closed_m5_start(pd.Timestamp("2026-07-09T12:07:00Z")) == pd.Timestamp(
        "2026-07-09T12:00:00Z"
    )
    assert _latest_closed_m5_start(pd.Timestamp("2026-07-09T12:10:00Z")) == pd.Timestamp(
        "2026-07-09T12:05:00Z"
    )


def test_model_native_direction_pocket_gate_fails_rising_support_label_error():
    summaries = _passing_direction_pocket_summaries()
    summaries["rising_channel_support_touch"].update(
        selected_label_correct_count=60,
        selected_label_error_count=60,
        selected_label_correct_rate=0.50,
        selected_label_error_rate=0.50,
        selected_label_error_wilson_upper_95=0.60,
    )

    decision, failures = _decision(summaries)

    assert decision == "FAIL"
    assert any(
        "rising_channel_support_touch selected label error rate" in x
        for x in failures
    )


def test_model_native_direction_pocket_gate_fails_countertrend_short_trap() -> None:
    summaries = _passing_direction_pocket_summaries()
    summaries["countertrend_short_trap"].update(
        selected_label_correct_count=48,
        selected_label_error_count=72,
        selected_label_correct_rate=0.40,
        selected_label_error_rate=0.60,
        selected_label_error_wilson_upper_95=0.70,
        selected_mean_proxy_pnl_bps=-4.0,
    )

    decision, failures = _decision(summaries)

    assert decision == "FAIL"
    assert any(
        "countertrend_short_trap selected label error rate" in x
        for x in failures
    )


def test_smart_direction_pocket_gate_fails_low_selected_support() -> None:
    summaries = _passing_direction_pocket_summaries()
    summaries["rising_channel_support_touch"].update(
        rows=30,
        selected_rows=12,
        selected_label_correct_count=11,
        selected_label_error_count=1,
        selected_label_correct_rate=11.0 / 12.0,
        selected_label_error_rate=1.0 / 12.0,
        selected_label_error_wilson_upper_95=0.25,
    )

    decision, failures = _decision(summaries)

    assert decision == "FAIL"
    assert any(
        "rising_channel_support_touch selected coverage 12 < 100" in x
        for x in failures
    )


def test_direction_pocket_utility_is_an_offline_launch_gate_only() -> None:
    summaries = _passing_direction_pocket_summaries()
    summaries["rising_channel_support_touch"][
        "selected_mean_proxy_pnl_bps"
    ] = -999.0

    decision, failures = _decision(summaries)

    assert decision == "FAIL"
    assert any("selected mean proxy pnl -999.00 <= 0.00" in failure for failure in failures)


def test_model_native_direction_pocket_gate_fails_when_edge_support_is_empty() -> None:
    summaries = _passing_direction_pocket_summaries()
    for row in summaries.values():
        row.update(rows=0, selected_rows=0)

    decision, failures = _decision(summaries)

    assert decision == "FAIL"
    assert any("pocket support 0 < 100" in failure for failure in failures)
    assert any("direction edge is unproven" in failure for failure in failures)


def test_direction_audits_use_model_argmax_including_flat_without_threshold() -> None:
    frame = _canonical_model_direction_predictions()
    frame["edge_score"] = [-99.0, 99.0, 99.0]
    frame["selection_score"] = [99.0, -99.0, 99.0]
    frame["selection_score_threshold"] = [100.0, -100.0, -100.0]
    frame["expected_utility_long_bps"] = [-10.0, 50.0, 50.0]
    frame["expected_utility_short_bps"] = [50.0, -10.0, -10.0]
    frame["trade_side"] = [1, 0, 0]
    frame["action"] = ["SHORT", "LONG", "LONG"]

    assert _model_direction_contract_failures(frame) == []
    assert _side_from_predictions(frame).tolist() == [0, 1, 2]
    assert _selected_from_predictions(frame).tolist() == [True, True, False]


def test_direction_audits_require_model_direction_selection_mode() -> None:
    frame = _canonical_model_direction_predictions()
    frame["selection_score_mode"] = ["expected_utility"] * len(frame)

    failures = _model_direction_contract_failures(frame)

    assert any("selection_score_mode mismatch" in item for item in failures)
    assert any(MODEL_DIRECTION_SELECTION_MODE in item for item in failures)
    assert _assert_selection_score_mode(frame)


def test_direction_audits_require_canonical_public_trade_flat_columns() -> None:
    frame = _canonical_model_direction_predictions().drop(
        columns=["public_trade_flat_margin"]
    )

    failures = _model_direction_contract_failures(frame)

    assert any("missing prediction columns" in item for item in failures)
    assert any("public_trade_flat_margin" in item for item in failures)


def test_direction_audits_reject_pred_direction_probability_argmax_mismatch() -> None:
    frame = _canonical_model_direction_predictions()
    frame.loc[0, "pred_direction"] = 1

    failures = _model_direction_contract_failures(frame)

    assert any("pred_direction mismatches" in item for item in failures)


def test_direction_audits_reject_tied_top_direction_logits() -> None:
    frame = _canonical_model_direction_predictions()
    tied_probabilities = np.asarray([0.45, 0.45, 0.10], dtype=np.float64)
    tied_logits = np.log(tied_probabilities)
    frame.at[0, "direction_logits"] = tied_logits.tolist()
    frame.at[0, "public_trade_flat_decision_logits"] = [
        float(tied_logits[0]),
        float(tied_logits[2]),
    ]
    frame.loc[0, ["p_long", "p_short", "p_flat"]] = tied_probabilities
    public_trade_probability = float(
        tied_probabilities[0] / (tied_probabilities[0] + tied_probabilities[2])
    )
    frame.loc[0, ["public_trade_probability", "public_flat_probability"]] = [
        public_trade_probability,
        1.0 - public_trade_probability,
    ]
    frame.loc[0, "public_trade_flat_margin"] = float(tied_logits[0] - tied_logits[2])

    failures = _model_direction_contract_failures(frame)

    assert any("no unique top class" in item for item in failures)


def test_direction_audits_reject_public_hard_decision_vs_pred_direction() -> None:
    frame = _canonical_model_direction_predictions()
    frame.loc[2, "public_trade_flat_hard_decision"] = 0
    frame.loc[2, "public_trade_probability"] = 0.80
    frame.loc[2, "public_flat_probability"] = 0.20
    frame.loc[2, "public_trade_flat_margin"] = 1.0

    failures = _model_direction_contract_failures(frame)

    assert any("hard_decision mismatches pred_direction" in item for item in failures)


def test_direction_audits_reject_public_probability_and_margin_disagreement() -> None:
    frame = _canonical_model_direction_predictions()
    frame.loc[0, "public_trade_flat_hard_decision"] = 1

    failures = _model_direction_contract_failures(frame)

    assert any("hard_decision mismatches canonical public probabilities" in item for item in failures)
    assert any("hard_decision mismatches canonical public margin" in item for item in failures)


def test_direction_audits_reject_noncanonical_public_pair_surface() -> None:
    frame = _canonical_model_direction_predictions()
    frame.loc[0, "public_trade_probability"] = 0.80
    frame.loc[0, "public_flat_probability"] = 0.20
    frame.loc[0, "public_trade_flat_margin"] = 1.38629436

    failures = _model_direction_contract_failures(frame)

    assert any(
        "public trade/FLAT probabilities do not match final direction" in item
        for item in failures
    )
    assert any(
        "public trade/FLAT margin does not match final direction" in item
        for item in failures
    )


def test_direction_audit_sources_expose_no_selection_threshold_cli() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = [root / "gx1/scripts/audit_model_native_direction_pockets_v1.py"]

    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert '"--edge-threshold"' not in text
        assert '"--selection-score-threshold"' not in text


def test_model_native_pocket_audit_is_exact_seq513_and_immutable_only() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "gx1/scripts/audit_model_native_direction_pockets_v1.py"
    ).read_text(encoding="utf-8")

    assert "MODEL_NATIVE_SIGNAL_DIM" in source
    assert "require_model_native_signal_contract(" in source
    assert "EXPECTED_CTX_CONT_DIM = MODEL_NATIVE_CTX_CONT_DIM" in source
    assert "EXPECTED_CTX_CAT_DIM = MODEL_NATIVE_CTX_CAT_DIM" in source
    assert "MODEL_NATIVE_DIRECTION_POCKET_AUDIT" in source
    assert "MODEL_NATIVE_DIRECTION_POCKET_SCHEMA_VERSION" in source
    assert '"--dataset-parquet"' in source
    assert '"--bundle-metadata-json"' in source
    assert '"--max-bad-side-rate"' not in source
    assert '"--min-selected-rows"' not in source
    assert '"--min-mean-proxy-pnl-bps"' not in source
    assert '"--split"' not in source
    assert '"--model-name"' not in source
    assert "replace_latest_json_mirror" not in source
    assert "_latest.json" not in source
    assert "_latest.md" not in source
    assert "_first_named_column" not in source
    assert "required=False" not in source
    assert "smart520" not in source.lower()
    assert "audit_thresholds_are_live_direction_rules" in source
    assert "argmax(final_model_forward_after_learned_evidence_fusion_and_calibration.direction_logits)" in source
