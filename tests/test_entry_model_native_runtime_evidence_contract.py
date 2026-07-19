from __future__ import annotations

import math
import copy
from pathlib import Path

import pytest
import pandas as pd

from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
    ModelNativeRuntimeEvidenceError,
    require_model_native_entry_time,
    require_model_native_runtime_evidence,
)
from tests.model_native_sizing_support import unverified_learned_sizing_authority
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)
from gx1.execution.v12_paper_runner import (
    MODEL_NATIVE_EXECUTABLE_DECISION_REQUIRED_FIELDS,
    require_executable_model_native_entry_decision,
)
from gx1.execution.v12_pipeline import ENTRY_SEQ_LEN, V12Pipeline
from gx1.monitoring.trade_journal import TradeJournal


def _softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exponentials = [math.exp(value - peak) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def _valid_evidence() -> dict:
    direction_logits = [2.0, 0.2, -1.0]
    direction_probs = _softmax(direction_logits)
    public_logits = [2.0, -1.0]
    public_probs = _softmax(public_logits)
    side_logits = [1.0, -0.5]
    side_probs = _softmax(side_logits)
    side_bad_logits = [-1.2, 0.7]
    side_validity_logits = [1.4, -0.3]
    mtf_logits = [0.8, -0.2, -0.6]
    rail_logits = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    tf_logit = 0.4
    size_logit = -0.2
    path_log_var = math.log(0.25)
    return {
        "decision_ts": "2026-07-08T17:55:00+00:00",
        "runtime_evidence_schema_version": "entry_model_native_runtime_evidence_v1",
        "model_policy": "xau_seq513_model_native_direction_argmax_v1",
        "session_id": 2,
        "session": "OVERLAP",
        "raw_direction_logits": [2.09, 0.33, -1.1],
        "direction_logits": direction_logits,
        "direction_probs": direction_probs,
        "model_direction_index": 0,
        "model_direction": "LONG",
        "selected_side": 0,
        "public_trade_flat_decision_logits": public_logits,
        "public_trade_flat_decision_probs": public_probs,
        "public_trade_flat_decision_index": 0,
        "public_trade_flat_decision": "TRADE",
        "p_trade": public_probs[0],
        "p_flat_hier": public_probs[1],
        "model_native_logits": [0.5, -0.25, 0.1],
        "path_quality_raw": 1.25,
        "path_quality": 1.25,
        "path_quality_pred": 1.25,
        "path_quality_log_var": path_log_var,
        "path_quality_std": math.exp(0.5 * path_log_var),
        "mfe_first_n": 6.5,
        "mfe_first_n_pred": 6.5,
        "tradable_logit": _logit(0.82),
        "tradable_prob": 0.82,
        "trade_logit": 0.7,
        "bad_path_logit_raw": _logit(0.11),
        "bad_path_logit": _logit(0.11),
        "bad_path_prob": 0.11,
        "clean_edge_logit": _logit(0.76),
        "clean_edge_prob": 0.76,
        "survival_logit": _logit(0.68),
        "survival_prob": 0.68,
        "dip_pred": [0.0] * 18,
        "forecast_pred": [0.0] * 4,
        "timing_pred": [0.0] * 12,
        "tail_risk_pred": [0.0] * 6,
        "vol_forecast_pred": [0.0] * 3,
        "atr_bps": 12.0,
        "tf_agreement_logit": tf_logit,
        "tf_agreement_pred": _sigmoid(tf_logit),
        "position_size_logit": size_logit,
        "position_size_pred": _sigmoid(size_logit),
        "sizing_authority_contract": unverified_learned_sizing_authority(),
        "p_long_given_trade": side_probs[0],
        "p_short_given_trade": side_probs[1],
        "side_logits": side_logits,
        "side_probs": side_probs,
        "side_utility": [2.4, -0.8],
        "side_bad_path_logit": side_bad_logits,
        "long_bad_path_prob": _sigmoid(side_bad_logits[0]),
        "short_bad_path_prob": _sigmoid(side_bad_logits[1]),
        "side_validity_logit": side_validity_logits,
        "long_validity_prob": _sigmoid(side_validity_logits[0]),
        "short_validity_prob": _sigmoid(side_validity_logits[1]),
        "side_mae": [-3.2, -8.1],
        "mtf_dir_logits": mtf_logits,
        "mtf_dir_probs": _softmax(mtf_logits),
        "mtf_trend_evidence": 0.69,
        "specialist_names": list(MODEL_NATIVE_TRAINING_SPECIALISTS),
        "specialist_gate": [0.125] * len(MODEL_NATIVE_TRAINING_SPECIALISTS),
        "trendline_rail_logits": rail_logits,
        "trendline_rail_probs": [_sigmoid(value) for value in rail_logits],
        "geometry_channel_edge_pressure": 0.42,
        "geometry_rising_support_rail_long_pressure": 0.81,
        "geometry_rising_support_rail_short_trap_pressure": 0.07,
        "geometry_falling_resistance_rail_short_pressure": 0.02,
        "geometry_falling_resistance_rail_long_trap_pressure": 0.03,
        "calibration_version": "direction-cal-v1",
        "direction_calibration_enabled": True,
        "direction_calibration_temperature": 1.1,
        "direction_calibration_bias": [0.1, -0.1, 0.0],
        "path_calibration_enabled": True,
        "path_calibration": {
            "enabled": True,
            "version": "path-cal-v1",
            "path_quality_scale": 1.0,
            "path_quality_shift": 0.0,
            "bad_path_temperature": 1.0,
            "bad_path_bias": 0.0,
        },
    }


def _valid_executable_evidence() -> dict:
    evidence = _valid_evidence()
    evidence.update(
        {
            "decision_available_ts": "2026-07-08T18:00:00+00:00",
            "entry_signal_latency_sec": 30.0,
            "context_cutoff_ts": "2026-07-08T17:55:00+00:00",
            "context_age_m5_bars": 0,
        }
    )
    return evidence


def _valid_executable_decision() -> dict:
    snapshot = _valid_executable_evidence()
    probabilities = snapshot["direction_probs"]
    decision = {
        key: copy.deepcopy(snapshot[key])
        for key in MODEL_NATIVE_EXECUTABLE_DECISION_REQUIRED_FIELDS
        if key in snapshot
    }
    decision.update(
        {
            "action": "TAKE_LONG_NOW",
            "action_id": 1,
            "edge_score": max(probabilities[0], probabilities[1]) - probabilities[2],
            "selection_score_mode": "model_direction_argmax",
            "selection_score": probabilities[0],
            "p_long": probabilities[0],
            "p_short": probabilities[1],
            "p_flat": probabilities[2],
            "v10_path_quality_pred": snapshot["path_quality_pred"],
            "v10_mfe_pred_at_entry": snapshot["mfe_first_n_pred"],
            "v10_tradable_prob": snapshot["tradable_prob"],
            "v10_bad_path_prob": snapshot["bad_path_prob"],
            "_v10_snapshot": copy.deepcopy(snapshot),
            "policy": snapshot["model_policy"],
            "stub": False,
            "entry_signal_latency_min": snapshot["entry_signal_latency_sec"] / 60.0,
            "entry_signal_latency_cap_sec": 90.0,
            "entry_signal_stale": False,
            "context_refresh_in_flight": False,
            "context_mtf_incremental": False,
        }
    )
    assert set(decision) == set(MODEL_NATIVE_EXECUTABLE_DECISION_REQUIRED_FIELDS)
    return decision


def test_runtime_evidence_contract_returns_copy_and_preserves_values() -> None:
    evidence = _valid_evidence()

    validated = require_model_native_runtime_evidence(
        evidence,
        context="CONTRACT_TEST",
    )

    assert validated == evidence
    assert validated is not evidence
    assert validated["position_size_pred"] == pytest.approx(_sigmoid(-0.2))


def test_runtime_evidence_rejects_dynamic_sizing_authority_tamper() -> None:
    evidence = _valid_evidence()
    evidence["sizing_authority_contract"]["dynamic_sizing_authorized"] = True

    with pytest.raises(
        ModelNativeRuntimeEvidenceError,
        match="sizing_authority_contract",
    ):
        require_model_native_runtime_evidence(
            evidence,
            context="CONTRACT_TEST",
        )


@pytest.mark.parametrize(
    "missing_key",
    [
        "specialist_gate",
        "side_utility",
        "mtf_dir_logits",
        "mtf_trend_evidence",
        "calibration_version",
        "path_calibration",
        "runtime_evidence_schema_version",
        "model_policy",
        "session_id",
    ],
)
def test_runtime_evidence_contract_rejects_missing_required_surface(
    missing_key: str,
) -> None:
    evidence = _valid_evidence()
    del evidence[missing_key]

    with pytest.raises(
        ModelNativeRuntimeEvidenceError,
        match=missing_key,
    ):
        require_model_native_runtime_evidence(evidence, context="CONTRACT_TEST")


def test_runtime_evidence_contract_rejects_position_size_parity_mismatch() -> None:
    evidence = _valid_evidence()
    evidence["position_size_pred"] = 0.99

    with pytest.raises(
        ModelNativeRuntimeEvidenceError,
        match="position_size_pred: parity mismatch",
    ):
        require_model_native_runtime_evidence(evidence, context="CONTRACT_TEST")


def test_runtime_evidence_contract_fails_closed_on_path_variance_overflow() -> None:
    evidence = _valid_evidence()
    evidence["path_quality_log_var"] = 2_000.0

    with pytest.raises(
        ModelNativeRuntimeEvidenceError,
        match="path_quality_std: derived value overflowed",
    ):
        require_model_native_runtime_evidence(evidence, context="CONTRACT_TEST")


def test_runtime_evidence_contract_rejects_unknown_soft_passthrough() -> None:
    evidence = _valid_evidence()
    evidence["unvalidated_live_hint"] = 1.0

    with pytest.raises(
        ModelNativeRuntimeEvidenceError,
        match=r"unexpected=\['unvalidated_live_hint'\]",
    ):
        require_model_native_runtime_evidence(evidence, context="CONTRACT_TEST")


def test_runtime_evidence_contract_rejects_aux_head_parity_mismatch() -> None:
    evidence = _valid_evidence()
    evidence["clean_edge_prob"] = 0.01

    with pytest.raises(
        ModelNativeRuntimeEvidenceError,
        match="clean_edge_prob: parity mismatch",
    ):
        require_model_native_runtime_evidence(evidence, context="CONTRACT_TEST")


def test_runtime_evidence_contract_rejects_partial_timing_evidence() -> None:
    evidence = _valid_evidence()
    evidence["entry_signal_latency_sec"] = 1.0

    with pytest.raises(
        ModelNativeRuntimeEvidenceError,
        match="timing evidence: must be absent or complete",
    ):
        require_model_native_runtime_evidence(evidence, context="CONTRACT_TEST")


def test_runtime_evidence_contract_accepts_complete_zero_staleness_timing() -> None:
    evidence = _valid_executable_evidence()

    assert require_model_native_runtime_evidence(evidence) == evidence


def test_executable_entry_time_is_derived_exactly_at_runtime_minute_resolution() -> None:
    evidence = _valid_executable_evidence()

    observed = require_model_native_entry_time(
        evidence,
        "2026-07-08T18:00:00+00:00",
    )

    assert observed.isoformat() == "2026-07-08T18:00:00+00:00"
    with pytest.raises(ModelNativeRuntimeEvidenceError, match="model-derived minute"):
        require_model_native_entry_time(
            evidence,
            "2026-07-08T18:01:00+00:00",
        )


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("decision_available_ts", "2026-07-08T18:00:01+00:00", "decision_ts \\+ 300"),
        ("entry_signal_latency_sec", 90.0001, r"immutable \[0,90\]"),
        ("context_cutoff_ts", "2026-07-08T17:50:00+00:00", "must equal decision_ts"),
        ("context_age_m5_bars", 1, "must be exact integer 0"),
    ],
)
def test_runtime_evidence_contract_rejects_tampered_executable_timing(
    key: str,
    value: object,
    match: str,
) -> None:
    evidence = _valid_executable_evidence()
    evidence[key] = value

    with pytest.raises(ModelNativeRuntimeEvidenceError, match=match):
        require_model_native_runtime_evidence(evidence, context="CONTRACT_TEST")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("model_direction_index", False),
        ("selected_side", False),
        ("public_trade_flat_decision_index", False),
        ("session_id", True),
    ],
)
def test_runtime_evidence_contract_rejects_boolean_integer_lookalikes(
    key: str,
    value: object,
) -> None:
    evidence = _valid_evidence()
    evidence[key] = value

    with pytest.raises(ModelNativeRuntimeEvidenceError, match=key):
        require_model_native_runtime_evidence(evidence, context="CONTRACT_TEST")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("runtime_evidence_schema_version", "entry_model_native_runtime_evidence_v0"),
        ("model_policy", "manual_direction_override"),
        ("session", "US"),
        ("session_id", 4),
    ],
)
def test_runtime_evidence_contract_rejects_schema_policy_or_session_tamper(
    key: str,
    value: object,
) -> None:
    evidence = _valid_evidence()
    evidence[key] = value

    with pytest.raises(ModelNativeRuntimeEvidenceError, match=key):
        require_model_native_runtime_evidence(evidence, context="CONTRACT_TEST")


def test_runner_pre_order_boundary_accepts_only_exact_executable_envelope() -> None:
    decision = _valid_executable_decision()

    validated = require_executable_model_native_entry_decision(
        decision,
        "2026-07-08T18:00:00+00:00",
    )

    assert validated == decision["_v10_snapshot"]
    assert set(validated) == {
        *MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
        "decision_available_ts",
        "entry_signal_latency_sec",
        "context_cutoff_ts",
        "context_age_m5_bars",
    }


@pytest.mark.parametrize(
    ("target", "key", "value", "match"),
    [
        ("outer", "unexpected_live_hint", 1.0, "unexpected"),
        ("outer", "policy", "manual_override", "policy mismatch"),
        ("outer", "session", "US", "snapshot parity mismatch"),
        ("outer", "action", "TAKE_SHORT_NOW", "action/direction parity"),
        ("outer", "entry_signal_latency_min", 9.0, "latency contract"),
        ("snapshot", "model_policy", "manual_override", "model_policy"),
        ("snapshot", "context_age_m5_bars", 1, "context_age_m5_bars"),
        ("snapshot", "unvalidated_live_hint", 1.0, "unexpected"),
    ],
)
def test_runner_pre_order_boundary_rejects_envelope_or_snapshot_tamper(
    target: str,
    key: str,
    value: object,
    match: str,
) -> None:
    decision = _valid_executable_decision()
    container = decision if target == "outer" else decision["_v10_snapshot"]
    container[key] = value

    with pytest.raises(RuntimeError, match=match):
        require_executable_model_native_entry_decision(
            decision,
            "2026-07-08T18:00:00+00:00",
        )


def test_runner_pre_order_boundary_rejects_pure_snapshot_without_timing() -> None:
    decision = _valid_executable_decision()
    for key in (
        "decision_available_ts",
        "entry_signal_latency_sec",
        "context_cutoff_ts",
        "context_age_m5_bars",
    ):
        decision["_v10_snapshot"].pop(key)

    with pytest.raises(RuntimeError, match="complete timing evidence missing"):
        require_executable_model_native_entry_decision(
            decision,
            "2026-07-08T18:00:00+00:00",
        )


def test_runner_pre_order_boundary_rejects_wrapper_entry_minute_tamper() -> None:
    decision = _valid_executable_decision()

    with pytest.raises(RuntimeError, match="model-derived minute"):
        require_executable_model_native_entry_decision(
            decision,
            "2026-07-08T18:01:00+00:00",
        )


def test_runner_dry_run_persists_entry_before_strict_exit_and_never_claims_oanda_close() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "gx1"
        / "execution"
        / "v12_paper_runner.py"
    ).read_text(encoding="utf-8")
    units_anchor = source.index('                event["units"] = trade_units')
    dry_open_start = source.index("                if args.dry_run:", units_anchor)
    live_open_start = source.index("                else:", dry_open_start)
    dry_open = source[dry_open_start:live_open_start]
    assert "journal.log_entry_snapshot(" in dry_open
    assert 'model_evidence=dict(decision["_v10_snapshot"])' in dry_open
    assert '"virtual_dry_run"' in dry_open
    assert (
        "if not args.dry_run:\n"
        "                            journal.log_oanda_trade_update("
    ) in source


def test_pipeline_binds_complete_timing_into_the_frozen_snapshot_before_return() -> None:
    class _SmartEntry:
        def predict_live_bar(self, _loader: object, end_ts: pd.Timestamp) -> dict:
            return {
                "context_cutoff_ts": str(end_ts),
                "context_age_m5_bars": 0,
            }

        def decide(self, head: dict, atr_bps: float) -> dict:
            snapshot = _valid_evidence()
            snapshot["atr_bps"] = atr_bps
            return {
                "_v10_snapshot": snapshot,
                "policy": snapshot["model_policy"],
                "context_cutoff_ts": head["context_cutoff_ts"],
                "context_age_m5_bars": head["context_age_m5_bars"],
            }

    decision_ts = pd.Timestamp("2026-07-08T17:55:00Z")
    index = pd.date_range(
        end=decision_ts,
        periods=ENTRY_SEQ_LEN,
        freq="5min",
    )
    pipeline = V12Pipeline(
        prebuilt_loader=object(),
        exit_xgb=object(),
        smart_entry=_SmartEntry(),
    )
    pipeline._last_augmented = pd.DataFrame({"atr_bps": 12.0}, index=index)
    pipeline._refresh_entry_canonical = lambda _now: None  # type: ignore[method-assign]

    decision = pipeline.make_entry_decision(
        pd.Timestamp("2026-07-08T18:00:30Z"),
        bid=2360.0,
        ask=2360.2,
    )

    snapshot = decision["_v10_snapshot"]
    assert snapshot["decision_available_ts"] == "2026-07-08 18:00:00+00:00"
    assert snapshot["entry_signal_latency_sec"] == 30.0
    assert snapshot["context_cutoff_ts"] == str(decision_ts)
    assert snapshot["context_age_m5_bars"] == 0
    assert set(snapshot) == {
        *MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
        "decision_available_ts",
        "entry_signal_latency_sec",
        "context_cutoff_ts",
        "context_age_m5_bars",
    }


def test_trade_journal_rejects_wrapper_entry_minute_tamper(tmp_path: Path) -> None:
    journal = TradeJournal(tmp_path, "MODEL_NATIVE_TEST")
    try:
        with pytest.raises(RuntimeError) as raised:
            journal.log_entry_snapshot(
                trade_id="T-TIME-TAMPER",
                entry_time="2026-07-08T18:01:00+00:00",
                instrument="XAU_USD",
                side="long",
                entry_price=2360.2,
                model_evidence=_valid_executable_evidence(),
                entry_bid=2360.0,
                entry_ask=2360.2,
                entry_spread_bps=(0.2 / 2360.0) * 10_000.0,
                session="OVERLAP",
                model_policy="xau_seq513_model_native_direction_argmax_v1",
                execution_checks=["fresh_quote", "model_native_sizing_authority"],
                atr_bps=12.0,
            )
        assert raised.value.__cause__ is not None
        assert "model-derived minute" in str(raised.value.__cause__)
    finally:
        journal.close()


@pytest.mark.parametrize(
    "retired_key",
    [
        "xgb_anchor_probs",
        "q_take_long",
        "advantage_over_skip",
        "hold_horizon_bars_pred",
        "sniper_overlay",
        "expected_utility_side",
    ],
)
def test_runtime_evidence_contract_rejects_retired_keys(retired_key: str) -> None:
    evidence = _valid_evidence()
    evidence[retired_key] = 0.0

    with pytest.raises(
        ModelNativeRuntimeEvidenceError,
        match=rf"retired fields=.*{retired_key}",
    ):
        require_model_native_runtime_evidence(evidence, context="CONTRACT_TEST")


def test_runtime_evidence_contract_error_carries_call_site_context() -> None:
    evidence = _valid_evidence()
    del evidence["specialist_names"]

    with pytest.raises(
        ModelNativeRuntimeEvidenceError,
        match=(
            r"\[JOURNAL_MODEL_NATIVE_RUNTIME_EVIDENCE_INVALID\] evidence: "
            r"exact schema mismatch missing=\['specialist_names'\]"
        ),
    ):
        require_model_native_runtime_evidence(evidence, context="JOURNAL")
