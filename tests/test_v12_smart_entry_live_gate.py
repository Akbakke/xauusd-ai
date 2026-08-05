from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.execution import v12_smart_entry_live as live
from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_MTF_CONTEXT_COUNT
from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
    MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION,
    MODEL_NATIVE_RUNTIME_POLICY,
)
from tests.model_native_offline_rl_support import offline_rl_evidence
from gx1.contracts.model_native_serve_gate_v1 import (
    DIRECTION_POCKET_REQUIRED_EVIDENCE_POCKETS,
    UTC_TIME_COVERAGE_SCHEMA_VERSION,
)
from gx1.execution.v12_paper_runner import (
    MODEL_NATIVE_EXECUTABLE_DECISION_REQUIRED_FIELDS,
    require_executable_model_native_entry_decision,
)
from gx1.features.htf_features import (
    HTF_V4_MATRIX_CONTRACT,
    MULTI_TF_PER_BAR_FEATURES_V4,
)
from tests.model_native_serve_gate_support import (
    passing_direction_repair_pockets,
)
from tests.model_native_sizing_support import unverified_learned_sizing_authority


REQUIRED_REPAIR_POCKETS = set(DIRECTION_POCKET_REQUIRED_EVIDENCE_POCKETS)


def _coverage(rows: int = 1_000) -> dict[str, object]:
    return {
        "schema_version": UTC_TIME_COVERAGE_SCHEMA_VERSION,
        "rows": rows,
        "first_utc": "2026-01-01T00:00:00+00:00",
        "last_utc": "2026-04-10T00:00:00+00:00",
        "utc_ns_sha256": "c" * 64,
    }


def _passing_pocket_metrics(name: str, overrides: dict | None = None) -> dict:
    row = dict(passing_direction_repair_pockets()[name])
    if overrides:
        row.update(overrides)
    return row


def test_smart_entry_mtf_window_uses_closed_bar_availability_shift(tmp_path: Path) -> None:
    idx = pd.DatetimeIndex(
        [
            "2026-07-08T12:00:00Z",
            "2026-07-08T12:05:00Z",
        ]
    )
    values = np.zeros(
        (2, len(MULTI_TF_PER_BAR_FEATURES_V4)),
        dtype=np.float32,
    )
    values[:, 0] = [1.0, 2.0]
    frame = pd.DataFrame(
        values,
        columns=MULTI_TF_PER_BAR_FEATURES_V4,
        index=idx,
    )
    frame.attrs["ts_int64"] = idx.asi8.astype("int64")
    frame.attrs["feats_np"] = values
    frame.attrs["causal_warmup_rows"] = 0
    frame.attrs["htf_feature_contract"] = HTF_V4_MATRIX_CONTRACT
    engine = live.SmartEntryLiveInference(
        bundle_dir=tmp_path,
        operating_point={
            "selection_score": live.MODEL_DIRECTION_SELECTION_MODE,
            "max_trades": 3,
        },
    )
    engine._per_tf_seq_lens = {"M5": 1}
    engine._multi_tf_shift = {"M5": pd.Timedelta(minutes=5)}

    out = engine._multi_tf_window_tensors(
        pd.Timestamp("2026-07-08T12:05:00Z"),
        multi_tf={"M5": frame},
    )

    assert float(out["seq_m5"][0, 0, 0].item()) == 2.0


def _softmax(values: list[float] | tuple[float, ...]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    exp = np.exp(arr - np.max(arr))
    return (exp / exp.sum()).tolist()


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(value))))


def _decision_engine(tmp_path: Path, selection_mode: str = live.MODEL_DIRECTION_SELECTION_MODE):
    engine = live.SmartEntryLiveInference(
        bundle_dir=tmp_path,
        operating_point={
            "selection_score": selection_mode,
            "max_trades": 3,
        },
    )
    # These unit tests exercise direction/runtime-envelope parity without
    # claiming executable capital authority. The exact learned-sizing schema
    # is still mandatory; its adoption event remains deliberately unverified.
    engine._sizing_authority = unverified_learned_sizing_authority()
    return engine


def _decision_head(
    direction_logits: tuple[float, float, float] = (5.0, 1.0, 0.0),
    public_logits: tuple[float, float] = (5.0, 0.0),
) -> dict:
    direction_probs = _softmax(direction_logits)
    public_probs = _softmax(public_logits)
    direction_index = int(np.argmax(np.asarray(direction_logits, dtype=np.float64)))
    public_index = int(np.argmax(np.asarray(public_logits, dtype=np.float64)))
    side_logits = [2.0, 0.0]
    side_bad_path_logits = [-2.0, 1.0]
    side_validity_logits = [2.5, -1.0]
    rail_logits = [1.0, -1.0, 0.5, -0.5, 0.25, -0.25]
    bad_path_logit = -1.0
    tradable_logit = 1.0
    tf_agreement_logit = 0.5
    position_size_logit = 0.25
    clean_edge_logit = 0.75
    survival_logit = 1.25
    return {
        "time": pd.Timestamp("2026-07-08T18:00:00Z"),
        "session_id": 0,  # ASIA is outside the legacy sessions=[US] gate.
        "entry_vol_regime_id": 3,
        "entry_atr_bucket": 4,
        "entry_spread_bucket": 1,
        "entry_h4_trend_sign_cat": 0,
        "entry_trend_regime_id": 2,
        "raw_direction_logits": list(direction_logits),
        "direction_logits": list(direction_logits),
        "direction_probs": direction_probs,
        "model_direction_index": direction_index,
        "model_direction": ("LONG", "SHORT", "FLAT")[direction_index],
        "entry_shared_representation": [
            float(index - 64) / 64.0 for index in range(128)
        ],
        "p_long": direction_probs[0],
        "p_short": direction_probs[1],
        "p_flat": direction_probs[2],
        "edge_score": max(direction_probs[0], direction_probs[1]) - direction_probs[2],
        "public_trade_flat_decision_logits": list(public_logits),
        "public_trade_flat_decision_probs": public_probs,
        "public_trade_flat_decision_index": public_index,
        "public_trade_flat_decision": ("TRADE", "FLAT")[public_index],
        "p_trade": public_probs[0],
        "p_flat_hier": public_probs[1],
        "model_native_logits": [0.4, -0.2, 0.1],
        "path_quality_raw": 1.5,
        "path_quality": 1.5,
        "path_quality_pred": 1.5,
        "mfe_first_n": 12.0,
        "bad_path_logit_raw": bad_path_logit,
        "bad_path_logit": bad_path_logit,
        "bad_path_prob": _sigmoid(bad_path_logit),
        "tradable_logit": tradable_logit,
        "tradable_prob": _sigmoid(tradable_logit),
        "mfe_first_n_pred": 12.0,
        "clean_edge_logit": clean_edge_logit,
        "clean_edge_prob": _sigmoid(clean_edge_logit),
        "survival_logit": survival_logit,
        "survival_prob": _sigmoid(survival_logit),
        "dip_pred": [float(value) / 10.0 for value in range(18)],
        "forecast_pred": [0.1, 0.2, 0.3, 0.4],
        "timing_pred": [float(value) / 20.0 for value in range(12)],
        "tail_risk_pred": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "vol_forecast_pred": [0.5, 0.75, 1.0],
        **offline_rl_evidence(),
        "specialist_names": list(live.MODEL_NATIVE_REQUIRED_SPECIALISTS),
        "specialist_gate": [1.0 / len(live.MODEL_NATIVE_REQUIRED_SPECIALISTS)]
        * len(live.MODEL_NATIVE_REQUIRED_SPECIALISTS),
        "tf_gate": [1.0 / ENTRY_MTF_CONTEXT_COUNT] * ENTRY_MTF_CONTEXT_COUNT,
        "family_tf_cooperation_gate": [1.0 / (ENTRY_MTF_CONTEXT_COUNT * len(live.MODEL_NATIVE_REQUIRED_SPECIALISTS))]
        * (ENTRY_MTF_CONTEXT_COUNT * len(live.MODEL_NATIVE_REQUIRED_SPECIALISTS)),
        "family_tf_feature_gate": [1.0] * (ENTRY_MTF_CONTEXT_COUNT * live.MULTI_TF_FEATURE_COUNT_V4),
        "p_long_given_trade": _softmax(side_logits)[0],
        "p_short_given_trade": _softmax(side_logits)[1],
        "side_logits": side_logits,
        "trade_logit": tradable_logit,
        "side_probs": _softmax(side_logits),
        "side_utility": [12.0, -4.0],
        "side_bad_path_logit": side_bad_path_logits,
        "long_bad_path_prob": _sigmoid(side_bad_path_logits[0]),
        "short_bad_path_prob": _sigmoid(side_bad_path_logits[1]),
        "side_validity_logit": side_validity_logits,
        "long_validity_prob": _sigmoid(side_validity_logits[0]),
        "short_validity_prob": _sigmoid(side_validity_logits[1]),
        "side_mae": [-3.0, -8.0],
        "mtf_dir_logits": [2.0, 0.0, -1.0],
        "mtf_dir_probs": _softmax((2.0, 0.0, -1.0)),
        "geometry_channel_edge_pressure": 0.4,
        "geometry_rising_support_rail_long_pressure": 0.7,
        "geometry_rising_support_rail_short_trap_pressure": 0.2,
        "geometry_falling_resistance_rail_short_pressure": 0.1,
        "geometry_falling_resistance_rail_long_trap_pressure": 0.05,
        "trendline_rail_logits": rail_logits,
        "trendline_rail_probs": [_sigmoid(value) for value in rail_logits],
        "mtf_trend_evidence": 0.65,
        "calibration_version": "test-v1",
        "direction_calibration_enabled": True,
        "direction_calibration_temperature": 1.0,
        "path_calibration_enabled": True,
        "path_calibration": {
            "enabled": True,
            "version": "test-v1",
            "path_quality_scale": 1.0,
            "path_quality_shift": 0.0,
            "bad_path_temperature": 1.0,
            "bad_path_bias": 0.0,
        },
        "tf_agreement_logit": tf_agreement_logit,
        "tf_agreement_pred": _sigmoid(tf_agreement_logit),
        "path_quality_log_var": 0.0,
        "path_quality_std": 1.0,
        "position_size_logit": position_size_logit,
        "position_size_pred": _sigmoid(position_size_logit),
    }


@pytest.mark.parametrize(
    ("direction_logits", "public_logits", "expected_direction", "expected_action", "expected_side"),
    [
        ((5.0, 1.0, 0.0), (5.0, 0.0), "LONG", "TAKE_LONG_NOW", 0),
        ((1.0, 5.0, 0.0), (5.0, 0.0), "SHORT", "TAKE_SHORT_NOW", 1),
        ((0.0, 1.0, 5.0), (1.0, 5.0), "FLAT", "SKIP", None),
    ],
)
def test_smart_decision_follows_final_model_argmax_exactly(
    tmp_path: Path,
    direction_logits: tuple[float, float, float],
    public_logits: tuple[float, float],
    expected_direction: str,
    expected_action: str,
    expected_side: int | None,
) -> None:
    engine = _decision_engine(tmp_path)

    decision = engine.decide(
        _decision_head(direction_logits=direction_logits, public_logits=public_logits),
        atr_bps=9.0,
    )
    snapshot = decision["_v10_snapshot"]

    assert decision["action"] == expected_action
    assert decision["model_direction"] == expected_direction
    assert decision["selected_side"] == expected_side
    assert decision["selection_score_mode"] == live.MODEL_DIRECTION_SELECTION_MODE
    assert "selection_score_threshold" not in decision
    assert decision["session"] == "ASIA"
    assert "smart_skip_reason" not in decision
    assert decision["policy"] == MODEL_NATIVE_RUNTIME_POLICY
    assert set(snapshot) == set(MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS)
    assert snapshot["runtime_evidence_schema_version"] == (
        MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION
    )
    assert snapshot["model_policy"] == decision["policy"]
    assert snapshot["session_id"] == 0
    assert snapshot["session"] == decision["session"]
    assert snapshot["model_direction"] == expected_direction
    assert snapshot["selected_side"] == expected_side
    assert decision["side_utility"] == [12.0, -4.0]
    assert decision["trendline_rail_logits"] == [1.0, -1.0, 0.5, -0.5, 0.25, -0.25]
    assert not any(key.startswith("expected_utility") for key in decision)
    assert not any(key.startswith("expected_utility") for key in snapshot)


def test_actual_smart_decision_keyset_forms_exact_executable_pipeline_envelope(
    tmp_path: Path,
) -> None:
    head = _decision_head()
    head.update(
        {
            "context_age_m5_bars": 0,
            "context_cutoff_ts": "2026-07-08T18:00:00+00:00",
            "context_refresh_in_flight": False,
            "context_mtf_incremental": False,
        }
    )
    decision = _decision_engine(tmp_path).decide(head, atr_bps=9.0)
    timing = {
        "decision_available_ts": "2026-07-08T18:05:00+00:00",
        "entry_signal_latency_sec": 30.0,
        "context_cutoff_ts": decision["context_cutoff_ts"],
        "context_age_m5_bars": decision["context_age_m5_bars"],
    }
    decision["_v10_snapshot"] = {**decision["_v10_snapshot"], **timing}
    decision.update(
        {
            "decision_available_ts": timing["decision_available_ts"],
            "entry_signal_latency_sec": timing["entry_signal_latency_sec"],
            "entry_signal_latency_min": 0.5,
                "entry_signal_latency_cap_sec": 90.0,
                "entry_signal_stale": False,
                "entry_source_pair_generation_id": "1" * 64,
                "entry_source_pair_manifest_sha256": "2" * 64,
            }
        )

    assert set(decision) == set(MODEL_NATIVE_EXECUTABLE_DECISION_REQUIRED_FIELDS)
    assert require_executable_model_native_entry_decision(
        decision,
        "2026-07-08T18:05:00+00:00",
    ) == decision["_v10_snapshot"]


@pytest.mark.parametrize(
    "retired_key",
    [
        "expected_utility_long_bps",
        "expected_utility_short_bps",
        "expected_utility_side",
        "selection_score_threshold",
        "q_take_long",
    ],
)
def test_smart_decision_rejects_retired_live_overlay_fields(
    tmp_path: Path,
    retired_key: str,
) -> None:
    head = _decision_head()
    head[retired_key] = 0.0

    with pytest.raises(RuntimeError, match="retired live overlay fields are forbidden"):
        _decision_engine(tmp_path).decide(head, atr_bps=9.0)


def test_smart_decision_rejects_unknown_head_overlay_even_with_new_name(
    tmp_path: Path,
) -> None:
    head = _decision_head()
    head["trend_veto"] = False

    with pytest.raises(
        RuntimeError,
        match=r"decision head exact schema mismatch:.*trend_veto",
    ):
        _decision_engine(tmp_path).decide(head, atr_bps=9.0)


def test_smart_decision_rejects_partial_context_evidence_envelope(
    tmp_path: Path,
) -> None:
    head = _decision_head()
    head["context_age_m5_bars"] = 0

    with pytest.raises(RuntimeError, match="decision head exact schema mismatch"):
        _decision_engine(tmp_path).decide(head, atr_bps=9.0)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("model_direction", "SHORT"),
        ("model_direction_index", 1),
        ("public_trade_flat_decision", "FLAT"),
        ("public_trade_flat_decision_index", 1),
        ("p_trade", 0.1),
        ("edge_score", -100.0),
    ],
)
def test_smart_decision_rejects_reported_ssot_metadata_mismatch(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    head = _decision_head()
    head[field] = replacement

    with pytest.raises(RuntimeError, match="decision SSOT"):
        _decision_engine(tmp_path).decide(head, atr_bps=9.0)


@pytest.mark.parametrize("selection_mode", ["expected_utility", "edge_score", "unknown", "MODEL_DIRECTION_ARGMAX"])
def test_smart_decision_rejects_old_unknown_or_nonexact_selection_mode(
    tmp_path: Path,
    selection_mode: str,
) -> None:
    with pytest.raises(RuntimeError, match="selection_score must be exactly"):
        _decision_engine(tmp_path, selection_mode=selection_mode)


@pytest.mark.parametrize(
    "stale_key",
    ["edge_score_threshold", "expected_utility_threshold_bps", "sessions"],
)
def test_smart_entry_rejects_stale_runtime_direction_operating_point_keys(
    tmp_path: Path,
    stale_key: str,
) -> None:
    operating_point = {
        "selection_score": live.MODEL_DIRECTION_SELECTION_MODE,
        "max_trades": 3,
        stale_key: ["US"] if stale_key == "sessions" else 999.0,
    }

    with pytest.raises(RuntimeError, match="operating_point contract mismatch"):
        live.SmartEntryLiveInference(
            bundle_dir=tmp_path,
            operating_point=operating_point,
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "direction_logits",
        "direction_probs",
        "public_trade_flat_decision_logits",
        "public_trade_flat_decision_probs",
    ],
)
def test_smart_decision_rejects_missing_direction_ssot(
    tmp_path: Path,
    missing_field: str,
) -> None:
    engine = _decision_engine(tmp_path)
    head = _decision_head()
    del head[missing_field]

    with pytest.raises(RuntimeError, match=missing_field):
        engine.decide(head, atr_bps=9.0)


@pytest.mark.parametrize(
    "field",
    [
        "direction_logits",
        "direction_probs",
        "public_trade_flat_decision_logits",
        "public_trade_flat_decision_probs",
    ],
)
def test_smart_decision_rejects_nonfinite_direction_ssot(tmp_path: Path, field: str) -> None:
    engine = _decision_engine(tmp_path)
    head = _decision_head()
    head[field][0] = float("nan")

    with pytest.raises(RuntimeError, match="non-finite"):
        engine.decide(head, atr_bps=9.0)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("direction_probs", [0.1, 0.8, 0.1], "direction_probs do not match"),
        (
            "public_trade_flat_decision_probs",
            [0.1, 0.9],
            "public_trade_flat_decision_probs do not match",
        ),
    ],
)
def test_smart_decision_rejects_logits_probability_mismatch(
    tmp_path: Path,
    field: str,
    replacement: list[float],
    error: str,
) -> None:
    engine = _decision_engine(tmp_path)
    head = _decision_head()
    head[field] = replacement

    with pytest.raises(RuntimeError, match=error):
        engine.decide(head, atr_bps=9.0)


def test_smart_decision_rejects_trade_flat_vs_three_class_surface_mismatch(tmp_path: Path) -> None:
    engine = _decision_engine(tmp_path)
    head = _decision_head(direction_logits=(5.0, 1.0, 0.0), public_logits=(0.0, 3.0))

    with pytest.raises(RuntimeError, match="not the canonical"):
        engine.decide(head, atr_bps=9.0)


def test_smart_decision_rejects_tied_top_direction_logits(tmp_path: Path) -> None:
    engine = _decision_engine(tmp_path)
    head = _decision_head(
        direction_logits=(2.0, 2.0, 0.0),
        public_logits=(2.0, 0.0),
    )

    with pytest.raises(RuntimeError, match="no unique top class"):
        engine.decide(head, atr_bps=9.0)


def test_smart_decision_rejects_noncanonical_pair_even_when_argmax_matches(tmp_path: Path) -> None:
    engine = _decision_engine(tmp_path)
    head = _decision_head(direction_logits=(5.0, 1.0, 0.0), public_logits=(4.0, 0.0))

    with pytest.raises(RuntimeError, match="not the canonical"):
        engine.decide(head, atr_bps=9.0)


@pytest.mark.parametrize(
    "missing_field",
    [
        "path_quality_pred",
        "bad_path_logit",
        "tradable_logit",
        "clean_edge_logit",
        "survival_logit",
        "dip_pred",
        "forecast_pred",
        "timing_pred",
        "tail_risk_pred",
        "vol_forecast_pred",
        "specialist_gate",
        "side_utility",
        "mtf_dir_logits",
        "trendline_rail_logits",
        "tf_agreement_logit",
        "path_quality_log_var",
        "position_size_logit",
    ],
)
def test_smart_decision_requires_complete_model_native_evidence_surface(
    tmp_path: Path,
    missing_field: str,
) -> None:
    engine = _decision_engine(tmp_path)
    head = _decision_head()
    del head[missing_field]

    with pytest.raises(RuntimeError, match="decision head exact schema mismatch"):
        engine.decide(head, atr_bps=9.0)


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("bad_path_prob", 0.99, "bad_path_prob does not match"),
        ("tradable_prob", 0.01, "tradable_prob does not match"),
        ("clean_edge_prob", 0.01, "clean_edge_prob does not match"),
        ("tf_agreement_pred", 0.01, "tf_agreement_pred does not match"),
        ("position_size_pred", 0.01, "position_size_pred does not match"),
        ("specialist_gate", [1.0] * 8, "not a probability simplex"),
    ],
)
def test_smart_decision_rejects_inconsistent_learned_diagnostics(
    tmp_path: Path,
    field: str,
    replacement: object,
    error: str,
) -> None:
    engine = _decision_engine(tmp_path)
    head = _decision_head()
    head[field] = replacement

    with pytest.raises(RuntimeError, match=error):
        engine.decide(head, atr_bps=9.0)


@pytest.mark.parametrize("session_id", [-1, 4, 1.5, float("nan")])
def test_smart_decision_rejects_invalid_session_evidence(
    tmp_path: Path,
    session_id: float,
) -> None:
    engine = _decision_engine(tmp_path)
    head = _decision_head()
    head["session_id"] = session_id

    with pytest.raises(RuntimeError, match="session_id"):
        engine.decide(head, atr_bps=9.0)


@pytest.mark.parametrize("atr_bps", [0.0, -1.0, float("nan"), float("inf")])
def test_smart_decision_rejects_nonpositive_or_nonfinite_atr_evidence(
    tmp_path: Path,
    atr_bps: float,
) -> None:
    with pytest.raises(RuntimeError, match="finite and positive"):
        _decision_engine(tmp_path).decide(_decision_head(), atr_bps=atr_bps)


def _forward_states() -> dict:
    evidence_names = [
        "chart.geometry_channel_edge_pressure",
        "chart.geometry_rising_support_rail_long_pressure",
        "chart.geometry_rising_support_rail_short_trap_pressure",
        "chart.geometry_falling_resistance_rail_short_pressure",
        "chart.geometry_falling_resistance_rail_long_trap_pressure",
        "trend.mtf_confluence_trend_direction_score",
    ]
    return {
        "times": [pd.Timestamp("2026-07-08T18:00:00Z")],
        "seq": np.zeros((1, 2, 1), dtype=np.float32),
        "snap": np.asarray([[0.4, 0.7, 0.2, 0.1, 0.05, 0.65]], dtype=np.float32),
        "ctx_cont": np.zeros((1, 1), dtype=np.float32),
        "ctx_cat": np.asarray([[0, 3, 4, 1, 0]], dtype=np.int64),
        "entry_trend_regime_id": np.asarray([2], dtype=np.int64),
        "_evidence_names": evidence_names,
    }


def _forward_outputs() -> dict:
    return {
        "raw_direction_logits": torch.tensor([[5.0, 1.0, 0.0]], dtype=torch.float32),
        "direction_logits": torch.tensor([[5.0, 1.0, 0.0]], dtype=torch.float32),
        "shared_feature_representation": torch.arange(
            128, dtype=torch.float32
        ).reshape(1, 128),
        "public_trade_flat_decision_logits": torch.tensor([[5.0, 0.0]], dtype=torch.float32),
        "model_native_logits": torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float32),
        "mtf_dir_logits": torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float32),
        "side_logits": torch.tensor([[2.0, 0.0]], dtype=torch.float32),
        "side_utility": torch.tensor([[12.0, -4.0]], dtype=torch.float32),
        "side_bad_path_logit": torch.tensor([[-2.0, 1.0]], dtype=torch.float32),
        "side_mae": torch.tensor([[-3.0, -8.0]], dtype=torch.float32),
        "side_validity_logit": torch.tensor([[2.5, -1.0]], dtype=torch.float32),
        "trendline_rail_logits": torch.tensor(
            [[1.0, -1.0, 0.5, -0.5, 0.25, -0.25]], dtype=torch.float32
        ),
        "path_quality": torch.tensor([[1.5]], dtype=torch.float32),
        "path_quality_raw": torch.tensor([[1.5]], dtype=torch.float32),
        "bad_path_logit": torch.tensor([[-1.0]], dtype=torch.float32),
        "bad_path_logit_raw": torch.tensor([[-1.0]], dtype=torch.float32),
        "tradable_logit": torch.tensor([[1.0]], dtype=torch.float32),
        "trade_logit": torch.tensor([[1.0]], dtype=torch.float32),
        "mfe_first_n": torch.tensor([[12.0]], dtype=torch.float32),
        "clean_edge_logit": torch.tensor([[0.75]], dtype=torch.float32),
        "survival_logit": torch.tensor([[1.25]], dtype=torch.float32),
        "dip_pred": torch.arange(18, dtype=torch.float32).reshape(1, 18) / 10.0,
        "forecast_pred": torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32),
        "timing_pred": torch.arange(12, dtype=torch.float32).reshape(1, 12) / 20.0,
        "tail_risk_pred": torch.tensor(
            [[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]], dtype=torch.float32
        ),
        "vol_forecast_pred": torch.tensor([[0.5, 0.75, 1.0]], dtype=torch.float32),
        "specialist_gate": torch.full(
            (1, len(live.MODEL_NATIVE_REQUIRED_SPECIALISTS)),
            1.0 / len(live.MODEL_NATIVE_REQUIRED_SPECIALISTS),
            dtype=torch.float32,
        ),
        "tf_gate": torch.full(
            (1, ENTRY_MTF_CONTEXT_COUNT),
            1.0 / ENTRY_MTF_CONTEXT_COUNT,
            dtype=torch.float32,
        ),
        "family_tf_cooperation_gate": torch.full(
            (1, ENTRY_MTF_CONTEXT_COUNT * len(live.MODEL_NATIVE_REQUIRED_SPECIALISTS)),
            1.0 / (ENTRY_MTF_CONTEXT_COUNT * len(live.MODEL_NATIVE_REQUIRED_SPECIALISTS)),
            dtype=torch.float32,
        ),
        "family_tf_feature_gate": torch.ones(
            (
                1,
                ENTRY_MTF_CONTEXT_COUNT,
                live.MULTI_TF_FEATURE_COUNT_V4,
            ),
            dtype=torch.float32,
        ),
        "tf_agreement_logit": torch.tensor([[0.5]], dtype=torch.float32),
        "path_quality_log_var": torch.tensor([[0.0]], dtype=torch.float32),
        "position_size_logit": torch.tensor([[0.25]], dtype=torch.float32),
        **{
            name: torch.tensor([values], dtype=torch.float32)
            for name, values in offline_rl_evidence().items()
        },
    }


def _prepare_forward_engine(tmp_path: Path, outputs: dict) -> live.SmartEntryLiveInference:
    engine = _decision_engine(tmp_path)
    states = _forward_states()
    engine._meta = {
        "ordered_signal_names": states["_evidence_names"],
        "direction_calibration": {
            "enabled": True,
            "version": "test-v1",
            "temperature": 1.0,
            "bias": [0.0, 0.0, 0.0],
        },
        "path_calibration": {
            "enabled": True,
            "version": "test-v1",
            "path_quality_scale": 1.0,
            "path_quality_shift": 0.0,
            "bad_path_temperature": 1.0,
            "bad_path_bias": 0.0,
        },
        "specialist_fusion": {
            "trainable_specialists": list(live.MODEL_NATIVE_REQUIRED_SPECIALISTS),
        },
        "multi_tf": {
            "feature_names": list(range(live.MULTI_TF_FEATURE_COUNT_V4)),
        },
    }
    engine._model = lambda *args, **kwargs: outputs
    engine._multi_tf_window_tensors = lambda *args, **kwargs: {}
    return engine


def test_forward_states_requires_model_public_trade_flat_ssot(tmp_path: Path) -> None:
    outputs = _forward_outputs()
    del outputs["public_trade_flat_decision_logits"]
    engine = _prepare_forward_engine(tmp_path, outputs)

    with pytest.raises(RuntimeError, match="missing required SSOT 'public_trade_flat_decision_logits'"):
        engine.forward_states(_forward_states())


def test_forward_states_requires_and_reports_full_model_native_evidence(tmp_path: Path) -> None:
    engine = _prepare_forward_engine(tmp_path, _forward_outputs())

    head = engine.forward_states(_forward_states())[0]

    assert set(head) == set(live.MODEL_NATIVE_DECISION_HEAD_REQUIRED_FIELDS)
    assert head["model_direction"] == "LONG"
    assert head["public_trade_flat_decision"] == "TRADE"
    assert len(head["mtf_dir_logits"]) == 3
    assert len(head["side_probs"]) == 2
    assert head["side_utility"] == [12.0, -4.0]
    assert len(head["side_bad_path_logit"]) == 2
    assert len(head["side_validity_logit"]) == 2
    assert len(head["side_mae"]) == 2
    assert len(head["trendline_rail_logits"]) == 6
    assert len(head["dip_pred"]) == 18
    assert len(head["forecast_pred"]) == 4
    assert len(head["timing_pred"]) == 12
    assert len(head["tail_risk_pred"]) == 6
    assert len(head["vol_forecast_pred"]) == 3
    assert head["specialist_names"] == list(live.MODEL_NATIVE_REQUIRED_SPECIALISTS)
    assert sum(head["specialist_gate"]) == pytest.approx(1.0)
    assert head["mtf_trend_evidence"] == pytest.approx(0.65)
    assert not any(key.startswith("expected_utility") for key in head)
    assert 0.0 < head["tf_agreement_pred"] < 1.0
    assert head["path_quality_std"] == pytest.approx(1.0)
    assert 0.0 < head["position_size_pred"] < 1.0


@pytest.mark.parametrize(
    "missing_head",
    [
        "mtf_dir_logits",
        "side_logits",
        "side_utility",
        "side_bad_path_logit",
        "side_mae",
        "side_validity_logit",
        "trendline_rail_logits",
        "clean_edge_logit",
        "survival_logit",
        "dip_pred",
        "forecast_pred",
        "timing_pred",
        "tail_risk_pred",
            "vol_forecast_pred",
            "action_value",
            "expectile_value",
            "action_advantage",
            "specialist_gate",
        "tf_agreement_logit",
        "path_quality_log_var",
    ],
)
def test_forward_states_fails_closed_on_missing_model_native_evidence_head(
    tmp_path: Path,
    missing_head: str,
) -> None:
    outputs = _forward_outputs()
    del outputs[missing_head]
    engine = _prepare_forward_engine(tmp_path, outputs)

    with pytest.raises(RuntimeError, match=rf"missing required SSOT '{missing_head}'"):
        engine.forward_states(_forward_states())


def test_forward_states_requires_learned_sizing_evidence_head(tmp_path: Path) -> None:
    outputs = _forward_outputs()
    del outputs["position_size_logit"]
    engine = _prepare_forward_engine(tmp_path, outputs)

    with pytest.raises(RuntimeError, match="missing required SSOT 'position_size_logit'"):
        engine.forward_states(_forward_states())
