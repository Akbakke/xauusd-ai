from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.execution import v12_smart_entry_live as live
from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_MTF_CONTEXT_COUNT,
    ENTRY_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
    MODEL_NATIVE_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
    MODEL_NATIVE_RUNTIME_EVIDENCE_SCHEMA_VERSION,
    MODEL_NATIVE_RUNTIME_POLICY,
)
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


def test_smart_entry_requires_route_specific_bundle_mtf_availability_shifts() -> None:
    assert live._require_bundle_mtf_availability_shifts(
        {
            "entry_target_availability_shift_minutes": 5.0,
            "exit_target_availability_shift_minutes": 1.0,
        }
    ) == (5.0, 1.0)

    with pytest.raises(RuntimeError, match="route-specific"):
        live._require_bundle_mtf_availability_shifts(
            {"target_availability_shift_minutes": 5.0}
        )

    with pytest.raises(RuntimeError, match="invalid"):
        live._require_bundle_mtf_availability_shifts(
            {
                "entry_target_availability_shift_minutes": 1.0,
                "exit_target_availability_shift_minutes": 5.0,
            }
        )


def test_smart_entry_rejects_ambient_or_mismatched_mtf_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = Path("/tmp/gx1-bound-v4-cache")
    binding = {
        "shared_cache_dir": str(cache_dir),
        "shared_cache_manifest_path": str(cache_dir / "manifest.json"),
        "shared_cache_identity_sha256": "a" * 64,
        "shared_cache_manifest_sha256": "b" * 64,
        "shared_cache_m5_source": "/tmp/gx1-v46-m5.parquet",
        "shared_cache_m5_source_sha256": "c" * 64,
    }
    fake_cache = SimpleNamespace(
        cache_identity_sha256="a" * 64,
        manifest_sha256="b" * 64,
        m5_prebuilt_source="/tmp/gx1-v46-m5.parquet",
        m5_prebuilt_source_sha256="c" * 64,
    )
    seen: list[Path] = []
    monkeypatch.setattr(
        live,
        "load_multi_tf_v4_cache",
        lambda path: seen.append(Path(path)) or fake_cache,
    )

    assert live._load_and_require_bundle_mtf_cache(binding) is fake_cache
    assert seen == [cache_dir]

    fake_cache.cache_identity_sha256 = "d" * 64
    with pytest.raises(RuntimeError, match="identity mismatch"):
        live._load_and_require_bundle_mtf_cache(binding)


def test_smart_entry_rejects_context_from_same_timestamp_different_pair() -> None:
    current_pair = SimpleNamespace(
        pair_generation_id="a" * 64,
        pair_manifest_sha256="b" * 64,
    )
    ctx = SimpleNamespace(
        pair_generation_id="a" * 64,
        pair_manifest_sha256="b" * 64,
    )
    live._require_context_pair_identity(ctx, current_pair)

    replacement_pair = SimpleNamespace(
        pair_generation_id="c" * 64,
        pair_manifest_sha256="d" * 64,
    )
    with pytest.raises(live.SmartContextPairMismatchError, match="different immutable"):
        live._require_context_pair_identity(ctx, replacement_pair)


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
    engine._per_tf_seq_lens = {
        timeframe: 1 for timeframe in ENTRY_MTF_CONTEXT_TIMEFRAMES
    }
    engine._multi_tf_shift = {
        timeframe: pd.Timedelta(minutes=5)
        for timeframe in ENTRY_MTF_CONTEXT_TIMEFRAMES
    }

    out = engine._multi_tf_window_tensors(
        pd.Timestamp("2026-07-08T12:05:00Z"),
        multi_tf={
            timeframe: frame for timeframe in ENTRY_MTF_CONTEXT_TIMEFRAMES
        },
    )

    assert set(out) == {
        f"seq_{timeframe.lower()}"
        for timeframe in ENTRY_MTF_CONTEXT_TIMEFRAMES
    }
    assert "seq_m5" not in out
    assert float(out["seq_m15"][0, 0, 0].item()) == 2.0


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
    entry_action_q_bps: tuple[float, float, float] = (12.0, -3.0, 0.0),
) -> dict:
    """One complete decision head at the exact Fitted-Q contract.

    Direction is the unique argmax of raw ``entry_action_q_bps``; the retired
    calibrated-probability, hierarchy, utility and rail surfaces are forbidden
    live overlay fields and must not appear here.
    """

    q_values = np.asarray(entry_action_q_bps, dtype=np.float64)
    direction_index = int(np.argmax(q_values))
    ordered = np.sort(q_values)
    position_size_logit = 0.25
    specialists = list(live.MODEL_NATIVE_REQUIRED_SPECIALISTS)
    cooperation_width = ENTRY_MTF_CONTEXT_COUNT * len(specialists)
    feature_width = ENTRY_MTF_CONTEXT_COUNT * live.MULTI_TF_FEATURE_COUNT_V4
    return {
        "time": pd.Timestamp("2026-07-08T18:00:00Z"),
        "session_id": 0,  # ASIA is outside the legacy sessions=[US] gate.
        "entry_action_q_bps": list(entry_action_q_bps),
        "entry_action_q_margin_bps": float(ordered[-1] - ordered[-2]),
        "model_direction_index": direction_index,
        "model_direction": ("LONG", "SHORT", "FLAT")[direction_index],
        "entry_decision_representation": [
            float(index - 64) / 64.0 for index in range(128)
        ],
        "entry_q_joint_hidden": [0.02] * 128,
        "side_mae_bps": [3.0, 8.0],
        "trendline_event_logits": [0.1, -0.1, 0.2, -0.2],
        "dip_pred": [float(value) / 10.0 for value in range(18)],
        "forecast_pred": [0.1, 0.2, 0.3, 0.4],
        "timing_pred": [float(value) / 20.0 for value in range(12)],
        "tail_risk_pred": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "vol_forecast_pred": [0.5, 0.75, 1.0],
        "specialist_names": specialists,
        "specialist_gate": [1.0 / len(specialists)] * len(specialists),
        "tf_gate": [1.0 / ENTRY_MTF_CONTEXT_COUNT] * ENTRY_MTF_CONTEXT_COUNT,
        "family_tf_cooperation_gate": [1.0 / cooperation_width] * cooperation_width,
        "family_tf_feature_gate": [1.0] * feature_width,
        "position_size_logit": position_size_logit,
        "position_size_pred": _sigmoid(position_size_logit),
    }


@pytest.mark.parametrize(
    ("entry_action_q_bps", "expected_direction", "expected_action", "expected_side"),
    [
        ((12.0, -3.0, 0.0), "LONG", "TAKE_LONG_NOW", 0),
        ((-4.0, 9.5, 0.0), "SHORT", "TAKE_SHORT_NOW", 1),
        ((-2.0, -6.0, 0.0), "FLAT", "SKIP", None),
    ],
)
def test_smart_decision_follows_final_model_argmax_exactly(
    tmp_path: Path,
    entry_action_q_bps: tuple[float, float, float],
    expected_direction: str,
    expected_action: str,
    expected_side: int | None,
) -> None:
    engine = _decision_engine(tmp_path)

    decision = engine.decide(
        _decision_head(entry_action_q_bps=entry_action_q_bps),
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
    assert decision["entry_action_q_bps"] == list(entry_action_q_bps)
    assert decision["side_mae_bps"] == [3.0, 8.0]
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
    "fragment",
    sorted(live.RETIRED_RUNTIME_EVIDENCE_FRAGMENTS),
)
def test_smart_decision_rejects_retired_live_overlay_fields(
    tmp_path: Path,
    fragment: str,
) -> None:
    """Every retired evidence fragment is refused by name, not by luck.

    Parametrized from the runtime-evidence owner's own retired-fragment set,
    so a fragment added there is covered immediately.
    """

    head = _decision_head()
    head[f"{fragment}_probe"] = 0.0

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
        ("entry_action_q_margin_bps", 0.5),
    ],
)
def test_smart_decision_rejects_reported_ssot_metadata_mismatch(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    head = _decision_head()
    head[field] = replacement

    with pytest.raises(RuntimeError, match="mismatches raw"):
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


def test_smart_decision_rejects_missing_direction_ssot(tmp_path: Path) -> None:
    engine = _decision_engine(tmp_path)
    head = _decision_head()
    del head["entry_action_q_bps"]

    with pytest.raises(RuntimeError, match="entry_action_q_bps"):
        engine.decide(head, atr_bps=9.0)


def test_smart_decision_rejects_nonfinite_direction_ssot(tmp_path: Path) -> None:
    engine = _decision_engine(tmp_path)
    head = _decision_head()
    head["entry_action_q_bps"][0] = float("nan")

    with pytest.raises(RuntimeError, match="non-finite"):
        engine.decide(head, atr_bps=9.0)


def test_smart_decision_rejects_tied_top_action_values(tmp_path: Path) -> None:
    engine = _decision_engine(tmp_path)
    head = _decision_head(entry_action_q_bps=(5.0, 5.0, 0.0))

    with pytest.raises(RuntimeError, match="no unique top action"):
        engine.decide(head, atr_bps=9.0)


@pytest.mark.parametrize(
    "missing_field",
    sorted(live.MODEL_NATIVE_DECISION_HEAD_REQUIRED_FIELDS),
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
        ("position_size_pred", 0.01, "position_size_pred does not match"),
        (
            "specialist_gate",
            [1.0] * len(live.MODEL_NATIVE_REQUIRED_SPECIALISTS),
            "not a probability simplex",
        ),
        (
            "tf_gate",
            [1.0] * ENTRY_MTF_CONTEXT_COUNT,
            "not a probability simplex",
        ),
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
    evidence_names = ["chart.local_ema50_200_spread_atr"]
    return {
        "times": [pd.Timestamp("2026-07-08T18:00:00Z")],
        "seq": np.zeros((1, 2, 1), dtype=np.float32),
        "snap": np.asarray([[0.65]], dtype=np.float32),
        "ctx_cont": np.zeros((1, 1), dtype=np.float32),
        "ctx_cat": np.zeros(
            (1, len(live.MODEL_NATIVE_CTX_CAT_FIELDS)), dtype=np.int64
        ),
        "_evidence_names": evidence_names,
    }


def _forward_outputs() -> dict:
    """The exact head tensors the current model forward must emit."""

    specialists = list(live.MODEL_NATIVE_REQUIRED_SPECIALISTS)
    cooperation_width = ENTRY_MTF_CONTEXT_COUNT * len(specialists)
    return {
        "entry_action_q_bps": torch.tensor(
            [[12.0, -3.0, 0.0]], dtype=torch.float32
        ),
        "entry_q_joint_hidden": torch.full((1, 128), 0.02, dtype=torch.float32),
        "entry_decision_representation": torch.arange(
            128, dtype=torch.float32
        ).reshape(1, 128),
        "side_mae_bps": torch.tensor([[3.0, 8.0]], dtype=torch.float32),
        "trendline_event_logits": torch.tensor(
            [[0.1, -0.1, 0.2, -0.2]], dtype=torch.float32
        ),
        "dip_pred": torch.arange(18, dtype=torch.float32).reshape(1, 18) / 10.0,
        "forecast_pred": torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32),
        "timing_pred": torch.arange(12, dtype=torch.float32).reshape(1, 12) / 20.0,
        "tail_risk_pred": torch.tensor(
            [[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]], dtype=torch.float32
        ),
        "vol_forecast_pred": torch.tensor([[0.5, 0.75, 1.0]], dtype=torch.float32),
        "specialist_gate": torch.full(
            (1, len(specialists)),
            1.0 / len(specialists),
            dtype=torch.float32,
        ),
        "tf_gate": torch.full(
            (1, ENTRY_MTF_CONTEXT_COUNT),
            1.0 / ENTRY_MTF_CONTEXT_COUNT,
            dtype=torch.float32,
        ),
        "family_tf_cooperation_gate": torch.full(
            (1, cooperation_width),
            1.0 / cooperation_width,
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
        "position_size_logit": torch.tensor([[0.25]], dtype=torch.float32),
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


def test_forward_states_requires_model_entry_q_ssot(tmp_path: Path) -> None:
    outputs = _forward_outputs()
    del outputs["entry_action_q_bps"]
    engine = _prepare_forward_engine(tmp_path, outputs)

    with pytest.raises(
        RuntimeError, match="missing required SSOT 'entry_action_q_bps'"
    ):
        engine.forward_states(_forward_states())


def test_forward_states_requires_and_reports_full_model_native_evidence(tmp_path: Path) -> None:
    engine = _prepare_forward_engine(tmp_path, _forward_outputs())

    head = engine.forward_states(_forward_states())[0]

    assert set(head) == set(live.MODEL_NATIVE_DECISION_HEAD_REQUIRED_FIELDS)
    assert head["model_direction"] == "LONG"
    assert head["entry_action_q_bps"] == [12.0, -3.0, 0.0]
    assert head["entry_action_q_margin_bps"] == pytest.approx(12.0)
    assert len(head["entry_decision_representation"]) == 128
    assert len(head["entry_q_joint_hidden"]) == 128
    assert len(head["side_mae_bps"]) == 2
    assert len(head["trendline_event_logits"]) == 4
    assert len(head["dip_pred"]) == 18
    assert len(head["forecast_pred"]) == 4
    assert len(head["timing_pred"]) == 12
    assert len(head["tail_risk_pred"]) == 6
    assert len(head["vol_forecast_pred"]) == 3
    assert head["specialist_names"] == list(live.MODEL_NATIVE_REQUIRED_SPECIALISTS)
    assert sum(head["specialist_gate"]) == pytest.approx(1.0)
    assert sum(head["tf_gate"]) == pytest.approx(1.0)
    assert not any(key.startswith("expected_utility") for key in head)
    assert 0.0 < head["position_size_pred"] < 1.0


@pytest.mark.parametrize("missing_head", sorted(_forward_outputs()))
def test_forward_states_fails_closed_on_missing_model_native_evidence_head(
    tmp_path: Path,
    missing_head: str,
) -> None:
    """Every tensor the forward owner reads must fail closed when absent.

    Parametrized from the fixture's own key set, which mirrors exactly the
    tensors ``forward_states`` requires, so a newly required head cannot be
    added without a matching fail-closed case.
    """

    outputs = _forward_outputs()
    del outputs[missing_head]
    engine = _prepare_forward_engine(tmp_path, outputs)

    with pytest.raises(RuntimeError, match=re.escape(missing_head)):
        engine.forward_states(_forward_states())


def test_forward_states_requires_learned_sizing_evidence_head(tmp_path: Path) -> None:
    outputs = _forward_outputs()
    del outputs["position_size_logit"]
    engine = _prepare_forward_engine(tmp_path, outputs)

    with pytest.raises(RuntimeError, match="missing required SSOT 'position_size_logit'"):
        engine.forward_states(_forward_states())
